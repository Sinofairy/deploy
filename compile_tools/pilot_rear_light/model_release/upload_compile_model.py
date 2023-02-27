import argparse
import copy
import json
import logging
import os
import subprocess
import sys
import time

import horizon_plugin_pytorch as horizon
import torch
from aidisdk import AIDIClient
from easydict import EasyDict

from hat.registry import RegistryContext
from hat.utils.apply_func import _as_list
from hat.utils.config import Config
from hat.utils.logger import DisableLogger

sys.path.append(os.path.join(os.path.dirname(__file__), "../../../"))
from tools.trainer_wrapper import TrainerWrapper  # noqa: E402

logger = logging.getLogger()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pub-config", type=str, required=True)
    parser.add_argument(
        "--compile-mode",
        type=str,
        default="local",
        help="chose compile in aidi or local, default is local.",
    )
    args = parser.parse_args()
    return args


def update_descs(model_dict, config):
    """
    Update thresholds of models according to input arguments through
    modifying configs.
    """

    if "task_modules" not in model_dict:
        return
    for task, value in model_dict["task_modules"].items():
        if "segmentation" not in task:
            if "rpn_thresh" in config:
                try:
                    anchor_desc = json.loads(
                        value["rpn_module"]["desc"]["per_tensor_desc"][0]
                    )
                    classnames = anchor_desc["class_name"]
                    thresh = []
                    for clss in classnames:
                        assert clss in config.rpn_thresh
                        thresh.append(config.rpn_thresh[clss])
                    anchor_desc["score_threshold"] = thresh
                    value["rpn_module"]["desc"]["per_tensor_desc"][
                        0
                    ] = json.dumps(anchor_desc)
                    value["rpn_module"]["postprocess"][
                        "box_filter_threshold"
                    ] = min(thresh)
                except KeyError as e:
                    print(e)
                    pass

            if "roi_module" not in value:
                continue

            target_desc = []
            try:
                task_desc = copy.deepcopy(
                    value["roi_module"]["head_desc"]["per_tensor_desc"]
                )
                value["roi_module"]["head_desc"][
                    "per_tensor_desc"
                ] = target_desc
            except KeyError:
                task_desc = copy.deepcopy(
                    value["roi_module"]["head"]["modules"][-1][
                        "per_tensor_desc"
                    ]
                )
                value["roi_module"]["head"]["modules"][-1][
                    "per_tensor_desc"
                ] = target_desc

            for desc in task_desc:
                desc = json.loads(desc)
                if desc["task"] == "frcnn_roi_detection":
                    if "subbox_score_thresh" in desc.keys():
                        classnames = desc["class_name"]
                        for clss in classnames:
                            if clss in config.roi_det_thresh.keys():
                                desc[
                                    "subbox_score_thresh"
                                ] = config.roi_det_thresh[clss]
                            else:
                                print(
                                    "roi detection score thresh of %s is empty"
                                    % clss
                                )

                elif desc["task"] == "frcnn_detection":
                    if (
                        "box_reg" in desc["output_name"]
                        or "box_score" in desc["output_name"]
                    ):
                        classnames = desc["class_name"]
                        thresh = []
                        for clss in classnames:
                            assert clss in config.det_thresh
                            thresh.append(config.det_thresh[clss])
                        desc["score_threshold"] = thresh

                elif desc["task"] == "frcnn_roi_3d_detection":
                    if "score_threshold" in desc.keys():
                        classnames = desc["class_name"]
                        if len(classnames) == 1:
                            desc["score_threshold"] = [
                                config.thresh_3d[classnames[0]]
                            ]
                        else:
                            desc["score_threshold"] = [
                                config.thresh_3d[classname]
                                for classname in classnames
                            ]
                target_desc.append(json.dumps(desc))


def check_and_convert(config):
    """
    Update model descs and convert it to IntInfer model.
    """
    input_sources = ["pyramid", "resizer", "ddr"]
    for cfg_i in _as_list(config.models):
        model_cfg = Config.fromfile(cfg_i.cfg_path)

        pt_file = os.path.join(
            model_cfg.ckpt_dir,
            f"int_infer_{cfg_i.name}-deploy-checkpoint-last.pt",
        )
        if os.path.exists(pt_file) and not cfg_i.get("override", False):
            print(f"Skip {cfg_i.name}")
            continue

        assert any(
            [
                input_source in cfg_i.input_source
                for input_source in input_sources
            ]
        ), ("Invalid input type %s" % cfg_i.input_source)
        model_file = cfg_i.model_address.gpfs_path.model

        # 1. update dpp thresh and detection thresh for descs in cfg.
        update_cfg = cfg_i.get("update_cfg", {})
        if update_cfg:
            update_descs(model_cfg.test_model, update_cfg)

        # 2. modify cfg of int_infer stage
        model_cfg["int_checkpoint"]["save_hash"] = False
        model_cfg["int_checkpoint"]["name_prefix"] = f"int_infer_{cfg_i.name}-"
        model_cfg["int_solver"]["pre_step_checkpoint"] = str(model_file)

        # 3. generate int_infer model according to modified qat model
        horizon.quantization.march = model_cfg.march
        model_cfg["march"] = model_cfg.march

        with DisableLogger(False), RegistryContext():
            trainer = TrainerWrapper(
                cfg=model_cfg, train_step="int_infer", logger=logger, device=0
            )
            trainer.prepare_fit(
                export_ckpt_only=False,
                val_only=False,
                val_ckpt=None,
            )
            trainer.fit()


def compile_local(config):
    """
    Compile model accoridng to configs in local environment.
    """

    hbms = []
    for cfg_i in _as_list(config.models):
        model_cfg = Config.fromfile(cfg_i.cfg_path)
        out_dir = config.output_dir
        if out_dir != "" and not os.path.exists(out_dir):
            os.makedirs(out_dir)

        hbm = os.path.join(
            out_dir,
            f"{cfg_i.name}_{config.optimization_level}.hbm",
        )

        hbms.append(hbm)

        if os.path.exists(hbm) and not cfg_i.get("override", False):
            print(f"skip compile {cfg_i.name}")
            continue

        pt_file = os.path.join(
            model_cfg.ckpt_dir,
            f"int_infer_{cfg_i.name}-deploy-checkpoint-last.pt",
        )

        # load model file
        model = torch.jit.load(pt_file)

        example_inputs = None
        input_size = cfg_i.input_shape

        if cfg_i.input_type == "dict":
            group_input_size = [
                list(map(int, _input_size.split("x")))
                for _input_size in input_size.split(",")
            ]
            assert len(group_input_size[0]) in [4, 5]

            if len(group_input_size) > 1:
                example_inputs = {}
                example_inputs[cfg_i.input_key] = []
                for gis in group_input_size:
                    assert len(gis) == 4
                    example_inputs[cfg_i.input_key] += [torch.randn(gis)]
            elif len(group_input_size[0]) == 5:
                input_size = group_input_size[0]
                sequence_length = input_size[0]
                example_inputs = {
                    cfg_i.input_key: [
                        torch.randn(input_size[1:])
                        for _ in range(sequence_length)
                    ]
                }
            elif len(group_input_size[0]) == 4:
                input_size = group_input_size[0]
                example_inputs = {cfg_i.input_key: torch.randn(input_size)}
            else:
                print("param invalid")
                return
        else:
            size = [int(i) for i in input_size.split("x")]
            example_inputs = [torch.ones(size[0], size[1], size[2], size[3])]

        cpu_num = cfg_i.jobs_num
        print("compiling model ...")
        extra_args = cfg_i.get("extend_flags", None)
        if extra_args:
            extra_args = extra_args.split(" ")
        t1 = time.time()
        horizon.quantization.compile_model(
            module=model.eval(),
            example_inputs=example_inputs,  # list or dict
            march=config.march,
            input_source=[cfg_i.input_source],  # pyramid
            hbm=hbm,
            name=cfg_i.compiled_name,
            input_layout=cfg_i.output_layout,
            output_layout=cfg_i.output_layout,
            opt=config.optimization_level,
            progressbar=True,
            jobs=cpu_num,
            extra_args=extra_args,
        )
        print(f"compile cost {(time.time() - t1):.2f} sec")

    result = subprocess.run(
        "hbdk-pack --output "
        + os.path.join(out_dir, config.compiled_hbm_name)
        + f" --tag {config.desc} "
        + " ".join(hbms),
        shell=True,
    )
    print(result)


def upload_to_aidiexp(config):
    client = AIDIClient()
    framework = config.framework

    for cfg_i in _as_list(config.models):
        model_version = cfg_i.get("model_version", config.model_version)
        model_cfg = Config.fromfile(cfg_i.cfg_path)
        model_name = cfg_i.name
        task_type = cfg_i.task_type
        intinfer_model_file = os.path.join(
            model_cfg.ckpt_dir,
            f"int_infer_{cfg_i.name}-deploy-checkpoint-last.pt",
        )

        # create experimental model
        if not client.model.exist(model_name):
            client.model.set_public_authority()
            client.model.create(model_name=model_name, task_type=task_type)

        if not client.model.exist(model_name, model_version=model_version):
            model_version_item = client.model.create_version(
                model_name=model_name,
                model_version=model_version,
                framework=framework,
                version_tags=[model_version, framework],
            )
            print(f"created model version {str(model_version_item)}")

        if not client.model.exist(
            model_name, model_version=model_version, stage="IntInference"
        ):
            client.model.upload_checkpoint(
                model_name=model_name,
                model_version=model_version,
                stage="IntInference",
                model_file=intinfer_model_file,
            )


def compile_publish_aidi(config):
    client = AIDIClient()

    # create publish model
    publish_name = config.model_name
    model_version = config.model_version

    if not client.modelpublish.exist(publish_name=publish_name):
        client.modelpublish.create(
            publish_name=publish_name, desc=config.desc
        ).create_version(
            publish_name, publish_version=model_version, commit="a commit"
        )

    if not client.modelpublish.exist(
        publish_name=publish_name, publish_version=model_version
    ):
        client.modelpublish.create_version(
            publish_name=publish_name,
            publish_version=model_version,
            commit=config.get("commit", None),
        )

    # model compile
    for cfg_i in _as_list(config.models):
        input_shape = cfg_i.input_shape

        client.modelpublish.set_compile_parameters(
            input_shape=input_shape,
            input_type=cfg_i.input_type,
            optimization_level=int(config.optimization_level[1:]),
            plugin_version=config.plugin_version,
            input_source=cfg_i.input_source,
            cpu=4,
            input_key=cfg_i.input_key,
            extend_flags=cfg_i.get("extend_flags", None),
        ).append_model_to_compile(
            client.model.finditem(
                cfg_i.name,
                cfg_i.get("model_version", model_version),
                "IntInference",
            ),
            cfg_i.compiled_name,
        )

    client.modelpublish.compile(
        publish_name=publish_name,
        publish_version=model_version,
        hbcc_version=config.hbcc_version,
        march=config.march,
        optimization_level=int(config.optimization_level[1:]),
    )


if __name__ == "__main__":
    args = parse_args()

    pub_config = EasyDict(Config.fromfile(args.pub_config))
    os.environ["HAT_TRAINING_STEP"] = "int_infer"

    check_and_convert(pub_config)

    if args.compile_mode == "local":
        compile_local(pub_config)
    elif args.compile_mode == "aidi":
        upload_to_aidiexp(pub_config)
        compile_publish_aidi(pub_config)
