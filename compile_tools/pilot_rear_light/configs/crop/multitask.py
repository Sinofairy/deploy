import os
import sys
from copy import deepcopy
from importlib import import_module
from pathlib import Path

import torch

# -------------------------- common --------------------------
from common import input_hw, log_freq, save_prefix
from schedule import (
    base_lr,
    freeze_bn_modules,
    interval_by,
    num_steps,
    save_interval,
    warmup_steps,
)

from hat.core.task_sampler import TaskSampler
from hat.engine.processors.loss_collector import collect_loss_by_regex

num_machines = 1

# step-specific args
training_step = os.environ.get("HAT_TRAINING_STEP", "with_bn")

num_steps = num_steps[training_step]
base_lr = base_lr[training_step]

# -------------------------- multitask --------------------------
model_name = "pilot_multitask_crop"

seed = None
log_rank_zero_only = True
cudnn_benchmark = True
sync_bn = True
enable_amp = False
march = "bernoulli2"

device_ids = [0]


ckpt_dir = Path(save_prefix) / model_name
log_dir = ckpt_dir / "logs"
redirect_config_logging_path = (
    log_dir / f"config_log_{training_step}.log"
).as_posix()

# -------------------------- task --------------------------
tasks = [
    # vehicle
    "vehicle_detection",
    "vehicle_category_classification",
    "vehicle_occlusion_classification",
    "vehicle_wheel_kps",
    "vehicle_wheel_detection",
    "vehicle_ground_line",
    # rear
    "rear_detection",
    "rear_occlusion_classification",
    "rear_part_classification",
    # person
    "person_detection",
    "person_occlusion_classification",
    "person_orientation_classification",
    "person_pose_classification",
    # cyclist
    "cyclist_detection",
]

TASK_CONFIGS = [import_module(t) for t in tasks]

task_sampler_configs = {
    T.task_name: dict(sampling_factor=1) for T in TASK_CONFIGS
}

task_sampler = TaskSampler(
    task_config=task_sampler_configs,
    method="sample_all",
)

# -------------------------- data --------------------------
loaders = {T.task_name: T.data_loader for T in TASK_CONFIGS}

data_loader = dict(
    type="MultitaskInfLoader",
    loaders=loaders,
    task_sampler=task_sampler,
    return_task=True,
    __build_recursive=False,
)

inputs = dict(img=torch.zeros((1, 3, *input_hw)))


# -------------------------- model --------------------------
def get_model(mode):
    return dict(
        type="MultitaskGraphModel",
        inputs=inputs,
        task_inputs={T.task_name: T.inputs[mode] for T in TASK_CONFIGS},
        task_modules={T.task_name: T.get_model(mode) for T in TASK_CONFIGS},
        lazy_forward=False,
        __build_recursive=False,
    )


model = get_model("train")
val_model = get_model("val")
test_model = get_model("test")

# -------------------------- callbacks --------------------------
batch_processor = dict(
    type="MultiBatchProcessor",
    need_grad_update=True,
    batch_transforms=[
        dict(type="BgrToYuv444", rgb_input=True),
        dict(
            type="TorchVisionAdapter",
            interface="Normalize",
            mean=128.0,
            std=128.0,
        ),
    ],
    loss_collector=collect_loss_by_regex("^.*loss.*"),
    enable_amp=enable_amp,
)

stat_callback = dict(
    type="StatsMonitor",
    log_freq=log_freq,
)

lr_callback = dict(
    type="PolyLrUpdater",
    max_update=num_steps // num_machines,
    power=1.0,
    warmup_len=warmup_steps,
    step_log_interval=25,
)

tb_update_funcs = [
    T.tb_update_func
    for T in TASK_CONFIGS
    if getattr(T, "tb_update_func", None) is not None
]

tensorboard_callback = dict(
    type="TensorBoard",
    save_dir=log_dir.as_posix(),
    update_freq=log_freq,
    tb_update_funcs=tb_update_funcs,
)

best_metric = None

checkpoint_callback = dict(
    type="Checkpoint",
    save_dir=ckpt_dir.as_posix(),
    name_prefix=training_step + "-",
    strict_match=False,
    best_refer_metric=best_metric,
    save_interval=save_interval,
    interval_by=interval_by,
    save_on_train_end=True,
)

int_checkpoint = deepcopy(checkpoint_callback)
int_checkpoint.update(
    dict(
        deploy_model=test_model,
        deploy_inputs=dict(
            img=torch.randn((1, 3, input_hw[0], input_hw[1])),
        ),
    )
)

ckpt_callback = (
    int_checkpoint if training_step == "int_infer" else checkpoint_callback
)

metric_updaters = [T.metric_updater for T in TASK_CONFIGS]

# the order of callbacks affects the logging order
callbacks = [
    stat_callback,
    lr_callback,
    checkpoint_callback,
    tensorboard_callback,
]

callbacks += metric_updaters

# -------------------------- solver --------------------------
trainer = dict(
    type="DistributedDataParallelTrainer",
    model=model,
    data_loader=data_loader,
    optimizer=dict(
        type="LegacyNadamEx",
        params={"weight": dict(weight_decay=5e-5)},
        lr=base_lr * num_machines,
        rescale_grad=1,
    ),
    batch_processor=batch_processor,
    stop_by="step",
    num_steps=(num_steps + warmup_steps) // num_machines,
    device=None,  # set when building
    sync_bn=sync_bn,
    callbacks=callbacks,
)

# -------------------------- training ----------------------------
quantize = True
qconfig_params = dict(
    # activation_fake_quant="lsq",
    # weight_fake_quant="lsq"
)

if quantize:
    if training_step == "int_infer":
        qat_mode = "fuse_bn"
    else:
        qat_mode = "with_bn_reverse_fold"

# -------------------------- step with_bn --------------------------
pretrain_checkpoint = None

base_solver = dict(
    trainer=trainer,
    check_quantize_model=False,
    qconfig_params=qconfig_params,
    pretrain_checkpoint=None,
    pre_step=None,
    pre_step_checkpoint=None,
    resume_checkpoint=None,
    resume_epoch_or_step=False,
    resume_optimizer=False,
    pre_fuse_patterns=[],
    qat_fuse_patterns=[],
    allow_miss=True,
    ignore_extra=True,
    verbose=1,
)

with_bn_solver = deepcopy(base_solver)
with_bn_solver.update(
    dict(
        quantize=quantize,
        pretrain_checkpoint=pretrain_checkpoint,
    )
)

# -------------------------- step freeze_bn_1 --------------------------
freeze_bn_1_solver = deepcopy(base_solver)
freeze_bn_1_solver.update(
    dict(
        quantize=quantize,
        pre_step="with_bn",
        pre_step_checkpoint=os.path.join(
            ckpt_dir, "with_bn-checkpoint-last.pth.tar"
        ),
        qat_fuse_patterns=freeze_bn_modules["freeze_bn_1"],
        allow_miss=False,
        ignore_extra=False,
    )
)

# -------------------------- step freeze_bn_2 --------------------------
freeze_bn_2_solver = deepcopy(base_solver)
freeze_bn_2_solver.update(
    dict(
        quantize=quantize,
        pre_step="freeze_bn_1",
        pre_step_checkpoint=os.path.join(
            ckpt_dir, "freeze_bn_1-checkpoint-last.pth.tar"
        ),
        pre_fuse_patterns=freeze_bn_modules["freeze_bn_1"],
        qat_fuse_patterns=freeze_bn_modules["freeze_bn_2"],
        allow_miss=False,
        ignore_extra=False,
    )
)

# -------------------------- step freeze_bn_3 --------------------------
freeze_bn_3_solver = deepcopy(base_solver)
freeze_bn_3_solver.update(
    dict(
        quantize=quantize,
        pre_step="freeze_bn_2",
        pre_step_checkpoint=os.path.join(
            ckpt_dir, "freeze_bn_2-checkpoint-last.pth.tar"
        ),
        pre_fuse_patterns=freeze_bn_modules["freeze_bn_1"]
        + freeze_bn_modules["freeze_bn_2"],
        qat_fuse_patterns=freeze_bn_modules["freeze_bn_3"],
        allow_miss=False,
        ignore_extra=False,
    )
)

# -------------------------- step int_infer --------------------------
int_solver = dict(
    trainer=dict(
        type="Trainer",
        model=test_model,
        num_epochs=0,
        device=None,
        optimizer=None,
        batch_processor=None,
        callbacks=[int_checkpoint],
    ),
    quantize=True,
    check_quantize_model=False,
    resume_checkpoint=None,
    pre_step="freeze_bn_3",
    pre_step_checkpoint=os.path.join(
        ckpt_dir, "freeze_bn_3-checkpoint-last.pth.tar"
    ),
    # other
    allow_miss=False,
    ignore_extra=True,  # qat has more params than int, such as loss params
    verbose=1,
)

# entry
step2solver = dict(
    with_bn=with_bn_solver,
    freeze_bn_1=freeze_bn_1_solver,
    freeze_bn_2=freeze_bn_2_solver,
    freeze_bn_3=freeze_bn_3_solver,
    int_infer=int_solver,
)

for t in tasks + ["common", "schedule"]:
    sys.modules.pop(t)
