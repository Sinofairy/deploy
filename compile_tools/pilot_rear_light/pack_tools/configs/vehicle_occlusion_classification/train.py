from __future__ import absolute_import, print_function
import os

import yaml

from hat.data.packer.utils import (
    get_default_pack_args_from_environment,
    init_logger,
)

CLASS_NAMES = [
    "full_visible",
    "occluded",
    "heavily_occluded",
    "invisible",
    "unknown",
]

# default args
packer_env = get_default_pack_args_from_environment()
num_workers = packer_env.num_worker
verbose = packer_env.verbose


# logger
logger = init_logger(log_file="log/train.log", overwrite=True)

task_name = "vehicle_occlusion_classification"
input_task_name = "vehicle_detection"

anno_ts_fn_config_path = f"configs/{task_name}/vehicle_occlusion_config.yaml"
with open(anno_ts_fn_config_path) as fin:
    anno_ts_fn_config = yaml.load(fin, Loader=yaml.FullLoader)

input_root_dir = f"data/{input_task_name}"
output_dir = f"data/lmdb/{task_name}/"

idx_path = os.path.join(output_dir, "idx")
img_path = os.path.join(output_dir, "img")
anno_path = os.path.join(output_dir, "anno")


viz_num_show = 20
viz_save_flag = True
viz_save_path = os.path.join(output_dir, "viz")


for root, _, files in os.walk(input_root_dir):
    for file in files:
        if file.endswith(".jpg"):
            input_img_dir = root
            break
        if file.endswith("data.json"):
            input_anno_file = os.path.join(root, file)
            break


data_packer = dict(
    type="DetSeg2DPacker",
    input_img_dir=input_img_dir,
    input_anno_file=input_anno_file,
    output_dir=output_dir,
    num_workers=num_workers,
)

# default mode
anno_transformer = dict(
    type="DenseBoxDetAnnoTs",
    anno_config=anno_ts_fn_config,
    root_dir=input_root_dir,  # noqa
    verbose=verbose,
    __build_recursive=False,
)

# dataset for visualization
viz_dataset = dict(
    type="DetSeg2DAnnoDataset",
    idx_path=idx_path,
    img_path=img_path,
    anno_path=anno_path,
)

# funcation for visualization
viz_fn = dict(
    type="VizDenseBoxDetAnno",
    save_flag=viz_save_flag,
    save_path=viz_save_path,
    viz_class_id=list(range(1, len(CLASS_NAMES) + 1)),
    class_name=CLASS_NAMES,
    lt_point_id=0,
    rb_point_id=2,
)
