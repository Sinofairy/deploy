from __future__ import absolute_import, print_function
import os

from hat.data.packer.utils import (
    get_default_pack_args_from_environment,
    init_logger,
)

# default args
packer_env = get_default_pack_args_from_environment()
num_workers = packer_env.num_worker
verbose = packer_env.verbose

# logger
logger = init_logger(log_file="log/packing.log", overwrite=True)

parent_class_name = "vehicle"
children_class_name = "vehicle_kps_8"
task_name = "vehicle_ground_line"
input_task_name = "vehicle_keypoints"
task_attribute = "train"

input_root_dir = f"data/{input_task_name}"
output_dir = f"data/lmdb/{task_name}/"

idx_path = os.path.join(output_dir, "idx")
img_path = os.path.join(output_dir, "img")
anno_path = os.path.join(output_dir, "anno")


for root, _, files in os.walk(input_root_dir):
    for file in files:
        if file.endswith(".jpg"):
            input_img_dir = root
            break
        if file.endswith("data.json"):
            input_anno_file = os.path.join(root, file)
            break

max_num_imgs = -1  # maximum number of samples to be packed

# pack image data
data_packer = dict(
    type="DetSeg2DPacker",
    input_img_dir=input_img_dir,
    input_anno_file=input_anno_file,
    output_dir=output_dir,
    num_workers=num_workers,
)


# transformer for process anno data
anno_transformer = dict(
    type="VehicleFlankAnnoTs",
    parent_classname="vehicle",
    child_classname="vehicle_flank",
    anno_adapter=dict(
        # the adapter for input dataset
        type="AnnoAdapterKps8",
        parent_classname="vehicle",
        child_classname="vehicle_kps_8",
        target_classname="vehicle_flank",
        empty_value=-10000,
    ),
    min_flank_width=4,
    empty_value=-10000,
)
