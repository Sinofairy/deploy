from __future__ import absolute_import, print_function
import os

from hat.data.packer.anno_transformer import default_anno_to_contours_fn
from hat.data.packer.utils import (
    get_colors_and_class_names_for_parsing,
    get_default_pack_args_from_environment,
    init_logger,
)
from hat.utils import Config

# default args
packer_env = get_default_pack_args_from_environment()
num_workers = packer_env.num_worker
verbose = packer_env.verbose

# logger
logger = init_logger(log_file="log/train.log", overwrite=True)

task_name = "default_parsing"

# default_parsing cls
cls = 16.2
label_map_config = Config.fromfile(
    os.path.join("configs", task_name, "parsing_labelmap_16cls_v2.py")
)  # noqa
src_label = label_map_config.src_label
dst_label = label_map_config.dst_label
color_map = label_map_config.color_map

input_root_dir = f"data/{task_name}"
output_dir = f"data/lmdb/{task_name}/"
label_map_output_dir = f"{output_dir}/gt/anno_{cls}"

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

colors, clsnames = get_colors_and_class_names_for_parsing(color_map)

anno_transformer = [
    dict(
        type="DefaultGenerateLabelMapAnnoTs",
        __build_recursive=False,
        output_dir=label_map_output_dir,
        src_label=src_label,
        dst_label=dst_label,
        colors=colors,
        clsnames=clsnames,
        anno_to_contours_fn=default_anno_to_contours_fn,
        is_merge=False,
    ),
    dict(
        type="DenseBoxSegAnnoTs",
        __build_recursive=False,
        class_ids=list(range(1, 20)),
        verify_image=True,
        verify_label=True,
    ),
]

data_packer = dict(
    type="DetSeg2DPacker",
    input_img_dir=input_img_dir,
    input_anno_file=input_anno_file,
    output_dir=output_dir,
    num_workers=num_workers,
)
