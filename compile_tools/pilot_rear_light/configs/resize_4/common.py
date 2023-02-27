from copy import deepcopy
from datasets import datapaths

batch_size = 4
log_freq = 5
save_prefix = "tmp_output"

input_hw = resize_hw = (320, 480)
actual_input_hw = (320, 512)
# input_padding (w_left, w_right, h_top, h_bottom)
input_padding = [16, 16, 0, 0]
roi_region = (0, 0, *input_hw[::-1])
vanishing_point = (input_hw[1] // 2, input_hw[0] // 2)

bn_kwargs = dict(eps=1e-5, momentum=0.1)

# data_config
inter_method = 10
pixel_center_aligned = False
min_valid_clip_area_ratio = 0.5
rand_translation_ratio = 0.1

# 3d config
undistort_depth_uv = True

test_roi_num = 30

backbone = dict(
    type="ZeroPad2DPatcher",
    backbone=dict(
        type="VargNetV2Stage2631",
        num_classes=1000,
        multiplier=0.5,
        group_base=4,
        last_channels=1024,
        stages=(1, 2, 3, 4, 5),
        include_top=False,
        extend_features=True,
        bn_kwargs=bn_kwargs,
    ),
    input_padding=input_padding,
    __graph_model_name="backbone",
)

fpn_neck = dict(
    type="FPN",
    in_strides=[2, 4, 8, 16, 32, 64],
    in_channels=[16, 16, 32, 64, 128, 128],
    out_strides=[4, 8, 16, 32, 64],
    out_channels=[16, 32, 64, 128, 128],
    bn_kwargs=bn_kwargs,
    __graph_model_name="fpn_neck",
)

fix_channel_neck = dict(
    type="FixChannelNeck",
    in_strides=[4, 8, 16, 32, 64],
    in_channels=[16, 32, 64, 128, 128],
    out_strides=[8, 16, 32, 64],
    out_channel=64,
    bn_kwargs=bn_kwargs,
    __graph_model_name="fix_channel_neck",
)

ufpn_seg_neck = dict(
    type="UFPN",
    in_strides=[4, 8, 16, 32, 64],
    in_channels=[16, 32, 64, 128, 128],
    out_channels=[16, 32, 64, 128, 128],
    bn_kwargs=bn_kwargs,
    group_base=4,
    __graph_model_name="ufpn_seg_neck",
)

ufpn_3d_neck = deepcopy(ufpn_seg_neck)
ufpn_3d_neck.update(
    dict(
        output_strides=[4],
        __graph_model_name="ufpn_3d_neck",
    )
)
