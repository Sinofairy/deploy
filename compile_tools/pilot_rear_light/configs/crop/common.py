from datasets import datapaths

batch_size = 4
log_freq = 5
save_prefix = "tmp_output"


input_hw = (192, 512)
resize_hw = input_hw
roi_region = (768, 544, 1280, 736)
vanishing_point = (int(2048 / 2), int(1280 / 2))

bn_kwargs = dict(eps=1e-5, momentum=0.1)

# data_config
inter_method = 10
pixel_center_aligned = False
min_valid_clip_area_ratio = 0.5
rand_translation_ratio = 0.1

rand_translation_ratio = 0.0
rand_crop_scale_range = (0.8, 1.0 / 0.8)
center_crop_scale_range = (0.8, 1.0 / 0.8)
norm_scale = 1.0

test_roi_num = 50

backbone = dict(
    type="VargNetV2Stage2631",
    num_classes=1000,
    multiplier=0.5,
    group_base=4,
    last_channels=1024,
    stages=(1, 2, 3, 4, 5),
    include_top=False,
    extend_features=True,
    bn_kwargs=bn_kwargs,
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
