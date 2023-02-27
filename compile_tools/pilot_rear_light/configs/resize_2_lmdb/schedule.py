import os

pipeline_test = os.environ.get("HAT_PIPELINE_TEST", "0") == "1"

if pipeline_test:
    num_steps = dict(
        with_bn=20,
        calibration=10,
        freeze_bn_1=10,
        freeze_bn_2=10,
        freeze_bn_3=10,
        sparse_3d_freeze_bn_1=10,
        sparse_3d_freeze_bn_2=10,
        int_infer=0,
    )
    warmup_steps = 0
    save_interval = 5
else:
    num_steps = dict(
        with_bn=80000,
        calibration=100,
        freeze_bn_1=20000,
        freeze_bn_2=20000,
        freeze_bn_3=20000,
        sparse_3d_freeze_bn_1=80000,
        sparse_3d_freeze_bn_2=20000,
        int_infer=0,
    )
    warmup_steps = 1000
    save_interval = 1000

base_lr = dict(
    with_bn=0.0015,
    calibration=0,
    freeze_bn_1=0.00005,
    freeze_bn_2=0.00001,
    freeze_bn_3=0.00001,
    sparse_3d_freeze_bn_1=0.0015,
    sparse_3d_freeze_bn_2=0.00001,
    int_infer=0.0,
)

interval_by = "step"

# freeze bn config
f1 = [
    "^.*backbone",  # backbone
    "^.*fpn_neck",  # fpn
    "^.*fix_channel_neck",  # fix channel fpn
]
f2 = [
    "^.*ufpn.*neck",  # ufpn in seg and 3d tasks
    "^.*_anchor_head",  # rpn heads
]
f3 = [
    "^.*_roi[^3]*head",  # roi head (without roi 3d head)
    "^(?!(.*roi|.*anchor)).*head",  # seg and 3d heads
]
f4 = [
    "^.*_roi_3d.*head",
]

freeze_bn_modules = dict(
    calibration=None,
    freeze_bn_1=f1,
    freeze_bn_2=f2,
    freeze_bn_3=f3,
    sparse_3d_freeze_bn_2=f4,
    int_infer=None,
)
