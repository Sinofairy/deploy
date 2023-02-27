import os

cfg_dir = os.path.join(os.path.dirname(__file__), "../../configs")

march = "bayes"
output_layout = "NHWC"
optimization_level = "O3"

# 模型名称，编译后hbm模型整体的模型名称，aidi编译模式下 也是发版模型的名称
model_name = f"MCP3.0_8.0_no_privacy_{march}"
desc = "MCP3.0-8.0-test"  # 模型的描述

# 本地编译模式相关参数
output_dir = f"tmp_compile_{march}_x3c"  # 编译后模型存放目录
compiled_hbm_name = "model.hbm"  # hbm文件名
jobs_num = 24  # 编译worker数量

# aidi编译模式相关参数
framework = "PyTorch"  # framework 无需修改
model_version = "v8.0.0"  # 模型版本
plugin_version = "0.14.0"  # 编译用horizon_plugin_pytorch版本号
hbcc_version = "v3.28.0.post0.dev202203260142+62e0efa"  # 编译用hbdk版本号

# 需要编译的模型
models = [
    dict(
        cfg_path=os.path.join(cfg_dir, "resize_2/multitask.py"),
        # local模式下，为单个模型输出的hbm文件名，aidi模式下，为实验模型的名称
        name=f"pilot_legorcnn_multitask_resize_2_day_torch_{march}",
        # 模型在打包后模型中的名称
        compiled_name="pilot_legorcnn_multitask_resize_2",
        input_shape="1x3x640x960",
        input_type="dict",
        input_source="pyramid",
        input_key="img",
        jobs_num=jobs_num,
        extend_flags="--pyramid-stride 1024 -g --max-time-per-fc 1000",
        output_layout=output_layout,
        task_type="detection",
        framework="PyTorch",
        # 更新模型阈值中的相关参数
        update_cfg=dict(
            rpn_thresh=dict(
                person=0.40,
                cyclist=0.52,
                vehicle=0.3,
                vehicle_rear=0.3,
                rear_light=0.3,
            ),
            det_thresh=dict(
                person=0.55,
                cyclist=0.85,
                vehicle=0.60,
                vehicle_rear=0.52,
                rear_light=0.3,
            ),
            roi_det_thresh=dict(
                vehicle=0.3,
                vehicle_rear=0.3,
                person=0.3,
                rear_light=0.3,
            ),
            thresh_3d=dict(
                person=0.03,
                cyclist=0.2,
                vehicle=0.2,
            ),
            bbox_clipping=dict(
                person=False,
                cyclist=False,
                vehicle=False,
                vehicle_rear=False,
                rear_light=False,
            ),
        ),
        model_address=dict(
            gpfs_path=dict(
                model="/nas/mwl/tmp/hat/tmp_bayes.pth.tar",  # noqa
            )
        ),
    ),
]
