import os

cfg_dir = os.path.join(os.path.dirname(__file__), "../../configs")

march = "bernoulli2"
output_layout = "NHWC"
optimization_level = "O3"

# 本地编译模式相关参数
output_dir = f"tmp_compile_{march}_as33"  # 编译后模型存放目录
compiled_hbm_name = "model.hbm"  # hbm文件名
jobs_num = 24  # 编译worker数量

# aidi编译模式相关参数
framework = "PyTorch"  # framework 无需修改
model_version = "v10.1.0"  # 模型版本
plugin_version = "0.14.3_openexplorer"  # 编译用horizon_plugin_pytorch版本号
hbcc_version = "v3.29.1"  # 编译用hbdk版本号

# 模型名称，编译后hbm模型整体的模型名称，aidi编译模式下 也是发版模型的名称
model_name = f"MCP3.0_{march}_as33_{model_version}"
desc = "MCP3.0-torch-test"  # 模型的描述

postfix = "test"
# 需要编译的模型
models = [
    dict(
        cfg_path=os.path.join(cfg_dir, "resize_2/multitask.py"),
        model_setting="as33_day",
        # local模式下，为单个模型输出的hbm文件名，aidi模式下，为实验模型的名称
        name=f"pilot_legorcnn_multitask_resize_2_day_torch_{march}_{postfix}",
        # 模型在打包后模型中的名称
        compiled_name="pilot_legorcnn_multitask_resize_2",
        input_shape="1x3x640x1024",
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
                cyclist=0.45,
                vehicle=0.3,
                vehicle_rear=0.3,
            ),
            det_thresh=dict(
                person=0.46,
                cyclist=0.58,
                vehicle=0.6,
                vehicle_rear=0.402,
            ),
            roi_det_thresh=dict(
                vehicle=0.3,
                vehicle_rear=0.58,
                person=0.6,
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
            ),
        ),
        model_address=dict(
            gpfs_path=dict(
                model="http://fm-meng01-wang.alitrain.hogpu.cc/plat_gpu/pilot_multitask_resize2_10.0_AS33_day-20220420-224640/output/models/pilot_multitask_resize2/sparse_3d_freeze_bn_2-checkpoint-last-6da2e823.pth.tar",  # noqa
            )
        ),
    ),
    dict(
        cfg_path=os.path.join(cfg_dir, "resize_2/multitask.py"),
        name=f"pilot_legorcnn_multitask_resize_2_night_torch_{march}_{postfix}",  # noqa
        model_setting="as33_night",
        compiled_name="pilot_legorcnn_multitask_resize_2_night",
        input_shape="1x3x640x1024",
        input_type="dict",
        input_source="pyramid",
        input_key="img",
        jobs_num=jobs_num,
        extend_flags="--pyramid-stride 1024 -g --max-time-per-fc 1000",
        output_layout=output_layout,
        task_type="detection",
        framework="PyTorch",
        update_cfg=dict(
            rpn_thresh=dict(
                person=0.40,
                cyclist=0.45,
                vehicle=0.45,
                vehicle_rear=0.3,
            ),
            det_thresh=dict(
                person=0.56,
                cyclist=0.66,
                vehicle=0.65,
                vehicle_rear=0.508,
            ),
            roi_det_thresh=dict(
                vehicle=0.3,
                vehicle_rear=0.68,
                person=0.65,
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
            ),
        ),
        model_address=dict(
            gpfs_path=dict(
                model="http://fm-zhengwei-hu.alitrain.hogpu.cc/plat_gpu/pilot_multitask_resize2_as33_night_ON_v10.1-20220505-203203/output/models/pilot_multitask_resize2/sparse_3d_freeze_bn_2-checkpoint-last-e4ad3db1.pth.tar",  # noqa
            )
        ),
    ),
    dict(
        cfg_path=os.path.join(cfg_dir, "resize_4/multitask.py"),
        name=f"pilot_legorcnn_multitask_resize_4_day_torch_{march}_{postfix}",
        model_setting="as33_day",
        compiled_name="pilot_legorcnn_multitask_resize_4",
        input_shape="1x3x320x512",
        input_type="dict",
        input_source="pyramid",
        input_key="img",
        jobs_num=jobs_num,
        extend_flags="--pyramid-stride 1024 -g --max-time-per-fc 1000",
        output_layout=output_layout,
        task_type="detection",
        framework="PyTorch",
        update_cfg=dict(
            rpn_thresh=dict(
                person=0.40,
                cyclist=0.45,
                vehicle=0.3,
                vehicle_rear=0.3,
            ),
            det_thresh=dict(
                person=0.56,
                cyclist=0.57,
                vehicle=0.62,
                vehicle_rear=0.533,
            ),
            roi_det_thresh=dict(
                vehicle=0.3,
                vehicle_rear=0.5,
                person=0.6,
            ),
            thresh_3d=dict(
                person=0.03,
                cyclist=0.17,
                vehicle=0.2,
            ),
            bbox_clipping=dict(
                person=False,
                cyclist=False,
                vehicle=False,
                vehicle_rear=False,
            ),
        ),
        model_address=dict(
            gpfs_path=dict(
                model="http://fm-tian-li.alitrain.hogpu.cc/plat_gpu/pilot_multitask_resize4_10.0_AS33_day-20220421-163253/output/models/pilot_multitask_resize4/sparse_3d_freeze_bn_2-checkpoint-last-cdf8e28c.pth.tar",  # noqa
            )
        ),
    ),
    dict(
        cfg_path=os.path.join(cfg_dir, "resize_4/multitask.py"),
        name=f"pilot_legorcnn_multitask_resize_4_night_torch_{march}_{postfix}",  # noqa
        model_setting="as33_night",
        compiled_name="pilot_legorcnn_multitask_resize_4_night",
        input_shape="1x3x320x512",
        input_type="dict",
        input_source="pyramid",
        input_key="img",
        jobs_num=jobs_num,
        extend_flags="--pyramid-stride 1024 -g --max-time-per-fc 1000",
        output_layout=output_layout,
        task_type="detection",
        framework="PyTorch",
        update_cfg=dict(
            rpn_thresh=dict(
                person=0.40,
                cyclist=0.45,
                vehicle=0.45,
                vehicle_rear=0.3,
            ),
            det_thresh=dict(
                person=0.56,
                cyclist=0.83,
                vehicle=0.60,
                vehicle_rear=0.506,
            ),
            roi_det_thresh=dict(
                vehicle=0.3,
                vehicle_rear=0.65,
                person=0.65,
            ),
            thresh_3d=dict(
                person=0.025,
                cyclist=0.118,
                vehicle=0.2,
            ),
            bbox_clipping=dict(
                person=False,
                cyclist=False,
                vehicle=False,
                vehicle_rear=False,
            ),
        ),
        model_address=dict(
            gpfs_path=dict(
                model="http://fm-zhengwei-hu.alitrain.hogpu.cc/plat_gpu/pilot_multitask_resize4_as33_night_ON_v10.1-20220505-203354/output/models/pilot_multitask_resize4/sparse_3d_freeze_bn_2-checkpoint-last-21cbd471.pth.tar",  # noqa
            )
        ),
    ),
    dict(
        cfg_path=os.path.join(cfg_dir, "crop/multitask.py"),
        name=f"pilot_legorcnn_multitask_crop_day_torch_{march}_{postfix}",
        model_setting="day",
        compiled_name="pilot_legorcnn_multitask_crop",
        input_shape="1x3x192x512",
        input_type="dict",
        input_source="pyramid",
        input_key="img",
        jobs_num=jobs_num,
        extend_flags="--pyramid-stride 2048 -g --max-time-per-fc 1000",
        output_layout=output_layout,
        task_type="detection",
        framework="PyTorch",
        update_cfg=dict(
            rpn_thresh=dict(
                person=0.3,
                cyclist=0.45,
                vehicle=0.3,
                vehicle_rear=0.3,
            ),
            det_thresh=dict(
                person=0.49,
                cyclist=0.69,
                vehicle=0.50,
                vehicle_rear=0.466,
            ),
            roi_det_thresh=dict(
                vehicle=0.3,
                vehicle_rear=0.3,
                person=0.3,
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
            ),
        ),
        model_address=dict(
            gpfs_path=dict(
                # model="http://fm-tian-li.alitrain.hogpu.cc/plat_gpu/pilot_multitask_crop_9.0_day-20220403-223145/output/models/pilot_multitask_crop/freeze_bn_3-checkpoint-last-c2cf9cd9.pth.tar",  # noqa
                model="http://fm-qingzhou-shen.alitrain.hogpu.cc/plat_gpu/pilot_multitask_crop_3.0_cc02_x3c_day-20220409-143311/output/models/pilot_multitask_crop/freeze_bn_3-checkpoint-last-870e279a.pth.tar",  # noqa
            )
        ),
    ),
    dict(
        cfg_path=os.path.join(cfg_dir, "crop/multitask.py"),
        name=f"pilot_legorcnn_multitask_crop_night_torch_{march}_{postfix}",
        model_setting="night",
        compiled_name="pilot_legorcnn_multitask_crop_night",
        input_shape="1x3x192x512",
        input_type="dict",
        input_source="pyramid",
        input_key="img",
        jobs_num=jobs_num,
        extend_flags="--pyramid-stride 2048 -g --max-time-per-fc 1000",
        output_layout=output_layout,
        task_type="detection",
        framework="PyTorch",
        update_cfg=dict(
            rpn_thresh=dict(
                person=0.2,
                cyclist=0.3,
                vehicle=0.3,
                vehicle_rear=0.3,
            ),
            det_thresh=dict(
                person=0.38,
                cyclist=0.58,
                vehicle=0.50,
                vehicle_rear=0.514,
            ),
            roi_det_thresh=dict(
                vehicle=0.3,
                vehicle_rear=0.3,
                person=0.3,
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
            ),
        ),
        model_address=dict(
            gpfs_path=dict(
                # model="http://fm-tian-li.alitrain.hogpu.cc/plat_gpu/pilot_multitask_crop_10.0_night-20220416-013004/output/models/pilot_multitask_crop/freeze_bn_3-checkpoint-last-53f561b0.pth.tar",  # noqa
                model="http://fm-qingzhou-shen.alitrain.hogpu.cc/plat_gpu/pilot_multitask_crop_3.0_cc02_x3c_night-20220409-143816/output/models/pilot_multitask_crop/freeze_bn_3-checkpoint-last-8d3bf8a3.pth.tar",  # noqa
            )
        ),
    ),
    dict(
        cfg_path=os.path.join(
            cfg_dir, "single_task/image_fail_segmentation.py"
        ),
        name=f"pilot_legorcnn_iqa_parsing_resize_4_day_torch_{march}_{postfix}",  # noqa
        compiled_name="pilot_legorcnn_iqa_parsing_resize_4",
        input_shape="1x3x320x512",
        input_type="dict",
        input_source="pyramid",
        input_key="img",
        jobs_num=jobs_num,
        extend_flags="--pyramid-stride 512 -g --max-time-per-fc 1000",
        output_layout=output_layout,
        task_type="segmentation",
        framework="PyTorch",
        model_address=dict(
            gpfs_path=dict(
                model="http://fm-bo-chen.alitrain.hogpu.cc/plat_gpu/image-fail-seg-float5w-freezebn1w-cls7-v06-3-input-320x512-hat-20220417-145028/output/models/image_fail_parsing/qat-checkpoint-last-e877e5c1.pth.tar",  # noqa
            )
        ),
    ),
    dict(
        cfg_path=os.path.join(
            cfg_dir, "single_task/image_fail_segmentation.py"
        ),
        name=f"pilot_legorcnn_iqa_parsing_resize_4_night_torch_{march}_{postfix}",  # noqa
        compiled_name="pilot_legorcnn_iqa_parsing_resize_4_night",
        input_shape="1x3x320x512",
        input_type="dict",
        input_source="pyramid",
        input_key="img",
        jobs_num=jobs_num,
        extend_flags="--pyramid-stride 512 -g --max-time-per-fc 1000",
        output_layout=output_layout,
        task_type="segmentation",
        framework="PyTorch",
        model_address=dict(
            gpfs_path=dict(
                model="http://fm-bo-chen.alitrain.hogpu.cc/plat_gpu/image-fail-seg-float5w-freezebn1w-cls7-v06-3-input-320x512-hat-20220417-145028/output/models/image_fail_parsing/qat-checkpoint-last-e877e5c1.pth.tar",  # noqa
            )
        ),
    ),
    dict(
        cfg_path=os.path.join(cfg_dir, "single_task/vehicle_reid.py"),
        name=f"pilot_veh_reid_day_torch_{march}_{postfix}",
        compiled_name="pilot_veh_reid",
        input_shape="1x3x128x128",
        input_type="dict",
        input_source="resizer",
        input_key="img",
        jobs_num=jobs_num,
        extend_flags="-g",
        output_layout=output_layout,
        task_type="segmentation",
        framework="PyTorch",
        model_address=dict(
            gpfs_path=dict(
                model="/mnt/cephfs-adas-boschhwy/adas/meng01.wang/model/pilot3.0/MCP3.0-AlgoModel5V-ON-8.0.0-20220224/reid/qat-checkpoint-last.pth.tar",  # noqa
            )
        ),
    ),
    dict(
        cfg_path=os.path.join(cfg_dir, "single_task/vehicle_reid.py"),
        name=f"pilot_veh_reid_night_torch_{march}_{postfix}",
        compiled_name="pilot_veh_reid_night",
        input_shape="1x3x128x128",
        input_type="dict",
        input_source="resizer",
        input_key="img",
        jobs_num=jobs_num,
        extend_flags="-g",
        output_layout=output_layout,
        task_type="segmentation",
        framework="PyTorch",
        model_address=dict(
            gpfs_path=dict(
                model="/mnt/cephfs-adas-boschhwy/adas/meng01.wang/model/pilot3.0/MCP3.0-AlgoModel5V-ON-8.0.0-20220224/reid/qat-checkpoint-last.pth.tar",  # noqa
            )
        ),
    ),
    dict(
        cfg_path=os.path.join(cfg_dir, "single_task/depth_estimation.py"),
        name=f"pilot_depth_resize_4_day_torch_{march}_{postfix}",
        compiled_name="pilot_depth_resize_4",
        input_shape="1x3x320x512",
        input_type="dict",
        input_source="pyramid",
        input_key="img",
        jobs_num=jobs_num,
        extend_flags="--pyramid-stride 512 -g --max-time-per-fc 1000",
        output_layout=output_layout,
        task_type="segmentation",
        framework="PyTorch",
        model_address=dict(
            gpfs_path=dict(
                model="/mnt/cephfs-adas-boschhwy/adas/meng01.wang/model/pilot3.0/MCP3.0-AlgoModel5V-ON-8.0.0-20220224/depth/qat-checkpoint-last.pth.tar",  # noqa
            )
        ),
    ),
    dict(
        cfg_path=os.path.join(cfg_dir, "single_task/depth_estimation.py"),
        name=f"pilot_depth_resize_4_night_torch_{march}_{postfix}",
        compiled_name="pilot_depth_resize_4_night",
        input_shape="1x3x320x512",
        input_type="dict",
        input_source="pyramid",
        input_key="img",
        jobs_num=jobs_num,
        extend_flags="--pyramid-stride 512 -g --max-time-per-fc 1000",
        output_layout=output_layout,
        task_type="segmentation",
        framework="PyTorch",
        model_address=dict(
            gpfs_path=dict(
                model="/mnt/cephfs-adas-boschhwy/adas/meng01.wang/model/pilot3.0/MCP3.0-AlgoModel5V-ON-8.0.0-20220224/depth/qat-checkpoint-last.pth.tar",  # noqa
            )
        ),
    ),
]
