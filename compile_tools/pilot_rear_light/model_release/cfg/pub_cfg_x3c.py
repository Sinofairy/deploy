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
            ),
            det_thresh=dict(
                person=0.55,
                cyclist=0.85,
                vehicle=0.60,
                vehicle_rear=0.52,
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
                model="http://fm-tian-li.alitrain.hogpu.cc/plat_gpu/pilot_multitask_resize2_8.0_AS33_day-20220223-153336/output/models/pilot_multitask_resize2/sparse_3d_freeze_bn_2-checkpoint-last-8ef3b44e.pth.tar",  # noqa
            )
        ),
    ),
    dict(
        cfg_path=os.path.join(cfg_dir, "resize_2/multitask.py"),
        name=f"pilot_legorcnn_multitask_resize_2_night_torch_{march}",
        compiled_name="pilot_legorcnn_multitask_resize_2_night",
        input_shape="1x3x640x960",
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
                cyclist=0.52,
                vehicle=0.3,
                vehicle_rear=0.3,
            ),
            det_thresh=dict(
                person=0.55,
                cyclist=0.85,
                vehicle=0.60,
                vehicle_rear=0.52,
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
                model="http://fm-tian-li.alitrain.hogpu.cc/plat_gpu/pilot_multitask_resize2_8.0_AS33_night-20220223-155055/output/models/pilot_multitask_resize2/sparse_3d_freeze_bn_2-checkpoint-last-6832f972.pth.tar",  # noqa
            )
        ),
    ),
    dict(
        cfg_path=os.path.join(cfg_dir, "resize_4/multitask.py"),
        name=f"pilot_legorcnn_multitask_resize_4_day_torch_{march}",
        compiled_name="pilot_legorcnn_multitask_resize_4",
        input_shape="1x3x320x480",
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
                cyclist=0.52,
                vehicle=0.3,
                vehicle_rear=0.3,
            ),
            det_thresh=dict(
                person=0.55,
                cyclist=0.85,
                vehicle=0.60,
                vehicle_rear=0.52,
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
                model="http://fm-tian-li.alitrain.hogpu.cc/plat_gpu/pilot_multitask_resize4_8.0_AS33_day-20220224-110253/output/models/pilot_multitask_resize4/sparse_3d_freeze_bn_2-checkpoint-last-7e843a03.pth.tar",  # noqa
            )
        ),
    ),
    dict(
        cfg_path=os.path.join(cfg_dir, "resize_4/multitask.py"),
        name=f"pilot_legorcnn_multitask_resize_4_night_torch_{march}",
        compiled_name="pilot_legorcnn_multitask_resize_4_night",
        input_shape="1x3x320x480",
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
                cyclist=0.52,
                vehicle=0.3,
                vehicle_rear=0.3,
            ),
            det_thresh=dict(
                person=0.55,
                cyclist=0.85,
                vehicle=0.60,
                vehicle_rear=0.52,
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
                model="http://fm-tian-li.alitrain.hogpu.cc/plat_gpu/pilot_multitask_resize4_8.0_AS33_night-20220224-104219/output/models/pilot_multitask_resize4/sparse_3d_freeze_bn_2-checkpoint-last-3b734de7.pth.tar",  # noqa
            )
        ),
    ),
    dict(
        cfg_path=os.path.join(cfg_dir, "crop/multitask.py"),
        name=f"pilot_legorcnn_multitask_crop_day_torch_{march}",
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
                person=0.40,
                cyclist=0.52,
                vehicle=0.3,
                vehicle_rear=0.3,
            ),
            det_thresh=dict(
                person=0.55,
                cyclist=0.85,
                vehicle=0.60,
                vehicle_rear=0.52,
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
                model="http://fm-zhengwei-hu.alitrain.hogpu.cc/plat_gpu/pilot_multitask_crop_8.0_day-20220216-143501/output/models/pilot_multitask_crop/freeze_bn_3-checkpoint-last-b4fb48f9.pth.tar",  # noqa
            )
        ),
    ),
    dict(
        cfg_path=os.path.join(cfg_dir, "crop/multitask.py"),
        name=f"pilot_legorcnn_multitask_crop_night_torch_{march}",
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
                person=0.40,
                cyclist=0.52,
                vehicle=0.3,
                vehicle_rear=0.3,
            ),
            det_thresh=dict(
                person=0.55,
                cyclist=0.85,
                vehicle=0.60,
                vehicle_rear=0.52,
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
                model="http://fm-zhengwei-hu.alitrain.hogpu.cc/plat_gpu/pilot_multitask_crop_8.0_night-20220216-143121/output/models/pilot_multitask_crop/freeze_bn_3-checkpoint-last-a244c44d.pth.tar",  # noqa
            )
        ),
    ),
    dict(
        cfg_path=os.path.join(
            cfg_dir, "single_task/image_fail_segmentation.py"
        ),
        name=f"pilot_legorcnn_iqa_parsing_resize_4_day_torch_{march}",
        compiled_name="pilot_legorcnn_iqa_parsing_resize_4",
        input_shape="1x3x320x480",
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
                model="/mnt/cephfs-adas-boschhwy/adas/meng01.wang/model/pilot3.0/MCP3.0-AlgoModel5V-ON-8.0.0-20220224/image_fail/2048x1280/qat-checkpoint-last.pth.tar",  # noqa
            )
        ),
    ),
    dict(
        cfg_path=os.path.join(
            cfg_dir, "single_task/image_fail_segmentation.py"
        ),
        name=f"pilot_legorcnn_iqa_parsing_resize_4_night_torch_{march}",
        compiled_name="pilot_legorcnn_iqa_parsing_resize_4_night",
        input_shape="1x3x320x480",
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
                model="/mnt/cephfs-adas-boschhwy/adas/meng01.wang/model/pilot3.0/MCP3.0-AlgoModel5V-ON-8.0.0-20220224/image_fail/2048x1280/qat-checkpoint-last.pth.tar",  # noqa
            )
        ),
    ),
    dict(
        cfg_path=os.path.join(cfg_dir, "single_task/vehicle_reid.py"),
        name=f"pilot_veh_reid_day_torch_{march}",
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
        name=f"pilot_veh_reid_night_torch_{march}",
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
        name=f"pilot_depth_resize_4_day_torch_{march}",
        compiled_name="pilot_depth_resize_4",
        input_shape="1x3x320x480",
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
        name=f"pilot_depth_resize_4_night_torch_{march}",
        compiled_name="pilot_depth_resize_4_night",
        input_shape="1x3x320x480",
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
