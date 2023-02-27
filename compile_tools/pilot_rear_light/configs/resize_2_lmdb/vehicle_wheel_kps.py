from copy import deepcopy

import torch
from common import (
    backbone,
    batch_size,
    bn_kwargs,
    datapaths,
    fix_channel_neck,
    fpn_neck,
    inter_method,
    log_freq,
    min_valid_clip_area_ratio,
    pixel_center_aligned,
    rand_translation_ratio,
    resize_hw,
)
from vehicle_detection import (
    anchor_desc,
    anchor_generator,
    anchor_head,
    anchor_pred,
    classnames,
    object_type,
    roi_feat_extractor,
    val_decoders,
)

from hat.callbacks.metric_updater import update_metric_using_regex
from hat.core.proj_spec.descs import frcnn_kps_desc
from hat.data.collates import collate_2d

task_name = "vehicle_wheel_kps"
task_type = "kps"

data_args = dict(legacy_bbox=True)

roi_args = dict(
    exclude_background=True,
    num_fg_classes=1,
    expand_param=1.2,
)

roi_head = dict(
    type="RCNNVarGNetShareHead",
    bn_kwargs=bn_kwargs,
    roi_out_channel=64,
    gc_num_filter=128,
    pw_num_filter=128,
    pw_num_filter2=128,
    group_base=8,
    factor=1,
    stride=1,
    __graph_model_name=f"{task_name}_roi_share_head",
)


def get_model(mode):
    train_pred = deepcopy(anchor_pred)
    train_pred.update(
        pre_nms_top_k=2000,
        post_nms_top_k=256,
        use_clippings=False,
        nms_padding_mode="rollover",
        bbox_min_hw=(8, 8),
        __graph_model_name=f"{task_name}_anchor_pred",
    )

    return dict(
        type="TwoStageDetector",
        backbone=backbone,
        neck=dict(type="ExtSequential", modules=[fpn_neck, fix_channel_neck]),
        rpn_out_keys=[] if mode != "train" else None,
        rpn_module=dict(
            type="AnchorModule",
            target_keys=("gt_boxes",),
            target_opt_keys=(
                "ig_regions",
                "im_hw",
            ),
            anchor_generator=anchor_generator,
            head=anchor_head,
            postprocess=train_pred if mode == "train" else anchor_pred,
            target=None,
            loss=None,
            desc=anchor_desc if mode != "train" else None,
        ),
        roi_module=dict(
            type="RoIModule",
            output_head_out=True,
            roi_key="pred_boxes",
            roi_feat_extractor=roi_feat_extractor,
            target_keys=("gt_boxes",),
            target_opt_keys=(
                "ig_regions",
                "im_hw",
            ),
            head=dict(
                type="ExtSequential",
                modules=[
                    roi_head,
                    dict(
                        type="RCNNKPSSplitHead",
                        in_channel=128,
                        points_num=2,
                        __graph_model_name=f"{task_name}_roi_split_head",
                    ),
                ]
                + (
                    [
                        dict(
                            type="AddDesc",
                            per_tensor_desc=frcnn_kps_desc(
                                task_name="frcnn_kps_detection",
                                class_names=classnames,
                                label_output_name=f"{task_name}_detection_label",
                                offset_output_name=f"{task_name}_detection_offset",
                                roi_expand_param=roi_args["expand_param"],
                            ),
                            __graph_model_name=f"{task_name}_roi_desc",
                        ),
                    ]
                    if "test" in mode
                    else []
                ),
            ),
            target=dict(
                type="ProposalTarget",
                matcher=dict(
                    type="MaxIoUMatcher",
                    pos_iou=0.5,
                    neg_iou=0.5,
                    allow_low_quality_match=False,
                    low_quality_match_iou=0.5,
                    legacy_bbox=data_args["legacy_bbox"],
                ),
                label_encoder=dict(
                    type="RCNNKPSLabelFromMatch",
                    feat_h=8,
                    feat_w=8,
                    kps_num=2,
                    ignore_labels=(0, 3),
                    roi_expand_param=roi_args["expand_param"],
                ),
                __graph_model_name=f"{task_name}_roi_target",
            )
            if mode == "train"
            else None,
            loss=dict(
                type="RCNNKPSLoss",
                kps_num=2,
                cls_loss=dict(
                    type="SmoothL1Loss", loss_weight=1, reduction="mean"
                ),
                reg_loss=dict(
                    type="SmoothL1Loss", loss_weight=1, reduction="mean"
                ),
                feat_height=8,
                feat_width=8,
                __graph_model_name=f"{task_name}_roi_loss",
            )
            if mode == "train"
            else None,
            postprocess=dict(
                type="KpsDecoder",
                num_kps=2,
                pos_distance=1,
                roi_expand_param=1.2,
                input_padding=[0, 0, 0, 0],
                __graph_model_name=f"{task_name}_roi_decoder",
            )
            if "val" in mode
            else None,
        ),
    )


if f"{object_type}_decoder" in val_decoders:
    val_decoders[f"{object_type}_decoder"]["task_desc"][task_name] = (
        "pred_kps",
        task_type,
    )

# inputs
inputs = dict(
    train=dict(
        gt_boxes=[torch.zeros((100, 5))],
        im_hw=torch.zeros((1, 2)),
    ),
    val=dict(),
    test=dict(),
)


metric_updater = dict(
    type="MetricUpdater",
    metrics=[
        dict(type="LossShow", name="rcnn_kps_class_loss"),
        dict(type="LossShow", name="rcnn_kps_reg_loss"),
    ],
    metric_update_func=update_metric_using_regex(
        per_metric_patterns=[  # corresponding to metrics
            dict(
                label_pattern=None,
                pred_pattern=f"^.*{task_name}_rcnn_kps_class_loss$",
            ),
            dict(
                label_pattern=None,
                pred_pattern=f"^.*{task_name}_rcnn_kps_reg_loss$",
            ),
        ]
    ),
    step_log_freq=log_freq,
    epoch_log_freq=1,
    log_prefix=task_name,
    reset_metrics_by="log",
)


# data
ds = datapaths.vehicle_wheel_kps

data_loader = dict(
    type=torch.utils.data.DataLoader,
    sampler=dict(type=torch.utils.data.DistributedSampler),
    shuffle=False,
    num_workers=0,
    batch_size=batch_size,
    collate_fn=collate_2d,
    dataset=dict(
        type="ComposeRandomDataset",
        sample_weights=[path.sample_weight for path in ds.train_data_paths],
        datasets=[
            dict(
                type="DetSeg2DAnnoDataset",
                idx_path=path.idx_path,
                img_path=path.img_path,
                anno_path=path.anno_path,
                transforms=[
                    dict(
                        type="KPSIterableDetRoITransform",
                        kps_num=2,
                        # roi transform
                        target_wh=resize_hw[::-1],
                        resize_wh=None
                        if resize_hw is None
                        else resize_hw[::-1],
                        img_scale_range=(0.5, 2.0),
                        roi_scale_range=(0.7, 1.0 / 0.7),
                        min_sample_num=1,
                        max_sample_num=1,
                        center_aligned=False,
                        inter_method=inter_method,
                        use_pyramid=True,
                        pyramid_min_step=0.7,
                        pyramid_max_step=0.8,
                        min_valid_area=100,
                        min_valid_clip_area_ratio=min_valid_clip_area_ratio,
                        min_edge_size=10,
                        rand_translation_ratio=rand_translation_ratio,
                        rand_aspect_ratio=0.0,
                        rand_rotation_angle=0,
                        flip_prob=0.5,
                        clip_bbox=False,
                        pixel_center_aligned=pixel_center_aligned,
                        min_kps_distance=4,
                        keep_aspect_ratio=True,
                    ),
                ],
            )
            for path in ds.train_data_paths
        ],
    ),
)
