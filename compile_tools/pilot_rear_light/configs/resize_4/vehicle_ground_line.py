from copy import deepcopy

import numpy as np
import torch
from common import (
    backbone,
    batch_size,
    bn_kwargs,
    datapaths,
    fix_channel_neck,
    fpn_neck,
    input_hw,
    input_padding,
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
    roi_feat_extractor,
)

from hat.callbacks.metric_updater import update_metric_using_regex
from hat.core.proj_spec.descs import frcnn_gdl_desc

task_name = "vehicle_ground_line"


data_args = dict(legacy_bbox=True)

roi_args = dict(
    exclude_background=True,
    num_fg_classes=1,
    class_agnostic_reg=True,
    with_tracking_feat=False,
)


def get_model(mode):
    train_pred = deepcopy(anchor_pred)
    train_pred.update(
        pre_nms_top_k=2000,
        post_nms_top_k=128,
        use_clippings=False,
        nms_padding_mode="rollover",
        __graph_model_name=f"{task_name}_anchor_pred",
    )

    return dict(
        type="TwoStageDetector",
        backbone=backbone,
        neck=dict(type="ExtSequential", modules=[fpn_neck, fix_channel_neck]),
        rpn_out_keys=[] if mode != "train" else None,
        rpn_module=dict(
            type="AnchorModule",
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
            target_keys=["gt_boxes", "gt_flanks"],
            target_opt_keys=["gt_boxes_num", "gt_flanks_num"],
            roi_feat_extractor=roi_feat_extractor,
            head=dict(
                type="RCNNVarGNetHead",
                with_box_cls=True,
                with_box_reg=True,
                num_fg_classes=roi_args["num_fg_classes"],
                bn_kwargs=bn_kwargs,
                class_agnostic_reg=roi_args["class_agnostic_reg"],
                with_background=not roi_args["exclude_background"],
                roi_out_channel=64,
                dw_num_filter=128,
                pw_num_filter=128,
                pw_num_filter2=96,
                group_base=8,
                factor=1,
                stride_step1=1,
                reg_channel_base=2,
                __graph_model_name=f"{task_name}_roi_head",
            ),
            target=dict(
                type="ProposalTargetGroundLine",
                matcher=dict(
                    type="MaxIoUMatcher",
                    pos_iou=0.7,
                    neg_iou=0.5,
                    allow_low_quality_match=False,
                    low_quality_match_iou=0.5,
                    legacy_bbox=data_args["legacy_bbox"],
                ),
                label_encoder=dict(
                    type="MatchLabelGroundLineEncoder",
                    limit_reg_length=False,
                    cls_use_pos_only=True,
                    cls_on_hard=False,
                    reg_on_hard=False,
                ),
                __graph_model_name=f"{task_name}_roi_target",
            )
            if mode == "train"
            else None,
            loss=dict(
                type="RCNNLoss",
                cls_loss=dict(type="FocalLossV2", from_logits=True),
                reg_loss=dict(type="SmoothL1Loss", loss_weight=2),
                __graph_model_name=f"{task_name}_roi_loss",
            )
            if mode == "train"
            else None,
            head_desc=dict(
                type="AddDesc",
                per_tensor_desc=frcnn_gdl_desc(
                    task_name="frcnn_gdl_detection",
                    class_names=classnames,
                    label_output_name=f"{task_name}_label",
                    reg_output_name=f"{task_name}_reg",
                ),
                __graph_model_name=f"{task_name}_roi_desc",
            )
            if mode != "train"
            else None,
        ),
    )


# inputs
inputs = dict(
    train=dict(
        gt_boxes=torch.zeros((1, 100, 5)),
        gt_boxes_num=torch.zeros(1),
        gt_flanks=torch.zeros((1, 10, 9)),
        gt_flanks_num=torch.zeros(1),
        ig_regions=torch.zeros((1, 110, 5)),
        ig_regions_num=torch.zeros(1),
        im_hw=torch.zeros((1, 2)),
    ),
    val=dict(),
    test=dict(),
)


metric_updater = dict(
    type="MetricUpdater",
    metrics=[
        dict(type="LossShow", name="rcnn_cls_loss"),
        dict(type="LossShow", name="rcnn_reg_loss"),
    ],
    metric_update_func=update_metric_using_regex(
        per_metric_patterns=[  # corresponding to metrics
            dict(
                label_pattern=None,
                pred_pattern=f"^.*{task_name}_rcnn_cls_loss$",
            ),
            dict(
                label_pattern=None,
                pred_pattern=f"^.*{task_name}_rcnn_reg_loss$",
            ),
        ]
    ),
    step_log_freq=log_freq,
    epoch_log_freq=1,
    log_prefix=task_name,
    reset_metrics_by="log",
)


# data
ds = datapaths.vehicle_ground_line
rec_paths = [d["rec_path"] for d in ds["train_data_paths"]]
anno_paths = [d["anno_path"] for d in ds["train_data_paths"]]
sample_weights = [d["sample_weight"] for d in ds["train_data_paths"]]

data_loader = dict(
    type="GluonDataLoader",
    __build_recursive=False,
    input_padding=input_padding,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0,
    last_batch="rollover",
    dataset=[
        dict(
            type="SplitDataset",
            dataset=dict(
                type="KPSDataset",
                img_rec_path=rec_path_i,
                anno_rec_path=anno_path_i,
                to_rgb=True,
            ),
            even_split=False,
        )
        for rec_path_i, anno_path_i in zip(rec_paths, anno_paths)
    ],
    transform=[
        dict(
            type="VehicleFlankRoiTransform",
            # roi transform
            target_wh=input_hw[::-1],
            resize_wh=None if resize_hw is None else resize_hw[::-1],
            img_scale_range=(0.5, 2.0),
            roi_scale_range=(0.7, 1.0 / 0.7),
            min_sample_num=1,
            max_sample_num=1,
            center_aligned=False,
            inter_method=10,
            use_pyramid=True,
            pyramid_min_step=0.7,
            pyramid_max_step=0.8,
            min_valid_area=80,
            min_valid_clip_area_ratio=min_valid_clip_area_ratio,
            min_edge_size=10,
            rand_translation_ratio=rand_translation_ratio,
            rand_aspect_ratio=0.0,
            rand_rotation_angle=0,
            flip_prob=0.5,
            pixel_center_aligned=pixel_center_aligned,
            keep_aspect_ratio=True,
            min_flank_width=8,
            min_flank_width_overlap=0.2,
        ),
        dict(
            type="PadVehicleFlankData",
            target_wh=input_hw[::-1],
            max_gt_boxes_num=200,
            max_ig_regions_num=100,
        ),
        dict(
            type="PadGDLGTBeforeLabelGenerator",
            pad_width=input_padding[:2],
        ),
        dict(type="CastEx", dtypes=(None,) + (np.float32,) * 7),
        dict(
            type="ToDict",
            keys=(
                "img",
                "im_hw",
                "gt_boxes",
                "gt_boxes_num",
                "gt_flanks",
                "gt_flanks_num",
                "ig_regions",
                "ig_regions_num",
            ),
        ),
    ],
)
