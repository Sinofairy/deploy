# flake8: noqa

import numpy as np
import torch
from common import (
    backbone,
    batch_size,
    bn_kwargs,
    datapaths,
    fpn_neck,
    log_freq,
    resize_hw,
    roi_region,
    ufpn_seg_neck,
    vanishing_point,
)

from hat.callbacks.metric_updater import update_metric_using_regex
from hat.core.proj_spec.descs import get_lane_parsing_desc
from hat.data.collates import collate_2d

task_name = "lane_segmentation"

# model
head_out_strides = [2, 8, 16, 32, 64]
num_classes = 5


def get_model(mode):
    out_strides = (
        head_out_strides if mode == "train" else [min(head_out_strides)]
    )

    return dict(
        type="Segmentor",
        backbone=backbone,
        neck=dict(type="ExtSequential", modules=[fpn_neck, ufpn_seg_neck]),
        head=dict(
            type="FRCNNSegHead",
            group_base=4,
            in_strides=[4, 8, 16, 32, 64],
            in_channels=[16, 32, 64, 128, 128],
            out_strides=out_strides,
            out_channels=[num_classes] * len(out_strides),
            bn_kwargs=bn_kwargs,
            argmax_output=mode != "train",
            dequant_output=mode == "train",
            with_extra_conv=False,
            __graph_model_name=f"{task_name}_head",
        ),
        loss=dict(
            type="MultiStrideLosses",
            num_classes=num_classes,
            out_strides=head_out_strides,
            loss=dict(
                type="WeightedSquaredHingeLoss",
                reduction="mean",
                weight_low_thr=0.6,
                weight_high_thr=1.0,
                hard_neg_mining_cfg=dict(
                    keep_pos=True,
                    neg_ratio=0.999,
                    hard_ratio=1.0,
                    min_keep_num=255,
                ),
            ),
            loss_weights=[4, 2, 2, 2, 2],
            __graph_model_name=f"{task_name}_loss",
        )
        if mode == "train"
        else None,
        desc=dict(
            type="AddDesc",
            per_tensor_desc=get_lane_parsing_desc(
                desc_id=f"wd_{num_classes}",
                roi_regions=roi_region,
                vanishing_point=vanishing_point,
            ),
            __graph_model_name=f"{task_name}_desc",
        )
        if mode is not "train"
        else None,
    )


# inputs
inputs = dict(
    train=dict(
        labels=[
            torch.zeros((1, 1, resize_hw[0] // s, resize_hw[1] // s))
            for s in head_out_strides
        ],
    ),
    val=dict(),
    test=dict(),
)


metric_updater = dict(
    type="MetricUpdater",
    metrics=[
        dict(type="LossShow", name=f"stride_{s}_loss")
        for s in head_out_strides
    ],
    metric_update_func=update_metric_using_regex(
        per_metric_patterns=[  # corresponding to metrics
            dict(
                label_pattern=None,
                pred_pattern=f"^.*{task_name}_stride_{s}_loss$",
            )
            for s in head_out_strides
        ]
    ),
    step_log_freq=log_freq,
    epoch_log_freq=1,
    log_prefix=task_name,
    reset_metrics_by="log",
)


ds = datapaths.lane_parsing

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
                        type="SemanticSegAffineAugTransformerEx",
                        target_wh=resize_hw[::-1],
                        inter_method=10,
                        label_scales=[
                            1.0 / stride_i for stride_i in head_out_strides
                        ],
                        use_pyramid=True,
                        pyramid_min_step=0.7,
                        pyramid_max_step=0.8,
                        flip_prob=0.5,
                        label_padding_value=-1,
                        rand_translation_ratio=0.0,
                        center_aligned=False,
                        rand_scale_range=(0.8, 1.3),
                        resize_wh=resize_hw[::-1],
                        adapt_diff_resolution=True,
                    ),
                ],
            )
            for path in ds.train_data_paths
        ],
    ),
)
