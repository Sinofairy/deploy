import os

import torch
from common import val_transforms

from hat.core.task_sampler import TaskSampler

tasks = ["vehicle"]
val_task_config = dict()
val_dataloaders = dict()
for task_name in tasks:
    val_task_config[task_name] = dict(sampling_factor=1)
    val_data_loader = dict(
        type=torch.utils.data.DataLoader,
        dataset=dict(
            type="FrameDataset",
            img_path="1.jpg",
            buf_only=True,
            transforms=val_transforms,
        ),
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
    )
    val_dataloaders[task_name] = val_data_loader

val_task_sampler = TaskSampler(
    task_config=val_task_config, method="sample_all"
)

val_data_loader = dict(
    type="MultitaskLoader",
    loaders=val_dataloaders,
    task_sampler=val_task_sampler,
    mode="validation",
    return_task=True,
    custom_length=None,
)
