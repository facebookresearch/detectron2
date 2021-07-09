from .mask_rcnn_regnety_4gf_dds_FPN_100ep_LSJ import (
    dataloader,
    lr_multiplier,
    model,
    optimizer,
    train,
)

train.max_iter *= 4  # 100ep -> 400ep

lr_multiplier.scheduler.milestones = [
    milestone * 4 for milestone in lr_multiplier.scheduler.milestones
]
lr_multiplier.scheduler.num_updates = train.max_iter
