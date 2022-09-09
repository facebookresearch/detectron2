from functools import partial

from .mask_rcnn_vitdet_b_100ep import (
    dataloader,
    lr_multiplier,
    model,
    train,
    optimizer,
    get_vit_lr_decay_rate,
)

train.init_checkpoint = "detectron2://ImageNetPretrained/MAE/mae_pretrain_vit_huge_p14to16.pth"

model.backbone.net.embed_dim = 1280
model.backbone.net.depth = 32
model.backbone.net.num_heads = 16
model.backbone.net.drop_path_rate = 0.5
# 7, 15, 23, 31 for global attention
model.backbone.net.window_block_indexes = (
    list(range(0, 7)) + list(range(8, 15)) + list(range(16, 23)) + list(range(24, 31))
)

optimizer.params.lr_factor_func = partial(get_vit_lr_decay_rate, lr_decay_rate=0.9, num_layers=32)
optimizer.params.overrides = {}
optimizer.params.weight_decay_norm = None

train.max_iter = train.max_iter * 3 // 4  # 100ep -> 75ep
lr_multiplier.scheduler.milestones = [
    milestone * 3 // 4 for milestone in lr_multiplier.scheduler.milestones
]
lr_multiplier.scheduler.num_updates = train.max_iter
