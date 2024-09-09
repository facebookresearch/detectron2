from fvcore.common.param_scheduler import MultiStepParamScheduler

from detectron2.config import LazyCall as L
from detectron2.solver import WarmupParamScheduler

from .cascade_mask_rcnn_mvitv2_b_in21k_100ep import (
    dataloader,
    lr_multiplier,
    model,
    train,
    optimizer,
)

model.backbone.bottom_up.embed_dim = 192
model.backbone.bottom_up.depth = 80
model.backbone.bottom_up.num_heads = 3
model.backbone.bottom_up.last_block_indexes = (3, 11, 71, 79)
model.backbone.bottom_up.drop_path_rate = 0.6
model.backbone.bottom_up.use_act_checkpoint = True


train.init_checkpoint = "detectron2://ImageNetPretrained/mvitv2/MViTv2_H_in21k.pyth"


# 36 epochs
train.max_iter = 67500
lr_multiplier = L(WarmupParamScheduler)(
    scheduler=L(MultiStepParamScheduler)(
        values=[1.0, 0.1, 0.01],
        milestones=[
            52500,
            62500,
            67500,
        ],
    ),
    warmup_length=250 / train.max_iter,
    warmup_factor=0.001,
)
optimizer.lr = 1.6e-4
