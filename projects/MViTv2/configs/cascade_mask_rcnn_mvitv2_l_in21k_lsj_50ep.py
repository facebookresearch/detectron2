from fvcore.common.param_scheduler import MultiStepParamScheduler

from detectron2.config import LazyCall as L
from detectron2.solver import WarmupParamScheduler

from .cascade_mask_rcnn_mvitv2_b_3x import model, optimizer, train
from .common.coco_loader_lsj import dataloader


model.backbone.bottom_up.embed_dim = 144
model.backbone.bottom_up.depth = 48
model.backbone.bottom_up.num_heads = 2
model.backbone.bottom_up.last_block_indexes = (1, 7, 43, 47)
model.backbone.bottom_up.drop_path_rate = 0.5

train.init_checkpoint = "detectron2://ImageNetPretrained/mvitv2/MViTv2_L_in21k.pyth"

# Schedule
# 50ep = 184375 // 2  iters * 64 images/iter / 118000 images/ep
train.max_iter = 184375 // 2
lr_multiplier = L(WarmupParamScheduler)(
    scheduler=L(MultiStepParamScheduler)(
        values=[1.0, 0.1, 0.01],
        milestones=[163889 // 2, 177546 // 2],
        num_updates=train.max_iter,
    ),
    warmup_length=250 / train.max_iter,
    warmup_factor=0.001,
)

optimizer.lr = 1e-4
