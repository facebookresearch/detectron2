from functools import partial
import torch.nn as nn
from fvcore.common.param_scheduler import MultiStepParamScheduler

from detectron2 import model_zoo
from detectron2.config import LazyCall as L
from detectron2.solver import WarmupParamScheduler
from detectron2.modeling import MViT

from .common.coco_loader import dataloader

model = model_zoo.get_config("common/models/mask_rcnn_fpn.py").model
constants = model_zoo.get_config("common/data/constants.py").constants
model.pixel_mean = constants.imagenet_rgb256_mean
model.pixel_std = constants.imagenet_rgb256_std
model.input_format = "RGB"
model.backbone.bottom_up = L(MViT)(
    embed_dim=96,
    depth=10,
    num_heads=1,
    last_block_indexes=(0, 2, 7, 9),
    residual_pooling=True,
    drop_path_rate=0.2,
    norm_layer=partial(nn.LayerNorm, eps=1e-6),
    out_features=("scale2", "scale3", "scale4", "scale5"),
)
model.backbone.in_features = "${.bottom_up.out_features}"


# Initialization and trainer settings
train = model_zoo.get_config("common/train.py").train
train.amp.enabled = True
train.ddp.fp16_compression = True
train.init_checkpoint = "detectron2://ImageNetPretrained/mvitv2/MViTv2_T_in1k.pyth"

dataloader.train.total_batch_size = 64

# 36 epochs
train.max_iter = 67500
lr_multiplier = L(WarmupParamScheduler)(
    scheduler=L(MultiStepParamScheduler)(
        values=[1.0, 0.1, 0.01],
        milestones=[52500, 62500, 67500],
    ),
    warmup_length=250 / train.max_iter,
    warmup_factor=0.001,
)

optimizer = model_zoo.get_config("common/optim.py").AdamW
optimizer.params.overrides = {
    "pos_embed": {"weight_decay": 0.0},
    "rel_pos_h": {"weight_decay": 0.0},
    "rel_pos_w": {"weight_decay": 0.0},
}
optimizer.lr = 1.6e-4
