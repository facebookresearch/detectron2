from .mask_rcnn_R_50_FPN_100ep_LSJ import (
    dataloader,
    lr_multiplier,
    model,
    optimizer,
    train,
)

model.backbone.bottom_up.stages.depth = 101
