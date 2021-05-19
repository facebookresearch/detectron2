from .mask_rcnn_R_101_FPN_100ep_LSJ import (
    dataloader,
    lr_multiplier,
    model,
    optimizer,
    train,
)

train.max_iter *= 4  # 100ep -> 400ep
