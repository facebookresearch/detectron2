from .cascade_mask_rcnn_swin_b_in21k_50ep import (
    dataloader,
    lr_multiplier,
    model,
    train,
    optimizer,
)

model.backbone.bottom_up.embed_dim = 192
model.backbone.bottom_up.num_heads = [6, 12, 24, 48]

train.init_checkpoint = "detectron2://ImageNetPretrained/swin/swin_large_patch4_window7_224_22k.pth"
