from .mask_rcnn_BNhead import dataloader, lr_multiplier, model, optimizer, train

model.roi_heads.box_head.conv_norm = model.roi_heads.mask_head.conv_norm = "SyncBN"
