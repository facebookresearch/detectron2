from .mask_rcnn_BNhead import model, dataloader, lr_multiplier, optimizer, train

model.roi_heads.box_head.conv_norm = model.roi_heads.mask_head.conv_norm = "SyncBN"
