from .cascade_mask_rcnn_mvitv2_b_3x import dataloader, lr_multiplier, model, optimizer, train

train.init_checkpoint = "detectron2://ImageNetPretrained/mvitv2/MViTv2_B_in21k.pyth"
