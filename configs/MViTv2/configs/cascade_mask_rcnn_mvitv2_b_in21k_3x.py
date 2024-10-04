from detectron2 import model_zoo

from .cascade_mask_rcnn_mvitv2_b_3x import dataloader, optimizer, lr_multiplier, train

model = model_zoo.get_config("MViTv2/configs/cascade_mask_rcnn_mvitv2_b_3x.py").model

train.init_checkpoint = "detectron2://ImageNetPretrained/mvitv2/MViTv2_B_in21k.pyth"
