from detectron2 import model_zoo

from .cascade_mask_rcnn_mvitv2_t_3x import dataloader, optimizer, lr_multiplier, train

model = model_zoo.get_config("MViTv2/configs/cascade_mask_rcnn_mvitv2_t_3x.py").model

model.backbone.bottom_up.depth = 16
model.backbone.bottom_up.last_block_indexes = (0, 2, 13, 15)

train.init_checkpoint = "detectron2://ImageNetPretrained/mvitv2/MViTv2_S_in1k.pyth"
