from ..common.optim import SGD as optimizer
from ..common.coco_schedule import lr_multiplier_1x as lr_multiplier
from ..common.data.coco_keypoint import dataloader
from ..common.models.keypoint_rcnn_fpn import model
from ..common.train import train

model.backbone.bottom_up.freeze_at = 2
