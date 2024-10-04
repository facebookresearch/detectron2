from detectron2 import model_zoo

from ..common.train import train
from ..common.optim import SGD as optimizer
from ..common.coco_schedule import lr_multiplier_1x as lr_multiplier
from ..common.data.coco import dataloader

model = model_zoo.get_config("common/models/mask_rcnn_c4.py").model

model.backbone.freeze_at = 2
train.init_checkpoint = "detectron2://ImageNetPretrained/MSRA/R-50.pkl"
