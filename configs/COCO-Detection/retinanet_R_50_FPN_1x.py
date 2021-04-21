from ..common.optim import SGD as optimizer
from ..common.coco_schedule import lr_multiplier_1x as lr_multiplier
from ..common.data.coco import dataloader
from ..common.models.retinanet import model
from ..common.train import train

dataloader.train.mapper.use_instance_mask = False
model.backbone.bottom_up.freeze_at = 2
optimizer.lr = 0.01
