from detectron2.model_zoo import get_config
from torch import nn

model = get_config("common/models/retinanet.py").model
model.backbone.bottom_up.freeze_at = 2

# The head will overwrite string "SyncBN" to use domain-specific BN, so we
# provide a class here to use shared BN in training.
model.head.norm = nn.SyncBatchNorm2d

dataloader = get_config("common/data/coco.py").dataloader
lr_multiplier = get_config("common/coco_schedule.py").lr_multiplier_3x
optimizer = get_config("common/optim.py").SGD
train = get_config("common/train.py").train

optimizer.lr = 0.01

train.init_checkpoint = "detectron2://ImageNetPretrained/MSRA/R-50.pkl"
train.max_iter = 270000  # 3x for batchsize = 16
