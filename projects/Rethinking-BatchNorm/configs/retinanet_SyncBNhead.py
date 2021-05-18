from detectron2.model_zoo import get_config

model = get_config("common/models/retinanet.py").model
model.backbone.bottom_up.freeze_at = 2
model.head.norm = "SyncBN"

dataloader = get_config("common/data/coco.py").dataloader
lr_multiplier = get_config("common/coco_schedule.py").lr_multiplier_3x
optimizer = get_config("common/optim.py").SGD
train = get_config("common/train.py").train

optimizer.lr = 0.01
train.max_iter = 270000  # 3x for batchsize = 16
