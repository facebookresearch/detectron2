from detectron2.model_zoo import get_config

model = get_config("common/models/mask_rcnn_fpn.py").model

model.backbone.bottom_up.freeze_at = 2

model.roi_heads.box_head.conv_norm = model.roi_heads.mask_head.conv_norm = "BN"
# 4conv1fc head
model.roi_heads.box_head.conv_dims = [256, 256, 256, 256]
model.roi_heads.box_head.fc_dims = [1024]

dataloader = get_config("common/data/coco.py").dataloader
lr_multiplier = get_config("common/coco_schedule.py").lr_multiplier_3x
optimizer = get_config("common/optim.py").SGD
train = get_config("common/train.py").train

train.init_checkpoint = "detectron2://ImageNetPretrained/MSRA/R-50.pkl"
train.max_iter = 270000  # 3x for batchsize = 16
