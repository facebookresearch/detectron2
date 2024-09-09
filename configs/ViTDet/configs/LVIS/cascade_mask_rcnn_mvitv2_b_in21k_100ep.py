from functools import partial
import torch.nn as nn

from detectron2.config import LazyCall as L
from detectron2.data.detection_utils import get_fed_loss_cls_weights
from detectron2.data.samplers import RepeatFactorTrainingSampler
from detectron2.evaluation.lvis_evaluation import LVISEvaluator

from ..COCO.cascade_mask_rcnn_mvitv2_b_in21k_100ep import (
    dataloader,
    model,
    train,
    lr_multiplier,
    optimizer,
)

dataloader.train.dataset.names = "lvis_v1_train"
dataloader.train.sampler = L(RepeatFactorTrainingSampler)(
    repeat_factors=L(RepeatFactorTrainingSampler.repeat_factors_from_category_frequency)(
        dataset_dicts="${dataloader.train.dataset}", repeat_thresh=0.001
    )
)
dataloader.test.dataset.names = "lvis_v1_val"
dataloader.evaluator = L(LVISEvaluator)(
    dataset_name="${..test.dataset.names}",
    max_dets_per_image=300,
)

model.roi_heads.num_classes = 1203
for i in range(3):
    model.roi_heads.box_predictors[i].test_score_thresh = 0.02
    model.roi_heads.box_predictors[i].test_topk_per_image = 300
    model.roi_heads.box_predictors[i].use_sigmoid_ce = True
    model.roi_heads.box_predictors[i].use_fed_loss = True
    model.roi_heads.box_predictors[i].get_fed_loss_cls_weights = lambda: get_fed_loss_cls_weights(
        dataloader.train.dataset.names, 0.5
    )

# Schedule
# 100 ep = 156250 iters * 64 images/iter / 100000 images/ep
train.max_iter = 156250
train.eval_period = 30000

lr_multiplier.scheduler.milestones = [138889, 150463]
lr_multiplier.scheduler.num_updates = train.max_iter
lr_multiplier.warmup_length = 250 / train.max_iter

optimizer.lr = 1e-4
