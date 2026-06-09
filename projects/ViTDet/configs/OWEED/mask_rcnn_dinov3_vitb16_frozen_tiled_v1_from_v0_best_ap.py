from fvcore.common.param_scheduler import MultiStepParamScheduler

import detectron2.data.transforms as T
from detectron2.config import LazyCall as L
from detectron2.data.datasets import register_coco_instances
from detectron2.modeling.meta_arch import GeneralizedRCNNWithBBoxOnlyEval
from detectron2.solver import WarmupParamScheduler

from ..COCO.mask_rcnn_dinov3_vitb16_100ep import (
    dataloader,
    lr_multiplier,
    model,
    optimizer,
    train,
)


_DATA_ROOT = "/home/nikhileswara/Datasets/oweed_structure_v1_177_images_coco_tiled"
_TRAIN_NAME = "oweed_structure_v1_tiled_train"
_VAL_NAME = "oweed_structure_v1_tiled_val"

register_coco_instances(
    _TRAIN_NAME,
    {},
    f"{_DATA_ROOT}/annotations/instances_train2017.json",
    f"{_DATA_ROOT}/train2017",
)
register_coco_instances(
    _VAL_NAME,
    {},
    f"{_DATA_ROOT}/annotations/instances_val2017.json",
    f"{_DATA_ROOT}/val2017",
)

dataloader.train.dataset.names = _TRAIN_NAME
dataloader.train.total_batch_size = 16
dataloader.train.num_workers = 16
dataloader.train.pin_memory = True
dataloader.train.persistent_workers = True
dataloader.train.prefetch_factor = 4
dataloader.train.mapper.augmentations = [
    L(T.RandomFlip)(horizontal=True),
    L(T.ResizeScale)(
        min_scale=0.8,
        max_scale=1.25,
        target_height=1024,
        target_width=1024,
    ),
    L(T.FixedSizeCrop)(crop_size=(1024, 1024), pad=True),
]
dataloader.train.mapper.image_format = "RGB"
dataloader.train.mapper.recompute_boxes = True

dataloader.test.dataset.names = _VAL_NAME
dataloader.test.num_workers = 16
dataloader.test.mapper.augmentations = [
    L(T.ResizeShortestEdge)(short_edge_length=1024, max_size=1024),
]

dataloader.evaluator.dataset_name = _VAL_NAME
dataloader.evaluator.tasks = ("bbox",)
dataloader.evaluator.max_dets_per_image = 500

model._target_ = GeneralizedRCNNWithBBoxOnlyEval
model.roi_heads.num_classes = 44
model.roi_heads.batch_size_per_image = 1024
model.roi_heads.positive_fraction = 0.5
model.roi_heads.box_predictor.test_score_thresh = 0.02
model.roi_heads.box_predictor.test_nms_thresh = 0.7
model.roi_heads.box_predictor.test_topk_per_image = 500

model.proposal_generator.anchor_generator.sizes = [[8], [16], [32], [64], [128]]
model.proposal_generator.batch_size_per_image = 512
model.proposal_generator.positive_fraction = 0.5
model.proposal_generator.pre_nms_topk = (4000, 4000)
model.proposal_generator.post_nms_topk = (2000, 2000)

optimizer.lr = 5e-5

train.init_checkpoint = (
    "./output/oweed_tiled_keepfrag_dinov3_vitb16_frozen_mask_rcnn/model_0047999.pth"
)
train.output_dir = "./output/oweed_v1_tiled_from_v0_best_ap_dinov3_vitb16_frozen_mask_rcnn"
train.max_iter = 15000
train.eval_period = 1000
train.checkpointer.period = 1000
train.checkpointer.max_to_keep = 10
train.best_checkpointer = dict(
    val_metric="bbox/AP",
    mode="max",
    file_prefix="model_best_bbox_ap",
)

lr_multiplier = L(WarmupParamScheduler)(
    scheduler=L(MultiStepParamScheduler)(
        values=[1.0, 0.1, 0.01],
        milestones=[10000, 13500],
        num_updates=train.max_iter,
    ),
    warmup_length=300 / train.max_iter,
    warmup_factor=0.001,
)
