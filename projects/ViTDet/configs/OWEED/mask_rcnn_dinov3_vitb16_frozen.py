from detectron2.data.datasets import register_coco_instances
from detectron2.modeling.meta_arch import GeneralizedRCNNWithBBoxOnlyEval

from ..COCO.mask_rcnn_dinov3_vitb16_100ep import (
    dataloader,
    lr_multiplier,
    model,
    optimizer,
    train,
)


_DATA_ROOT = "/home/nikhileswara/Datasets/oweed_structure_v0_100_images_coco"
_TRAIN_NAME = "oweed_structure_v0_train"
_VAL_NAME = "oweed_structure_v0_val"

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
# 2 RTX 4090s: total_batch_size is global across DDP workers, so this gives
# two 1024x1024 LSJ images per GPU when launched with --num-gpus 2.
dataloader.train.total_batch_size = 4
dataloader.train.num_workers = 8
dataloader.train.pin_memory = True
dataloader.train.persistent_workers = True
dataloader.train.prefetch_factor = 4
dataloader.test.dataset.names = _VAL_NAME
dataloader.test.num_workers = 8

dataloader.evaluator.dataset_name = _VAL_NAME
dataloader.evaluator.tasks = ("bbox",)

model._target_ = GeneralizedRCNNWithBBoxOnlyEval
model.roi_heads.num_classes = 44
model.roi_heads.box_predictor.test_score_thresh = 0.02
model.roi_heads.box_predictor.test_topk_per_image = 500

# The dataset has up to ~460 annotated instances per image, so keep more
# proposals and final detections than the COCO defaults.
model.proposal_generator.pre_nms_topk = (4000, 4000)
model.proposal_generator.post_nms_topk = (2000, 2000)

optimizer.lr = 1e-4

train.output_dir = "./output/oweed_dinov3_vitb16_frozen_mask_rcnn"
train.max_iter = 5000
train.eval_period = 1000
train.checkpointer.period = 500
train.checkpointer.max_to_keep = 10

lr_multiplier.scheduler.milestones = [3500, 4500]
lr_multiplier.scheduler.num_updates = train.max_iter
lr_multiplier.warmup_length = 100 / train.max_iter
