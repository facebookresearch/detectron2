from detectron2.modeling.meta_arch import GeneralizedRCNN

from .mask_rcnn_dinov3_vitb16_frozen_tiled_v1_from_v0_best_ap import (
    dataloader,
    lr_multiplier,
    model,
    optimizer,
    train,
)


_EVAL_OUTPUT_DIR = (
    "./output/oweed_v1_tiled_from_v0_best_ap_dinov3_vitb16_frozen_mask_rcnn/"
    "eval_mask_best_bbox_ap"
)

# Use the normal Mask R-CNN inference path for this one-off evaluation so COCO
# metrics include both bounding boxes and instance masks.
model._target_ = GeneralizedRCNN

dataloader.evaluator.tasks = ("bbox", "segm")
dataloader.evaluator.output_dir = _EVAL_OUTPUT_DIR

train.init_checkpoint = (
    "./output/oweed_v1_tiled_from_v0_best_ap_dinov3_vitb16_frozen_mask_rcnn/"
    "model_best_bbox_ap.pth"
)
train.output_dir = _EVAL_OUTPUT_DIR
