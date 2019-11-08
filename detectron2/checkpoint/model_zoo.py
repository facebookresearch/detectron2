# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import os

from detectron2.config import get_cfg
from detectron2.modeling import build_model

from .detection_checkpoint import DetectionCheckpointer

CONFIGS_DIR = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), os.pardir, os.pardir, "configs"
)


def faster_rcnn_R_50_C4_1x(pretrained: bool = False):
    config_path = os.path.join(CONFIGS_DIR, "COCO-Detection", "faster_rcnn_R_50_C4_1x.yaml")
    cfg = get_cfg()
    cfg.merge_from_file(config_path)
    cfg.MODEL.WEIGHTS = "detectron2://COCO-Detection/faster_rcnn_R_50_C4_1x"
    cfg.freeze()
    model = build_model(cfg)

    if pretrained:
        DetectionCheckpointer(model).resume_or_load(cfg.MODEL.WEIGHTS, resume=False)
    return model
