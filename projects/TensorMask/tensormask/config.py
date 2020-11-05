# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.

from detectron2.config import CfgNode as CN


def add_tensormask_config(cfg):
    """
    Add config for TensorMask.
    """
    cfg.MODEL.TENSOR_MASK = CN()

    # Anchor parameters
    cfg.MODEL.TENSOR_MASK.IN_FEATURES = ["p2", "p3", "p4", "p5", "p6", "p7"]

    # Convolutions to use in the towers
    cfg.MODEL.TENSOR_MASK.NUM_CONVS = 4

    # Number of foreground classes.
    cfg.MODEL.TENSOR_MASK.NUM_CLASSES = 80
    # Channel size for the classification tower
    cfg.MODEL.TENSOR_MASK.CLS_CHANNELS = 256

    cfg.MODEL.TENSOR_MASK.SCORE_THRESH_TEST = 0.05
    # Only the top (1000 * #levels) candidate boxes across all levels are
    # considered jointly during test (to improve speed)
    cfg.MODEL.TENSOR_MASK.TOPK_CANDIDATES_TEST = 6000
    cfg.MODEL.TENSOR_MASK.NMS_THRESH_TEST = 0.5

    # Box parameters
    # Channel size for the box tower
    cfg.MODEL.TENSOR_MASK.BBOX_CHANNELS = 128
    # Weights on (dx, dy, dw, dh)
    cfg.MODEL.TENSOR_MASK.BBOX_REG_WEIGHTS = (1.5, 1.5, 0.75, 0.75)

    # Loss parameters
    cfg.MODEL.TENSOR_MASK.FOCAL_LOSS_GAMMA = 3.0
    cfg.MODEL.TENSOR_MASK.FOCAL_LOSS_ALPHA = 0.3

    # Mask parameters
    # Channel size for the mask tower
    cfg.MODEL.TENSOR_MASK.MASK_CHANNELS = 128
    # Mask loss weight
    cfg.MODEL.TENSOR_MASK.MASK_LOSS_WEIGHT = 2.0
    # weight on positive pixels within the mask
    cfg.MODEL.TENSOR_MASK.POSITIVE_WEIGHT = 1.5
    # Whether to predict in the aligned representation
    cfg.MODEL.TENSOR_MASK.ALIGNED_ON = False
    # Whether to use the bipyramid architecture
    cfg.MODEL.TENSOR_MASK.BIPYRAMID_ON = False
