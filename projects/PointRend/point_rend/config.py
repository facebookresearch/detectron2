# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.

from detectron2.config import CfgNode as CN


def add_pointrend_config(cfg):
    """
    Add config for PointRend.
    """
    # We retry random cropping until no single category in semantic segmentation GT occupies more
    # than `SINGLE_CATEGORY_MAX_AREA` part of the crop.
    cfg.INPUT.CROP.SINGLE_CATEGORY_MAX_AREA = 1.0
    # Color augmentatition from SSD paper for semantic segmentation model during training.
    cfg.INPUT.COLOR_AUG_SSD = False

    # Names of the input feature maps to be used by a coarse mask head.
    cfg.MODEL.ROI_MASK_HEAD.IN_FEATURES = ("p2",)
    cfg.MODEL.ROI_MASK_HEAD.FC_DIM = 1024
    cfg.MODEL.ROI_MASK_HEAD.NUM_FC = 2
    # The side size of a coarse mask head prediction.
    cfg.MODEL.ROI_MASK_HEAD.OUTPUT_SIDE_RESOLUTION = 7
    # True if point head is used.
    cfg.MODEL.ROI_MASK_HEAD.POINT_HEAD_ON = False

    cfg.MODEL.POINT_HEAD = CN()
    cfg.MODEL.POINT_HEAD.NAME = "StandardPointHead"
    cfg.MODEL.POINT_HEAD.NUM_CLASSES = 80
    # Names of the input feature maps to be used by a mask point head.
    cfg.MODEL.POINT_HEAD.IN_FEATURES = ("p2",)
    # Number of points sampled during training for a mask point head.
    cfg.MODEL.POINT_HEAD.TRAIN_NUM_POINTS = 14 * 14
    # Oversampling parameter for PointRend point sampling during training. Parameter `k` in the
    # original paper.
    cfg.MODEL.POINT_HEAD.OVERSAMPLE_RATIO = 3
    # Importance sampling parameter for PointRend point sampling during training. Parametr `beta` in
    # the original paper.
    cfg.MODEL.POINT_HEAD.IMPORTANCE_SAMPLE_RATIO = 0.75
    # Number of subdivision steps during inference.
    cfg.MODEL.POINT_HEAD.SUBDIVISION_STEPS = 5
    # Maximum number of points selected at each subdivision step (N).
    cfg.MODEL.POINT_HEAD.SUBDIVISION_NUM_POINTS = 28 * 28
    cfg.MODEL.POINT_HEAD.FC_DIM = 256
    cfg.MODEL.POINT_HEAD.NUM_FC = 3
    cfg.MODEL.POINT_HEAD.CLS_AGNOSTIC_MASK = False
    # If True, then coarse prediction features are used as inout for each layer in PointRend's MLP.
    cfg.MODEL.POINT_HEAD.COARSE_PRED_EACH_LAYER = True
    cfg.MODEL.POINT_HEAD.COARSE_SEM_SEG_HEAD_NAME = "SemSegFPNHead"

    """
    Add config for Implicit PointRend.
    """
    cfg.MODEL.IMPLICIT_POINTREND = CN()

    cfg.MODEL.IMPLICIT_POINTREND.IMAGE_FEATURE_ENABLED = True
    cfg.MODEL.IMPLICIT_POINTREND.POS_ENC_ENABLED = True

    cfg.MODEL.IMPLICIT_POINTREND.PARAMS_L2_REGULARIZER = 0.00001
