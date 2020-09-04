# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from detectron2.config import CfgNode

from .filter import DensePoseDataFilter
from .losses import DensePoseLosses
from .predictors import DensePoseChartWithConfidencePredictor


def build_densepose_predictor(cfg: CfgNode, input_channels: int):
    """
    Create an instance of DensePose predictor based on configuration options.

    Args:
        cfg (CfgNode): configuration options
        input_channels (int): input tensor size along the channel dimension
    Return:
        An instance of DensePose predictor
    """
    predictor = DensePoseChartWithConfidencePredictor(cfg, input_channels)
    return predictor


def build_densepose_data_filter(cfg: CfgNode):
    """
    Build DensePose data filter which selects data for training

    Args:
        cfg (CfgNode): configuration options

    Return:
        Callable: list(Tensor), list(Instances) -> list(Tensor), list(Instances)
        An instance of DensePose filter, which takes feature tensors and proposals
        as an input and returns filtered features and proposals
    """
    dp_filter = DensePoseDataFilter(cfg)
    return dp_filter


def build_densepose_head(cfg: CfgNode, input_channels: int):
    """
    Build DensePose head based on configurations options

    Args:
        cfg (CfgNode): configuration options
        input_channels (int): input tensor size along the channel dimension
    Return:
        An instance of DensePose head
    """
    from .roi_heads.registry import ROI_DENSEPOSE_HEAD_REGISTRY

    head_name = cfg.MODEL.ROI_DENSEPOSE_HEAD.NAME
    return ROI_DENSEPOSE_HEAD_REGISTRY.get(head_name)(cfg, input_channels)


def build_densepose_losses(cfg: CfgNode):
    """
    Build DensePose loss based on configurations options

    Args:
        cfg (CfgNode): configuration options
    Return:
        An instance of DensePose loss
    """
    losses = DensePoseLosses(cfg)
    return losses
