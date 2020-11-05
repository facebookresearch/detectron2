# Copyright (c) Facebook, Inc. and its affiliates.

from detectron2.config import CfgNode

from .filter import DensePoseDataFilter
from .losses import *  # noqa
from .predictors import *  # noqa


def build_densepose_predictor(cfg: CfgNode, input_channels: int):
    """
    Create an instance of DensePose predictor based on configuration options.

    Args:
        cfg (CfgNode): configuration options
        input_channels (int): input tensor size along the channel dimension
    Return:
        An instance of DensePose predictor
    """
    from .predictors.registry import DENSEPOSE_PREDICTOR_REGISTRY

    predictor_name = cfg.MODEL.ROI_DENSEPOSE_HEAD.PREDICTOR_NAME
    return DENSEPOSE_PREDICTOR_REGISTRY.get(predictor_name)(cfg, input_channels)


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
    from .losses.registry import DENSEPOSE_LOSS_REGISTRY

    loss_name = cfg.MODEL.ROI_DENSEPOSE_HEAD.LOSS_NAME
    return DENSEPOSE_LOSS_REGISTRY.get(loss_name)(cfg)
