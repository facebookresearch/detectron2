# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from detectron2.layers import ShapeSpec
from detectron2.utils.registry import Registry

from .backbone import Backbone

BACKBONE_REGISTRY = Registry("BACKBONE")
"""
Registry for backbones, which extract feature maps from images.
"""


def build_backbone(cfg, input_shape=None):
    """
    Build a backbone from `cfg.MODEL.BACKBONE.NAME`.

    Returns:
        an instance of :class:`Backbone`
    """
    if input_shape is None:
        input_shape = ShapeSpec(channels=len(cfg.MODEL.PIXEL_MEAN))

    backbone_name = cfg.MODEL.BACKBONE.NAME
    backbone = BACKBONE_REGISTRY.get(backbone_name)(cfg, input_shape)
    assert isinstance(backbone, Backbone)
    return backbone
