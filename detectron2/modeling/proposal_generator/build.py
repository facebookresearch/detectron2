# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from detectron2.utils.registry import Registry

PROPOSAL_GENERATOR_REGISTRY = Registry("PROPOSAL_GENERATOR")
"""
Registry for proposal generator, which produces object proposals from feature maps.
"""

from . import rpn, rrpn  # noqa F401 isort:skip


def build_proposal_generator(cfg, input_shape):
    """
    Build a proposal generator from `cfg.MODEL.PROPOSAL_GENERATOR.NAME`.
    The name can be "PrecomputedProposals" to use no proposal generator.
    """
    name = cfg.MODEL.PROPOSAL_GENERATOR.NAME
    if name == "PrecomputedProposals":
        return None

    return PROPOSAL_GENERATOR_REGISTRY.get(name)(cfg, input_shape)
