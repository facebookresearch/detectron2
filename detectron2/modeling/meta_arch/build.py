# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from detectron2.utils.registry import Registry

META_ARCH_REGISTRY = Registry("META_ARCH")  # noqa F401 isort:skip
"""
Registry for meta-architectures, i.e. the whole model.
"""


def build_model(cfg):
    """
    Built the whole model, defined by `cfg.MODEL.META_ARCHITECTURE`.
    """
    meta_arch = cfg.MODEL.META_ARCHITECTURE
    return META_ARCH_REGISTRY.get(meta_arch)(cfg)
