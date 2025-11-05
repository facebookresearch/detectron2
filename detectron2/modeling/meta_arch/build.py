# Copyright (c) Facebook, Inc. and its affiliates.
import torch

from detectron2.utils.comm import _TORCH_NPU_AVAILABLE
from detectron2.utils.logger import _log_api_usage
from detectron2.utils.registry import Registry

META_ARCH_REGISTRY = Registry("META_ARCH")  # noqa F401 isort:skip
META_ARCH_REGISTRY.__doc__ = """
Registry for meta-architectures, i.e. the whole model.

The registered object will be called with `obj(cfg)`
and expected to return a `nn.Module` object.
"""


def build_model(cfg):
    """
    Build the whole model architecture, defined by ``cfg.MODEL.META_ARCHITECTURE``.
    Note that it does not load any weights from ``cfg``.
    """
    # TODO (cmq): The support of dynamic ops in torch-npu is limited.
    # Not supported kernel size [h=32, w=64] in Conv2DBackprop dynamic ops,
    # revert me after supported
    if "npu" in cfg.MODEL.DEVICE and _TORCH_NPU_AVAILABLE:
        torch.npu.set_compile_mode(jit_compile=True)

    meta_arch = cfg.MODEL.META_ARCHITECTURE
    model = META_ARCH_REGISTRY.get(meta_arch)(cfg)
    model.to(torch.device(cfg.MODEL.DEVICE))
    _log_api_usage("modeling.meta_arch." + meta_arch)
    return model
