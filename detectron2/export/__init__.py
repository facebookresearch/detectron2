# -*- coding: utf-8 -*-

import warnings

try:
    from caffe2.proto import caffe2_pb2 as _tmp

    # caffe2 is optional
except ImportError:
    pass
else:
    from .api import *

from .flatten import TracingAdapter
from .torchscript import scripting_with_instances, dump_torchscript_IR

SUPPORTED_ONNX_OPSET = 11


def add_export_config(cfg):
    warnings.warn("add_export_config has been deprecated and behaves as no-op function.")
    return cfg


__all__ = [k for k in globals().keys() if not k.startswith("_")]
