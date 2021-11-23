# -*- coding: utf-8 -*-

try:
    from caffe2.proto import caffe2_pb2 as _tmp

    # caffe2 is optional
except ImportError:
    pass
else:
    from .api import *

from .flatten import TracingAdapter
from .torchscript import scripting_with_instances, dump_torchscript_IR

__all__ = [k for k in globals().keys() if not k.startswith("_")]
