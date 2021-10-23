# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.
# File:


from fvcore.common.checkpoint import Checkpointer, PeriodicCheckpointer

from . import catalog as _UNUSED  # register the handler
from .detection_checkpoint import DetectionCheckpointer

__all__ = ["Checkpointer", "PeriodicCheckpointer", "DetectionCheckpointer"]
