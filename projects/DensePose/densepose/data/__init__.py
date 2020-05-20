# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from .build import build_detection_test_loader, build_detection_train_loader
from .dataset_mapper import DatasetMapper

# ensure the builtin datasets are registered
from . import datasets

__all__ = [k for k in globals().keys() if not k.startswith("_")]
