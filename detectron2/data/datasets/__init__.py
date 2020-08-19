# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from .cityscapes import load_cityscapes_instances
from .coco import load_coco_json, load_sem_seg
from .lvis import load_lvis_json, register_lvis_instances, get_lvis_instances_meta
from .pascal_voc import load_voc_instances, register_pascal_voc
from .register_coco import register_coco_instances, register_coco_panoptic_separated
from . import builtin  # ensure the builtin datasets are registered


__all__ = [k for k in globals().keys() if "builtin" not in k and not k.startswith("_")]
