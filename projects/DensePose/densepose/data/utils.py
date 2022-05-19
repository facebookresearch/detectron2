# Copyright (c) Facebook, Inc. and its affiliates.

import os
from typing import Dict, Optional

from detectron2.config import CfgNode


def is_relative_local_path(path: str) -> bool:
    path_str = os.fsdecode(path)
    return ("://" not in path_str) and not os.path.isabs(path)


def maybe_prepend_base_path(base_path: Optional[str], path: str):
    """
    Prepends the provided path with a base path prefix if:
    1) base path is not None;
    2) path is a local path
    """
    if base_path is None:
        return path
    if is_relative_local_path(path):
        return os.path.join(base_path, path)
    return path


def get_class_to_mesh_name_mapping(cfg: CfgNode) -> Dict[int, str]:
    return {
        int(class_id): mesh_name
        for class_id, mesh_name in cfg.DATASETS.CLASS_TO_MESH_NAME_MAPPING.items()
    }


def get_category_to_class_mapping(dataset_cfg: CfgNode) -> Dict[str, int]:
    return {
        category: int(class_id)
        for category, class_id in dataset_cfg.CATEGORY_TO_CLASS_MAPPING.items()
    }
