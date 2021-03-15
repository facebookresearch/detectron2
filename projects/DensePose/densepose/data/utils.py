# Copyright (c) Facebook, Inc. and its affiliates.

import os
from typing import Optional


def is_relative_local_path(path: str):
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
