# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import os
from typing import Optional

from detectron2.data import DatasetCatalog, MetadataCatalog

from ..utils import maybe_prepend_base_path
from .dataset_type import DatasetType

CHIMPNSEE_DATASET_NAME = "chimpnsee"


def register_dataset(datasets_root: Optional[os.PathLike] = None):
    def empty_load_callback():
        pass

    video_list_fpath = maybe_prepend_base_path(
        datasets_root, "chimpnsee/cdna.eva.mpg.de/video_list.txt"
    )
    video_base_path = maybe_prepend_base_path(datasets_root, "chimpnsee/cdna.eva.mpg.de")

    DatasetCatalog.register(CHIMPNSEE_DATASET_NAME, empty_load_callback)
    MetadataCatalog.get(CHIMPNSEE_DATASET_NAME).set(
        dataset_type=DatasetType.VIDEO_LIST,
        video_list_fpath=video_list_fpath,
        video_base_path=video_base_path,
    )
