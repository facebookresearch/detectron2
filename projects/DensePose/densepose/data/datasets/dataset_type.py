# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from enum import Enum


class DatasetType(Enum):
    """
    Dataset type, mostly used for datasets that contain data to bootstrap models on
    """

    VIDEO_LIST = "video_list"
