# Copyright (c) Facebook, Inc. and its affiliates.
from detectron2.data import MetadataCatalog
from detectron2.utils.file_io import PathManager

from densepose import DensePoseTransformData


def load_for_dataset(dataset_name):
    path = MetadataCatalog.get(dataset_name).densepose_transform_src
    densepose_transform_data_fpath = PathManager.get_local_path(path)
    return DensePoseTransformData.load(densepose_transform_data_fpath)


def load_from_cfg(cfg):
    return load_for_dataset(cfg.DATASETS.TEST[0])
