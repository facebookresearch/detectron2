# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_coco_json

DENSEPOSE_KEYS = ["dp_x", "dp_y", "dp_I", "dp_U", "dp_V", "dp_masks"]
DENSEPOSE_METADATA_URL_PREFIX = "https://dl.fbaipublicfiles.com/densepose/data/"


@dataclass
class CocoDatasetInfo:
    name: str
    images_root: str
    annotations_fpath: str


DATASETS = [
    CocoDatasetInfo(
        name="densepose_coco_2014_train",
        images_root="coco/train2014",
        annotations_fpath="coco/annotations/densepose_train2014.json",
    ),
    CocoDatasetInfo(
        name="densepose_coco_2014_minival",
        images_root="coco/val2014",
        annotations_fpath="coco/annotations/densepose_minival2014.json",
    ),
    CocoDatasetInfo(
        name="densepose_coco_2014_minival_100",
        images_root="coco/val2014",
        annotations_fpath="coco/annotations/densepose_minival2014_100.json",
    ),
    CocoDatasetInfo(
        name="densepose_coco_2014_valminusminival",
        images_root="coco/val2014",
        annotations_fpath="coco/annotations/densepose_valminusminival2014.json",
    ),
    CocoDatasetInfo(
        name="densepose_chimps",
        images_root="densepose_evolution/densepose_chimps",
        annotations_fpath="densepose_evolution/annotations/densepose_chimps_densepose.json",
    ),
]


def _is_relative_local_path(path: os.PathLike):
    path_str = os.fsdecode(path)
    return ("://" not in path_str) and not os.path.isabs(path)


def _maybe_prepend_base_path(base_path: Optional[os.PathLike], path: os.PathLike):
    """
    Prepends the provided path with a base path prefix if:
    1) base path is not None;
    2) path is a local path
    """
    if base_path is None:
        return path
    if _is_relative_local_path(path):
        return os.path.join(base_path, path)
    return path


def get_metadata(base_path: Optional[os.PathLike]) -> Dict[str, Any]:
    """
    Returns metadata associated with COCO DensePose datasets

    Args:
    base_path: Optional[os.PathLike]
        Base path used to load metadata from

    Returns:
    Dict[str, Any]
        Metadata in the form of a dictionary
    """
    meta = {
        "densepose_transform_src": _maybe_prepend_base_path(
            base_path, "UV_symmetry_transforms.mat"
        ),
        "densepose_smpl_subdiv": _maybe_prepend_base_path(base_path, "SMPL_subdiv.mat"),
        "densepose_smpl_subdiv_transform": _maybe_prepend_base_path(
            base_path, "SMPL_SUBDIV_TRANSFORM.mat"
        ),
    }
    return meta


def register_dataset(dataset_data: CocoDatasetInfo, datasets_root: Optional[os.PathLike] = None):
    """
    Registers provided COCO DensePose dataset

    Args:
    dataset_data: CocoDatasetInfo
        Dataset data
    datasets_root: Optional[os.PathLike]
        Datasets root folder (default: None)
    """
    annotations_fpath = _maybe_prepend_base_path(datasets_root, dataset_data.annotations_fpath)
    images_root = _maybe_prepend_base_path(datasets_root, dataset_data.images_root)

    def load_annotations():
        return load_coco_json(
            json_file=annotations_fpath,
            image_root=images_root,
            dataset_name=dataset_data.name,
            extra_annotation_keys=DENSEPOSE_KEYS,
        )

    DatasetCatalog.register(dataset_data.name, load_annotations)
    MetadataCatalog.get(dataset_data.name).set(
        json_file=annotations_fpath,
        image_root=images_root,
        **get_metadata(DENSEPOSE_METADATA_URL_PREFIX)
    )


def register_datasets(
    datasets_data: Iterable[CocoDatasetInfo], datasets_root: Optional[os.PathLike] = None
):
    """
    Registers provided COCO DensePose datasets

    Args:
    datasets_data: Iterable[CocoDatasetInfo]
        An iterable of dataset datas
    datasets_root: Optional[os.PathLike]
        Datasets root folder (default: None)
    """
    for dataset_data in datasets_data:
        register_dataset(dataset_data, datasets_root)
