# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import logging
import os

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets.builtin import _get_builtin_metadata
from detectron2.data.datasets.coco import load_coco_json

logger = logging.getLogger(__name__)


# COCO dataset
def register_coco_instances_with_points(name, metadata, json_file, image_root):
    """
    Register a dataset in COCO's json annotation format for
    instance segmentation with point annotation.

    The point annotation json does not have "segmentation" field, instead,
    it has "point_coords" and "point_labels" fields.

    Args:
        name (str): the name that identifies a dataset, e.g. "coco_2014_train".
        metadata (dict): extra metadata associated with this dataset.  You can
            leave it as an empty dict.
        json_file (str): path to the json instance annotation file.
        image_root (str or path-like): directory which contains all the images.
    """
    assert isinstance(name, str), name
    assert isinstance(json_file, (str, os.PathLike)), json_file
    assert isinstance(image_root, (str, os.PathLike)), image_root
    # 1. register a function which returns dicts
    DatasetCatalog.register(
        name, lambda: load_coco_json(json_file, image_root, name, ["point_coords", "point_labels"])
    )

    # 2. Optionally, add metadata about this dataset,
    # since they might be useful in evaluation, visualization or logging
    MetadataCatalog.get(name).set(
        json_file=json_file, image_root=image_root, evaluator_type="coco", **metadata
    )


_PREDEFINED_SPLITS_COCO = {}
_PREDEFINED_SPLITS_COCO["coco"] = {
    # point annotations without masks
    "coco_2017_train_points_n10_v1_without_masks": (
        "coco/train2017",
        "coco/annotations/instances_train2017_n10_v1_without_masks.json",
    ),
}


def register_all_coco_train_points(root):
    for dataset_name, splits_per_dataset in _PREDEFINED_SPLITS_COCO.items():
        for key, (image_root, json_file) in splits_per_dataset.items():
            # Assume pre-defined datasets live in `./datasets`.
            register_coco_instances_with_points(
                key,
                _get_builtin_metadata(dataset_name),
                os.path.join(root, json_file) if "://" not in json_file else json_file,
                os.path.join(root, image_root),
            )


# True for open source;
# Internally at fb, we register them elsewhere
if __name__.endswith(".register_point_annotations"):
    _root = os.getenv("DETECTRON2_DATASETS", "datasets")
    register_all_coco_train_points(_root)
