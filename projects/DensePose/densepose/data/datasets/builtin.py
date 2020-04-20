# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import os

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_coco_json

_URL_PREFIX = "https://dl.fbaipublicfiles.com/densepose/data/"


def get_densepose_metadata():
    meta = {
        "thing_classes": ["person"],
        "densepose_transform_src": _URL_PREFIX + "UV_symmetry_transforms.mat",
        "densepose_smpl_subdiv": _URL_PREFIX + "SMPL_subdiv.mat",
        "densepose_smpl_subdiv_transform": _URL_PREFIX + "SMPL_SUBDIV_TRANSFORM.mat",
    }
    return meta


SPLITS = {
    "densepose_coco_2014_train": ("coco/train2014", "coco/annotations/densepose_train2014.json"),
    "densepose_coco_2014_minival": ("coco/val2014", "coco/annotations/densepose_minival2014.json"),
    "densepose_coco_2014_minival_100": (
        "coco/val2014",
        "coco/annotations/densepose_minival2014_100.json",
    ),
    "densepose_coco_2014_valminusminival": (
        "coco/val2014",
        "coco/annotations/densepose_valminusminival2014.json",
    ),
}

DENSEPOSE_KEYS = ["dp_x", "dp_y", "dp_I", "dp_U", "dp_V", "dp_masks"]

for key, (image_root, json_file) in SPLITS.items():
    # Assume pre-defined datasets live in `./datasets`.
    json_file = os.path.join("datasets", json_file)
    image_root = os.path.join("datasets", image_root)

    DatasetCatalog.register(
        key,
        lambda key=key, json_file=json_file, image_root=image_root: load_coco_json(
            json_file, image_root, key, extra_annotation_keys=DENSEPOSE_KEYS
        ),
    )

    MetadataCatalog.get(key).set(
        json_file=json_file, image_root=image_root, **get_densepose_metadata()
    )
