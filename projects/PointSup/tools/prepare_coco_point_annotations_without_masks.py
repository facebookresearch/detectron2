#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import copy
import json
import numpy as np
import os
import sys
import pycocotools.mask as mask_utils

from detectron2.utils.env import seed_all_rng
from detectron2.utils.file_io import PathManager


def get_point_annotations(input_filename, output_filename, num_points_per_instance):
    with PathManager.open(input_filename, "r") as f:
        coco_json = json.load(f)

    coco_annos = coco_json.pop("annotations")
    coco_points_json = copy.deepcopy(coco_json)

    imgs = {}
    for img in coco_json["images"]:
        imgs[img["id"]] = img

    new_annos = []
    for ann in coco_annos:
        # convert mask
        t = imgs[ann["image_id"]]
        h, w = t["height"], t["width"]
        segm = ann.pop("segmentation")
        if type(segm) is list:
            # polygon -- a single object might consist of multiple parts
            # we merge all parts into one mask rle code
            rles = mask_utils.frPyObjects(segm, h, w)
            rle = mask_utils.merge(rles)
        elif type(segm["counts"]) is list:
            # uncompressed RLE
            rle = mask_utils.frPyObjects(segm, h, w)
        else:
            # rle
            rle = segm
        mask = mask_utils.decode(rle)
        new_ann = copy.deepcopy(ann)
        # sample points in image coordinates
        box = ann["bbox"]
        point_coords_wrt_image = np.random.rand(num_points_per_instance, 2)
        point_coords_wrt_image[:, 0] = point_coords_wrt_image[:, 0] * box[2]
        point_coords_wrt_image[:, 1] = point_coords_wrt_image[:, 1] * box[3]
        point_coords_wrt_image[:, 0] += box[0]
        point_coords_wrt_image[:, 1] += box[1]
        # round to integer coordinates
        point_coords_wrt_image = np.floor(point_coords_wrt_image).astype(int)
        # get labels
        assert (point_coords_wrt_image >= 0).all(), (point_coords_wrt_image, mask.shape)
        assert (point_coords_wrt_image[:, 0] < w).all(), (point_coords_wrt_image, mask.shape)
        assert (point_coords_wrt_image[:, 1] < h).all(), (point_coords_wrt_image, mask.shape)
        point_labels = mask[point_coords_wrt_image[:, 1], point_coords_wrt_image[:, 0]]
        # store new annotations
        new_ann["point_coords"] = point_coords_wrt_image.tolist()
        new_ann["point_labels"] = point_labels.tolist()
        new_annos.append(new_ann)
    coco_points_json["annotations"] = new_annos

    with PathManager.open(output_filename, "w") as f:
        json.dump(coco_points_json, f)

    print("{} is modified and stored in {}.".format(input_filename, output_filename))


if __name__ == "__main__":
    """
    Generate point-based supervision for COCO dataset.

    Usage:
        python tools/prepare_coco_point_annotations_without_masks.py \
            NUM_POINTS_PER_INSTANCE NUM_VERSIONS_WITH_DIFFERENT_SEED

    Example to generate point-based COCO dataset with 10 points per instance:
        python tools/prepare_coco_point_annotations_without_masks.py 10
    """

    # Fix random seed
    seed_all_rng(12345)

    assert len(sys.argv) >= 2, "Please provide number of points to sample per instance"
    dataset_dir = os.path.join(os.getenv("DETECTRON2_DATASETS", "datasets"), "coco/annotations")
    num_points_per_instance = int(sys.argv[1])
    if len(sys.argv) == 3:
        repeat = int(sys.argv[2])
    else:
        repeat = 1
    s = "instances_train2017"
    for version in range(repeat):
        print(
            "Start sampling {} points per instance for annotations {}.".format(
                num_points_per_instance, s
            )
        )
        get_point_annotations(
            os.path.join(dataset_dir, "{}.json".format(s)),
            os.path.join(
                dataset_dir,
                "{}_n{}_v{}_without_masks.json".format(s, num_points_per_instance, version + 1),
            ),
            num_points_per_instance,
        )
