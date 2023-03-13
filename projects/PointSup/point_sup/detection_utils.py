# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import numpy as np
import torch

# fmt: off
from detectron2.data.detection_utils import \
    annotations_to_instances as base_annotations_to_instances
from detectron2.data.detection_utils import \
    transform_instance_annotations as base_transform_instance_annotations

# fmt: on


def annotations_to_instances(annos, image_size, sample_points=0):
    """
    Create an :class:`Instances` object used by the models,
    from instance annotations in the dataset dict.

    Args:
        annos (list[dict]): a list of instance annotations in one image, each
            element for one instance.
        image_size (tuple): height, width
        sample_points (int): subsample points at each iteration

    Returns:
        Instances:
            It will contain fields "gt_boxes", "gt_classes",
            "gt_point_coords", "gt_point_labels", if they can be obtained from `annos`.
            This is the format that builtin models with point supervision expect.
    """
    target = base_annotations_to_instances(annos, image_size)

    assert ("point_coords" in annos[0]) == ("point_labels" in annos[0])

    if len(annos) and "point_labels" in annos[0]:
        point_coords = []
        point_labels = []
        for i, _ in enumerate(annos):
            # Already in the image coordinate system
            point_coords_wrt_image = np.array(annos[i]["point_coords"])
            point_labels_wrt_image = np.array(annos[i]["point_labels"])

            if sample_points > 0:
                random_indices = np.random.choice(
                    point_coords_wrt_image.shape[0],
                    sample_points,
                    replace=point_coords_wrt_image.shape[0] < sample_points,
                ).astype(int)
                point_coords_wrt_image = point_coords_wrt_image[random_indices]
                point_labels_wrt_image = point_labels_wrt_image[random_indices]
                assert point_coords_wrt_image.shape[0] == point_labels_wrt_image.size

            point_coords.append(point_coords_wrt_image)
            point_labels.append(point_labels_wrt_image)

        point_coords = torch.stack([torch.from_numpy(x) for x in point_coords])
        point_labels = torch.stack([torch.from_numpy(x) for x in point_labels])
        target.gt_point_coords = point_coords
        target.gt_point_labels = point_labels

    return target


def transform_instance_annotations(
    annotation, transforms, image_size, *, keypoint_hflip_indices=None
):
    """
    Apply transforms to box, and point annotations of a single instance.
    It will use `transforms.apply_box` for the box, and
    `transforms.apply_coords` for points.
    Args:
        annotation (dict): dict of instance annotations for a single instance.
            It will be modified in-place.
        transforms (TransformList or list[Transform]):
        image_size (tuple): the height, width of the transformed image
        keypoint_hflip_indices (ndarray[int]): see `create_keypoint_hflip_indices`.
    Returns:
        dict:
            the same input dict with fields "bbox", "point_coords", "point_labels"
            transformed according to `transforms`.
            The "bbox_mode" field will be set to XYXY_ABS.
    """
    annotation = base_transform_instance_annotations(
        annotation, transforms, image_size, keypoint_hflip_indices
    )

    assert ("point_coords" in annotation) == ("point_labels" in annotation)
    if "point_coords" in annotation and "point_labels" in annotation:
        point_coords = annotation["point_coords"]
        point_labels = np.array(annotation["point_labels"]).astype(np.float)
        point_coords = transforms.apply_coords(point_coords)

        # Set all out-of-boundary points to "unlabeled"
        inside = (point_coords >= np.array([0, 0])) & (point_coords <= np.array(image_size[::-1]))
        inside = inside.all(axis=1)
        point_labels[~inside] = -1

        annotation["point_coords"] = point_coords
        annotation["point_labels"] = point_labels

    return annotation
