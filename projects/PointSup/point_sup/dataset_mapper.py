# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import copy
import logging
import numpy as np
from typing import List, Union
import torch

import detectron2.data.detection_utils as utils
import detectron2.data.transforms as T
from detectron2.config import configurable

from .detection_utils import annotations_to_instances, transform_instance_annotations

__all__ = [
    "PointSupDatasetMapper",
]


class PointSupDatasetMapper:
    """
    The callable currently does the following:
    1. Read the image from "file_name"
    2. Applies transforms to the image and annotations
    3. Prepare data and annotations to Tensor and :class:`Instances`
    """

    @configurable
    def __init__(
        self,
        is_train: bool,
        *,
        augmentations: List[Union[T.Augmentation, T.Transform]],
        image_format: str,
        # Extra data augmentation for point supervision
        sample_points: int = 0,
    ):
        """
        NOTE: this interface is experimental.

        Args:
            is_train: whether it's used in training or inference
            augmentations: a list of augmentations or deterministic transforms to apply
            image_format: an image format supported by :func:`detection_utils.read_image`.
            sample_points: subsample points at each iteration
        """
        # fmt: off
        self.is_train               = is_train
        self.augmentations          = T.AugmentationList(augmentations)
        self.image_format           = image_format
        self.sample_points          = sample_points
        # fmt: on
        logger = logging.getLogger(__name__)
        mode = "training" if is_train else "inference"
        logger.info(f"[DatasetMapper] Augmentations used in {mode}: {augmentations}")
        logger.info(f"Point Augmentations used in {mode}: sample {sample_points} points")

    @classmethod
    def from_config(cls, cfg, is_train: bool = True):
        augs = utils.build_augmentation(cfg, is_train)
        if cfg.INPUT.CROP.ENABLED and is_train:
            raise ValueError("Crop augmentation not supported to point supervision.")

        ret = {
            "is_train": is_train,
            "augmentations": augs,
            "image_format": cfg.INPUT.FORMAT,
            "sample_points": cfg.INPUT.SAMPLE_POINTS,
        }

        return ret

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.
        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        image = utils.read_image(dataset_dict["file_name"], format=self.image_format)
        utils.check_image_size(dataset_dict, image)

        aug_input = T.AugInput(image)
        transforms = self.augmentations(aug_input)
        image = aug_input.image

        image_shape = image.shape[:2]  # h, w
        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))

        if not self.is_train:
            dataset_dict.pop("annotations", None)
            return dataset_dict

        if "annotations" in dataset_dict:
            # Maps points from the closed interval [0, image_size - 1] on discrete
            # image coordinates to the half-open interval [x1, x2) on continuous image
            # coordinates. We use the continuous-discrete conversion from Heckbert
            # 1990 ("What is the coordinate of a pixel?"): d = floor(c) and c = d + 0.5,
            # where d is a discrete coordinate and c is a continuous coordinate.
            for ann in dataset_dict["annotations"]:
                point_coords_wrt_image = np.array(ann["point_coords"]).astype(float)
                point_coords_wrt_image = point_coords_wrt_image + 0.5
                ann["point_coords"] = point_coords_wrt_image

            annos = [
                # also need to transform point coordinates
                transform_instance_annotations(
                    obj,
                    transforms,
                    image_shape,
                )
                for obj in dataset_dict.pop("annotations")
                if obj.get("iscrowd", 0) == 0
            ]
            instances = annotations_to_instances(
                annos,
                image_shape,
                sample_points=self.sample_points,
            )

            dataset_dict["instances"] = utils.filter_empty_instances(instances)
        return dataset_dict
