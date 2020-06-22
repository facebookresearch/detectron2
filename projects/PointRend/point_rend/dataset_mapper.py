# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import copy
import logging
import numpy as np
import torch
from fvcore.common.file_io import PathManager
from fvcore.transforms.transform import CropTransform
from PIL import Image

from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T

from .color_augmentation import ColorAugSSDTransform

"""
This file contains the mapping that's applied to "dataset dicts" for semantic segmentation models.
Unlike the default DatasetMapper this mapper uses cropping as the last transformation.
"""

__all__ = ["SemSegDatasetMapper"]


class SemSegDatasetMapper:
    """
    A callable which takes a dataset dict in Detectron2 Dataset format,
    and map it into a format used by semantic segmentation models.

    The callable currently does the following:

    1. Read the image from "file_name"
    2. Applies geometric transforms to the image and annotation
    3. Find and applies suitable cropping to the image and annotation
    4. Prepare image and annotation to Tensors
    """

    def __init__(self, cfg, is_train=True):
        if cfg.INPUT.CROP.ENABLED and is_train:
            self.crop_gen = T.RandomCrop(cfg.INPUT.CROP.TYPE, cfg.INPUT.CROP.SIZE)
            logging.getLogger(__name__).info("CropGen used in training: " + str(self.crop_gen))
        else:
            self.crop_gen = None

        self.tfm_gens = utils.build_transform_gen(cfg, is_train)

        if cfg.INPUT.COLOR_AUG_SSD:
            self.tfm_gens.append(ColorAugSSDTransform(img_format=cfg.INPUT.FORMAT))
            logging.getLogger(__name__).info(
                "Color augmnetation used in training: " + str(self.tfm_gens[-1])
            )

        # fmt: off
        self.img_format               = cfg.INPUT.FORMAT
        self.single_category_max_area = cfg.INPUT.CROP.SINGLE_CATEGORY_MAX_AREA
        self.ignore_value             = cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE
        # fmt: on

        self.is_train = is_train

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        image = utils.read_image(dataset_dict["file_name"], format=self.img_format)
        utils.check_image_size(dataset_dict, image)
        assert "sem_seg_file_name" in dataset_dict

        image, transforms = T.apply_transform_gens(self.tfm_gens, image)
        if self.is_train:
            with PathManager.open(dataset_dict.pop("sem_seg_file_name"), "rb") as f:
                sem_seg_gt = Image.open(f)
                sem_seg_gt = np.asarray(sem_seg_gt, dtype="uint8")
            sem_seg_gt = transforms.apply_segmentation(sem_seg_gt)
            if self.crop_gen:
                image, sem_seg_gt = crop_transform(
                    image,
                    sem_seg_gt,
                    self.crop_gen,
                    self.single_category_max_area,
                    self.ignore_value,
                )
            dataset_dict["sem_seg"] = torch.as_tensor(sem_seg_gt.astype("long"))

        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))

        if not self.is_train:
            dataset_dict.pop("sem_seg_file_name", None)
            return dataset_dict

        return dataset_dict


def crop_transform(image, sem_seg, crop_gen, single_category_max_area, ignore_value):
    """
    Find a cropping window such that no single category occupies more than
        `single_category_max_area` in `sem_seg`. The function retries random cropping 10 times max.
    """
    if single_category_max_area >= 1.0:
        crop_tfm = crop_gen.get_transform(image)
        sem_seg_temp = crop_tfm.apply_segmentation(sem_seg)
    else:
        h, w = sem_seg.shape
        crop_size = crop_gen.get_crop_size((h, w))
        for _ in range(10):
            y0 = np.random.randint(h - crop_size[0] + 1)
            x0 = np.random.randint(w - crop_size[1] + 1)
            sem_seg_temp = sem_seg[y0 : y0 + crop_size[0], x0 : x0 + crop_size[1]]
            labels, cnt = np.unique(sem_seg_temp, return_counts=True)
            cnt = cnt[labels != ignore_value]
            if len(cnt) > 1 and np.max(cnt) / np.sum(cnt) < single_category_max_area:
                break
        crop_tfm = CropTransform(x0, y0, crop_size[1], crop_size[0])
    image = crop_tfm.apply_image(image)
    return image, sem_seg_temp
