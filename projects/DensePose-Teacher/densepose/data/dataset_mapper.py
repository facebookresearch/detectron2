# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.

import copy
import logging
from typing import Any, Dict, List, Tuple
import torch
from random import choice
import numpy as np

from detectron2.data import MetadataCatalog
from detectron2.data import transforms as T
from detectron2.layers import ROIAlign
from detectron2.structures import BoxMode
from detectron2.utils.file_io import PathManager

from densepose.structures import DensePoseDataRelative, DensePoseList, DensePoseTransformData
from densepose.data.transform import RandErase
from densepose.data import detection_utils as utils


def build_augmentation(cfg, is_train):
    logger = logging.getLogger(__name__)
    result = utils.build_augmentation(cfg, is_train)
    if is_train:
        random_rotation = T.RandomRotation(
            cfg.INPUT.ROTATION_ANGLES, expand=False, sample_style="choice"
        )
        result.append(random_rotation)
        logger.info("DensePose-specific augmentation used in training: " + str(random_rotation))
    return result


def build_strong_augmentation(cfg, is_train):
    logger = logging.getLogger(__name__)
    result = []
    if is_train:
        min_size = cfg.INPUT.MIN_SIZE_TRAIN
        max_size = cfg.INPUT.MAX_SIZE_TRAIN

        # random_resize = T.ResizeShortestEdge(min_size, max_size, 'choice')
        # result.append(random_resize)
        # result.append(
        #     T.RandomFlip(
        #         horizontal=cfg.INPUT.RANDOM_FLIP == "horizontal",
        #         vertical=cfg.INPUT.RANDOM_FLIP == "vertical",
        #     )
        # )

        result.append(
            T.RandomRotation(
                cfg.INPUT.ST_ANGLES, expand=False, sample_style="range"
            )
        )

        # result.append(
        #     choice(
        #         [
        #             T.RandomContrast(1., 1.),  # Identity
        #             T.RandomContrast(0.5, 1.5),
        #             T.RandomBrightness(0.5, 1.5),
        #             T.RandomSaturation(0.5, 1.5),
        #         ]
        #     )
        # )
        logger.info("DensePose-specific strong augmentation used in training. ")
    return result


class DatasetMapper:
    """
    A customized version of `detectron2.data.DatasetMapper`
    """

    def __init__(self, cfg, is_train=True):
        self.augmentation = build_augmentation(cfg, is_train)
        self.strong_augmentation = build_strong_augmentation(cfg, is_train)

        self.random_erase = RandErase(
            size=cfg.MODEL.SEMI.ERASE_SIZE, n_iterations=cfg.MODEL.SEMI.ERASE_ITER
        )

        # fmt: off
        self.img_format     = cfg.INPUT.FORMAT
        self.mask_on        = (
            cfg.MODEL.MASK_ON or (
                cfg.MODEL.DENSEPOSE_ON
                and cfg.MODEL.ROI_DENSEPOSE_HEAD.COARSE_SEGM_TRAINED_BY_MASKS)
        )
        self.keypoint_on    = cfg.MODEL.KEYPOINT_ON
        self.densepose_on   = cfg.MODEL.DENSEPOSE_ON
        assert not cfg.MODEL.LOAD_PROPOSALS, "not supported yet"
        # fmt: on
        if self.keypoint_on and is_train:
            # Flip only makes sense in training
            self.keypoint_hflip_indices = utils.create_keypoint_hflip_indices(cfg.DATASETS.TRAIN)
        else:
            self.keypoint_hflip_indices = None

        if self.densepose_on:
            densepose_transform_srcs = [
                MetadataCatalog.get(ds).densepose_transform_src
                for ds in cfg.DATASETS.TRAIN + cfg.DATASETS.TEST
            ]
            assert len(densepose_transform_srcs) > 0
            # TODO: check that DensePose transformation data is the same for
            # all the datasets. Otherwise one would have to pass DB ID with
            # each entry to select proper transformation data. For now, since
            # all DensePose annotated data uses the same data semantics, we
            # omit this check.
            densepose_transform_data_fpath = PathManager.get_local_path(densepose_transform_srcs[0])
            self.densepose_transform_data = DensePoseTransformData.load(
                densepose_transform_data_fpath
            )

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

        image, weak_transforms = T.apply_transform_gens(self.augmentation, image)
        image_shape = image.shape[:2]  # h, w
        dataset_dict["image"] = torch.as_tensor(image.transpose(2, 0, 1).astype("float32"))

        if not self.is_train:
            dataset_dict.pop("annotations", None)
            return dataset_dict

        for anno in dataset_dict["annotations"]:
            if not self.mask_on:
                anno.pop("segmentation", None)
            if not self.keypoint_on:
                anno.pop("keypoints", None)

        # USER: Implement additional transformations if you have other types of data
        # USER: Don't call transpose_densepose if you don't need
        if self.is_train:
            strong_image, strong_transforms = T.apply_transform_gens(self.strong_augmentation, image.copy())
            strong_shape = strong_image.shape[:2]
            # dataset_dict["strong_image"] = torch.as_tensor(strong_image.transpose(2, 0, 1).astype("float32"))

            transforms = weak_transforms + strong_transforms
            annos = [
                self._transform_densepose(
                    utils.transform_train_instance_annotations(
                        obj, weak_transforms, strong_transforms, image_shape, strong_shape,
                        keypoint_hflip_indices=self.keypoint_hflip_indices
                    ),
                    transforms,
                )
                for obj in dataset_dict.pop("annotations")
                if obj.get("iscrowd", 0) == 0
            ]

            if self.mask_on:
                self._add_densepose_masks_as_segmentation(annos, strong_shape)

            do_hflip = sum(isinstance(t, T.HFlipTransform) for t in strong_transforms.transforms) % 2 == 1
            dataset_dict['rotate'] = None
            for t in strong_transforms.transforms:
                if isinstance(t, T.RotationTransform):
                    dataset_dict['rotate'] = t.angle

            instances = utils.annotations_to_instances(annos, strong_shape, do_hflip=do_hflip)
            weak_instances = utils.annotations_to_instances(annos, image_shape, bbox_name='weak_bbox', is_pred=True)

            densepose_annotations = [obj.get("densepose") for obj in annos]
            if densepose_annotations and not all(v is None for v in densepose_annotations):
                instances.gt_densepose = DensePoseList(
                    densepose_annotations, instances.gt_boxes, strong_shape
                )

            dataset_dict["instances"] = instances[instances.gt_boxes.nonempty()]
            dataset_dict['do_hflip'] = do_hflip
            dataset_dict["detected_instances"] = weak_instances[instances.gt_boxes.nonempty()]

            # erase image
            erase_transform = self.random_erase.get_transform(strong_image, dataset_dict['instances'])
            dataset_dict['strong_image'] = torch.as_tensor(
                erase_transform.apply_image(strong_image).transpose(2, 0, 1).astype("float32")
            )

        else:
            annos = [
                self._transform_densepose(
                    utils.transform_test_instance_annotations(
                        obj, weak_transforms, image_shape, keypoint_hflip_indices=self.keypoint_hflip_indices
                    ),
                    weak_transforms,
                )
                for obj in dataset_dict.pop("annotations")
                if obj.get("iscrowd", 0) == 0
            ]

            if self.mask_on:
                self._add_densepose_masks_as_segmentation(annos, image_shape)

            instances = utils.annotations_to_instances(annos, image_shape)
            densepose_annotations = [obj.get("densepose") for obj in annos]
            if densepose_annotations and not all(v is None for v in densepose_annotations):
                instances.gt_densepose = DensePoseList(
                    densepose_annotations, instances.gt_boxes, image_shape
                )

            dataset_dict["instances"] = instances[instances.gt_boxes.nonempty()]
        return dataset_dict

    def _transform_densepose(self, annotation, transforms):
        if not self.densepose_on:
            return annotation

        # Handle densepose annotations
        is_valid, reason_not_valid = DensePoseDataRelative.validate_annotation(annotation)
        if is_valid:
            densepose_data = DensePoseDataRelative(annotation, cleanup=True)
            densepose_data.apply_transform(transforms, self.densepose_transform_data)
            annotation["densepose"] = densepose_data
        else:
            # logger = logging.getLogger(__name__)
            # logger.debug("Could not load DensePose annotation: {}".format(reason_not_valid))
            DensePoseDataRelative.cleanup_annotation(annotation)
            # NOTE: annotations for certain instances may be unavailable.
            # 'None' is accepted by the DensePostList data structure.
            annotation["densepose"] = None
        return annotation

    def _add_densepose_masks_as_segmentation(
        self, annotations: List[Dict[str, Any]], image_shape_hw: Tuple[int, int]
    ):
        for obj in annotations:
            if ("densepose" not in obj) or ("segmentation" in obj):
                continue
            # DP segmentation: torch.Tensor [S, S] of float32, S=256
            segm_dp = torch.zeros_like(obj["densepose"].segm)
            segm_dp[obj["densepose"].segm > 0] = 1
            segm_h, segm_w = segm_dp.shape
            bbox_segm_dp = torch.tensor((0, 0, segm_h - 1, segm_w - 1), dtype=torch.float32)
            # image bbox
            x0, y0, x1, y1 = (
                v.item() for v in BoxMode.convert(obj["bbox"], obj["bbox_mode"], BoxMode.XYXY_ABS)
            )
            segm_aligned = (
                ROIAlign((y1 - y0, x1 - x0), 1.0, 0, aligned=True)
                .forward(segm_dp.view(1, 1, *segm_dp.shape), bbox_segm_dp)
                .squeeze()
            )
            image_mask = torch.zeros(*image_shape_hw, dtype=torch.float32)
            image_mask[y0:y1, x0:x1] = segm_aligned
            # segmentation for BitMask: np.array [H, W] of np.bool
            obj["segmentation"] = image_mask >= 0.5
