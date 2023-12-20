# Copyright (c) Facebook, Inc. and its affiliates.
import copy
import logging
import numpy as np
from typing import Callable, List, Union
import torch
from panopticapi.utils import rgb2id
import numpy as np 
import json
import os
import cv2

from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T

from .target_generator import PanopticDeepLabTargetGenerator

__all__ = ["PanopticDeeplabDatasetMapper"]


class PanopticDeeplabDatasetMapper:
    """
    The callable currently does the following:

    1. Read the image from "file_name" and label from "pan_seg_file_name"
    2. Applies random scale, crop and flip transforms to image and label
    3. Prepare data to Tensor and generate training targets from label
    """

    @configurable
    def __init__(
        self,
        *,
        augmentations: List[Union[T.Augmentation, T.Transform]],
        image_format: str,
        panoptic_target_generator: Callable,
    ):
        """
        NOTE: this interface is experimental.

        Args:
            augmentations: a list of augmentations or deterministic transforms to apply
            image_format: an image format supported by :func:`detection_utils.read_image`.
            panoptic_target_generator: a callable that takes "panoptic_seg" and
                "segments_info" to generate training targets for the model.
        """
        # fmt: off
        # augmentations = [augmentations[0]]
        # print(f"augmentations = {augmentations}")
        self.augmentations          = T.AugmentationList(augmentations)
        self.image_format           = image_format
        # fmt: on
        logger = logging.getLogger(__name__)
        logger.info("Augmentations used in training: " + str(augmentations))

        self.panoptic_target_generator = panoptic_target_generator

    @classmethod
    def from_config(cls, cfg):
        augs = [
            T.ResizeShortestEdge(
                cfg.INPUT.MIN_SIZE_TRAIN,
                cfg.INPUT.MAX_SIZE_TRAIN,
                cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING,
            )
        ]
        if cfg.INPUT.CROP.ENABLED:
            augs.append(T.RandomCrop(cfg.INPUT.CROP.TYPE, cfg.INPUT.CROP.SIZE))
        augs.append(T.RandomFlip())

        # Assume always applies to the training set.
        dataset_names = cfg.DATASETS.TRAIN
        meta = MetadataCatalog.get(dataset_names[0])
        panoptic_target_generator = PanopticDeepLabTargetGenerator(
            ignore_label=meta.ignore_label,
            thing_ids=list(meta.thing_dataset_id_to_contiguous_id.values()),
            sigma=cfg.INPUT.GAUSSIAN_SIGMA,
            ignore_stuff_in_offset=cfg.INPUT.IGNORE_STUFF_IN_OFFSET,
            small_instance_area=cfg.INPUT.SMALL_INSTANCE_AREA,
            small_instance_weight=cfg.INPUT.SMALL_INSTANCE_WEIGHT,
            ignore_crowd_in_semantic=cfg.INPUT.IGNORE_CROWD_IN_SEMANTIC,
        )

        ret = {
            "augmentations": augs,
            "image_format": cfg.INPUT.FORMAT,
            "panoptic_target_generator": panoptic_target_generator,
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
        # print(f"dataset_dict = {dataset_dict}")

        # Load RGB image.
        image = utils.read_image(dataset_dict["file_name"], format=self.image_format)
        utils.check_image_size(dataset_dict, image)
        
        # Load panoptic label image (encoded in RGB image)
        pan_seg_gt = utils.read_image(dataset_dict.pop("pan_seg_file_name"), "RGB")

        # Load camera info
        path_list = dataset_dict['file_name'].split('/')[:-1]
        path_list[6] = 'cityscapes/camera'
        camera_path = '/'.join(path_list)
    
        camera = json.load( open( os.path.join(camera_path, dataset_dict['image_id'] + "_camera.json") ) )
        # print(f"camera = {camera}")
        
        # Load disparity Image
        path_list = dataset_dict['file_name'].split('/')[:-1]
        path_list[6] = 'cityscapes/disparity'
        disparity_path = '/'.join(path_list)
        dis_label = cv2.imread( os.path.join(disparity_path, dataset_dict['image_id'] + "_disparity.png"), cv2.IMREAD_UNCHANGED) # read the 16-bit disparity png file
        dis_label = np.array(dis_label).astype(float)
        
        # convert the png file to real disparity values, according to the official documentation.
        dis_label[dis_label > 0] = (dis_label[dis_label > 0] - 1) / 256 
        # Add small number in disparity to avoid ZeroDivisionError
        dis_label = dis_label + 1e-6
        # Convert disparity map to depth map 
        depth = camera['extrinsic']['baseline'] * camera['intrinsic']['fx'] / dis_label
        # zero mean don't care pixels
        depth[depth == np.inf] = 0
        depth[depth == np.nan] = 0
        # Cap depth at 80m, this is suggested by DORN
        depth = np.minimum(depth, 80)
        # Filter boundary pixels that is too far away
        THRESHOLD = 40
        depth[        :, :90][depth[        :, :90] > THRESHOLD] = 0
        depth[1024-180:, :  ][depth[1024-180:, :  ] > THRESHOLD] = 0
        
        # print(f"depth = {depth.shape}") # (1024, 2048)
        # print(f"image = {image.shape}") # (1024, 2048, 3)
        # print(f"pan_seg_gt = {pan_seg_gt.shape}") # (1024, 2048, 3)

        # print(f"self.augmentations = {self.augmentations}")
        # ResizeShortestEdge(short_edge_length=..., max_size=4096, sample_style='choice')
        # RandomFlip(prob=0.5)

        # Reuses semantic transform for panoptic labels.
        # Data Augmentation
        aug_input = T.AugInput(image, sem_seg=pan_seg_gt)
        _ = self.augmentations(aug_input)
        image, pan_seg_gt = aug_input.image, aug_input.sem_seg

        aug_input = T.AugInput(depth)
        _ = self.augmentations(aug_input)
        depth = aug_input.image

        # print(f"self.augmentations = {self.augmentations}")
        # print(f"depth = {depth.shape}") # (512, 1024)
        # print(f"image = {image.shape}") # (512, 1024, 3)
        # print(f"pan_seg_gt = {pan_seg_gt.shape}") # (512, 1024, 3)

        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        dataset_dict["depth"] = torch.as_tensor(np.ascontiguousarray(depth))

        # Generates training targets for Panoptic-DeepLab.
        targets = self.panoptic_target_generator(rgb2id(pan_seg_gt), dataset_dict["segments_info"])
        dataset_dict.update(targets)
 
        return dataset_dict
