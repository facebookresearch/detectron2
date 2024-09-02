# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

# pyre-unsafe

from typing import Any, List
import torch

from detectron2.config import CfgNode
from detectron2.structures import Instances

from .mask import MaskLoss
from .segm import SegmentationLoss


class MaskOrSegmentationLoss:
    """
    Mask or segmentation loss as cross-entropy for raw unnormalized scores
    given ground truth labels. Ground truth labels are either defined by coarse
    segmentation annotation, or by mask annotation, depending on the config
    value MODEL.ROI_DENSEPOSE_HEAD.COARSE_SEGM_TRAINED_BY_MASKS
    """

    def __init__(self, cfg: CfgNode):
        """
        Initialize segmentation loss from configuration options

        Args:
            cfg (CfgNode): configuration options
        """
        self.segm_trained_by_masks = cfg.MODEL.ROI_DENSEPOSE_HEAD.COARSE_SEGM_TRAINED_BY_MASKS
        if self.segm_trained_by_masks:
            self.mask_loss = MaskLoss()
        self.segm_loss = SegmentationLoss(cfg)

    def __call__(
        self,
        proposals_with_gt: List[Instances],
        densepose_predictor_outputs: Any,
        packed_annotations: Any,
    ) -> torch.Tensor:
        """
        Compute segmentation loss as cross-entropy between aligned unnormalized
        score estimates and ground truth; with ground truth given
        either by masks, or by coarse segmentation annotations.

        Args:
            proposals_with_gt (list of Instances): detections with associated ground truth data
            densepose_predictor_outputs: an object of a dataclass that contains predictor outputs
                with estimated values; assumed to have the following attributes:
                * coarse_segm - coarse segmentation estimates, tensor of shape [N, D, S, S]
            packed_annotations: packed annotations for efficient loss computation
        Return:
            tensor: loss value as cross-entropy for raw unnormalized scores
                given ground truth labels
        """
        if self.segm_trained_by_masks:
            return self.mask_loss(proposals_with_gt, densepose_predictor_outputs)
        return self.segm_loss(proposals_with_gt, densepose_predictor_outputs, packed_annotations)

    def fake_value(self, densepose_predictor_outputs: Any) -> torch.Tensor:
        """
        Fake segmentation loss used when no suitable ground truth data
        was found in a batch. The loss has a value 0 and is primarily used to
        construct the computation graph, so that `DistributedDataParallel`
        has similar graphs on all GPUs and can perform reduction properly.

        Args:
            densepose_predictor_outputs: DensePose predictor outputs, an object
                of a dataclass that is assumed to have `coarse_segm`
                attribute
        Return:
            Zero value loss with proper computation graph
        """
        return densepose_predictor_outputs.coarse_segm.sum() * 0
