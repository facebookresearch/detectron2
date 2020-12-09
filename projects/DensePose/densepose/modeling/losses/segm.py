# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from typing import Any, List, Optional
import torch
from torch.nn import functional as F

from detectron2.config import CfgNode
from detectron2.structures import Instances

from .utils import SingleTensorsHelper, resample_data


class SegmentationLoss:
    """
    Segmentation loss as cross-entropy for raw unnormalized scores given ground truth
    labels. Segmentation ground truth labels are defined for the bounding box of
    interest at some fixed resolution [S, S], where
        S = MODEL.ROI_DENSEPOSE_HEAD.HEATMAP_SIZE.
    """

    def __init__(self, cfg: CfgNode):
        """
        Initialize segmentation loss from configuration options

        Args:
            cfg (CfgNode): configuration options
        """
        self.heatmap_size = cfg.MODEL.ROI_DENSEPOSE_HEAD.HEATMAP_SIZE
        self.n_segm_chan = cfg.MODEL.ROI_DENSEPOSE_HEAD.NUM_COARSE_SEGM_CHANNELS

    def __call__(
        self,
        proposals_with_gt: List[Instances],
        densepose_predictor_outputs: Any,
        tensors_helper: Optional[SingleTensorsHelper] = None,
    ) -> torch.Tensor:
        """
        Compute segmentation loss as cross-entropy on aligned segmentation
        ground truth and estimated scores.

        Args:
            proposals_with_gt (list of Instances): detections with associated ground truth data
            densepose_predictor_outputs: an object of a dataclass that contains predictor outputs
                with estimated values; assumed to have the following attributes:
                * coarse_segm - coarse segmentation estimates, tensor of shape [N, D, S, S]
            tensors_helper (SingleTensorsHelper or None): if not None, used to obtain
                packed data for efficient loss computation"""
        if tensors_helper is None:
            tensors_helper = SingleTensorsHelper(proposals_with_gt)
        coarse_segm_est = densepose_predictor_outputs.coarse_segm[tensors_helper.index_with_dp]
        with torch.no_grad():
            coarse_segm_gt = resample_data(
                tensors_helper.coarse_segm_gt.unsqueeze(1),
                tensors_helper.bbox_xywh_gt,
                tensors_helper.bbox_xywh_est,
                self.heatmap_size,
                self.heatmap_size,
                mode="nearest",
                padding_mode="zeros",
            ).squeeze(1)
        if self.n_segm_chan == 2:
            coarse_segm_gt = coarse_segm_gt > 0
        return F.cross_entropy(coarse_segm_est, coarse_segm_gt.long())

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
