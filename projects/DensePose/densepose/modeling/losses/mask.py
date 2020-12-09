# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from dataclasses import dataclass
from typing import Any, Iterable, List, Optional
import torch
from torch.nn import functional as F

from detectron2.structures import Instances


@dataclass
class DataForMaskLoss:
    """
    Contains mask GT and estimated data for proposals from multiple images:
    """

    # tensor of size (K, H, W) containing GT labels
    masks_gt: Optional[torch.Tensor] = None
    # tensor of size (K, C, H, W) containing estimated scores
    masks_est: Optional[torch.Tensor] = None


def extract_data_for_mask_loss_from_matches(
    proposals_targets: Iterable[Instances], estimated_segm: torch.Tensor
) -> DataForMaskLoss:
    """
    Extract data for mask loss from instances that contain matched GT and
    estimated bounding boxes.
    Args:
        proposals_targets: Iterable[Instances]
            matched GT and estimated results, each item in the iterable
            corresponds to data in 1 image
        estimated_segm: tensor(K, C, S, S) of float - raw unnormalized
            segmentation scores, here S is the size to which GT masks are
            to be resized
    Return:
        masks_est: tensor(K, C, S, S) of float - class scores
        masks_gt: tensor(K, S, S) of int64 - labels
    """
    data = DataForMaskLoss()
    masks_gt = []
    offset = 0
    assert estimated_segm.shape[2] == estimated_segm.shape[3], (
        f"Expected estimated segmentation to have a square shape, "
        f"but the actual shape is {estimated_segm.shape[2:]}"
    )
    mask_size = estimated_segm.shape[2]
    num_proposals = sum(inst.proposal_boxes.tensor.size(0) for inst in proposals_targets)
    num_estimated = estimated_segm.shape[0]
    assert (
        num_proposals == num_estimated
    ), "The number of proposals {} must be equal to the number of estimates {}".format(
        num_proposals, num_estimated
    )

    for proposals_targets_per_image in proposals_targets:
        n_i = proposals_targets_per_image.proposal_boxes.tensor.size(0)
        if not n_i:
            continue
        gt_masks_per_image = proposals_targets_per_image.gt_masks.crop_and_resize(
            proposals_targets_per_image.proposal_boxes.tensor, mask_size
        ).to(device=estimated_segm.device)
        masks_gt.append(gt_masks_per_image)
        offset += n_i
    if masks_gt:
        data.masks_est = estimated_segm
        data.masks_gt = torch.cat(masks_gt, dim=0)
    return data


class MaskLoss:
    """
    Mask loss as cross-entropy for raw unnormalized scores given ground truth labels.
    Mask ground truth labels are defined for the whole image and not only the
    bounding box of interest. They are stored as objects that are assumed to implement
    the `crop_and_resize` interface (e.g. BitMasks, PolygonMasks).
    """

    def __call__(
        self, proposals_with_gt: List[Instances], densepose_predictor_outputs: Any
    ) -> torch.Tensor:
        """
        Computes segmentation loss as cross-entropy for raw unnormalized
        scores given ground truth labels.

        Args:
            proposals_with_gt (list of Instances): detections with associated ground truth data
            densepose_predictor_outputs: an object of a dataclass that contains predictor outputs
                with estimated values; assumed to have the following attribute:
                * coarse_segm (tensor of shape [N, D, S, S]): coarse segmentation estimates
                    as raw unnormalized scores
                where N is the number of detections, S is the estimate size ( = width = height)
                and D is the number of coarse segmentation channels.
        Return:
            Cross entropy for raw unnormalized scores for coarse segmentation given
            ground truth labels from masks
        """
        if not len(proposals_with_gt):
            return self.fake_value(densepose_predictor_outputs)
        # densepose outputs are computed for all images and all bounding boxes;
        # i.e. if a batch has 4 images with (3, 1, 2, 1) proposals respectively,
        # the outputs will have size(0) == 3+1+2+1 == 7
        with torch.no_grad():
            mask_loss_data = extract_data_for_mask_loss_from_matches(
                proposals_with_gt, densepose_predictor_outputs.coarse_segm
            )
        if (mask_loss_data.masks_gt is None) or (mask_loss_data.masks_est is None):
            return self.fake_value(densepose_predictor_outputs)
        return F.cross_entropy(mask_loss_data.masks_est, mask_loss_data.masks_gt.long())

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
