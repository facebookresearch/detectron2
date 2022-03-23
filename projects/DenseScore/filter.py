# Copyright (c) Facebook, Inc. and its affiliates.

from typing import List
import torch

from detectron2.config import CfgNode
from detectron2.structures import Instances
from detectron2.structures.boxes import matched_boxlist_iou


class DensePoseDataFilter(object):
    def __init__(self, cfg: CfgNode):
        self.iou_threshold = cfg.MODEL.ROI_DENSEPOSE_HEAD.FG_IOU_THRESHOLD
        self.keep_masks = cfg.MODEL.ROI_DENSEPOSE_HEAD.COARSE_SEGM_TRAINED_BY_MASKS

    @torch.no_grad()  # pyre-ignore[56]
    def __call__(self, features: List[torch.Tensor], proposals_with_targets: List[Instances]):
        """
        Filters proposals with targets to keep only the ones relevant for
        DensePose training

        Args:
            features (list[Tensor]): input data as a list of features,
                each feature is a tensor. Axis 0 represents the number of
                images `N` in the input data; axes 1-3 are channels,
                height, and width, which may vary between features
                (e.g., if a feature pyramid is used).
            proposals_with_targets (list[Instances]): length `N` list of
                `Instances`. The i-th `Instances` contains instances
                (proposals, GT) for the i-th input image,
        Returns:
            list[Tensor]: filtered features
            list[Instances]: filtered proposals
        """
        proposals_filtered = []
        # TODO: the commented out code was supposed to correctly deal with situations
        # where no valid DensePose GT is available for certain images. The corresponding
        # image features were sliced and proposals were filtered. This led to performance
        # deterioration, both in terms of runtime and in terms of evaluation results.
        #
        # feature_mask = torch.ones(
        #    len(proposals_with_targets),
        #    dtype=torch.bool,
        #    device=features[0].device if len(features) > 0 else torch.device("cpu"),
        # )
        for i, proposals_per_image in enumerate(proposals_with_targets):
            if not proposals_per_image.has("gt_densepose") and (
                not proposals_per_image.has("gt_masks") or not self.keep_masks
            ):
                # feature_mask[i] = 0
                continue
            gt_boxes = proposals_per_image.gt_boxes
            est_boxes = proposals_per_image.proposal_boxes
            # apply match threshold for densepose head
            iou = matched_boxlist_iou(gt_boxes, est_boxes)
            iou_select = iou > self.iou_threshold
            proposals_per_image = proposals_per_image[iou_select]  # pyre-ignore[6]

            N_gt_boxes = len(proposals_per_image.gt_boxes)
            assert N_gt_boxes == len(proposals_per_image.proposal_boxes), (
                f"The number of GT boxes {N_gt_boxes} is different from the "
                f"number of proposal boxes {len(proposals_per_image.proposal_boxes)}"
            )
            # filter out any target without suitable annotation
            if self.keep_masks:
                gt_masks = (
                    proposals_per_image.gt_masks
                    if hasattr(proposals_per_image, "gt_masks")
                    else [None] * N_gt_boxes
                )
            else:
                gt_masks = [None] * N_gt_boxes
            gt_densepose = (
                proposals_per_image.gt_densepose
                if hasattr(proposals_per_image, "gt_densepose")
                else [None] * N_gt_boxes
            )
            assert len(gt_masks) == N_gt_boxes
            assert len(gt_densepose) == N_gt_boxes
            selected_indices = [
                i
                for i, (dp_target, mask_target) in enumerate(zip(gt_densepose, gt_masks))
                if (dp_target is not None) or (mask_target is not None)
            ]
            # if not len(selected_indices):
            #     feature_mask[i] = 0
            #     continue
            if len(selected_indices) != N_gt_boxes:
                proposals_per_image = proposals_per_image[selected_indices]  # pyre-ignore[6]
            assert len(proposals_per_image.gt_boxes) == len(proposals_per_image.proposal_boxes)
            proposals_filtered.append(proposals_per_image)
        # features_filtered = [feature[feature_mask] for feature in features]
        # return features_filtered, proposals_filtered
        return features, proposals_filtered
