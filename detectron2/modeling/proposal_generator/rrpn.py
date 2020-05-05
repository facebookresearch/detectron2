# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import itertools
import logging
from typing import Dict, List
import torch

from detectron2.layers import ShapeSpec, batched_nms_rotated, cat
from detectron2.structures import Instances, RotatedBoxes, pairwise_iou_rotated
from detectron2.utils.memory import retry_if_cuda_oom

from ..box_regression import Box2BoxTransformRotated
from .build import PROPOSAL_GENERATOR_REGISTRY
from .rpn import RPN
from .rpn_outputs import RPNOutputs

logger = logging.getLogger(__name__)


def find_top_rrpn_proposals(
    proposals,
    pred_objectness_logits,
    images,
    nms_thresh,
    pre_nms_topk,
    post_nms_topk,
    min_box_side_len,
    training,
):
    """
    For each feature map, select the `pre_nms_topk` highest scoring proposals,
    apply NMS, clip proposals, and remove small boxes. Return the `post_nms_topk`
    highest scoring proposals among all the feature maps if `training` is True,
    otherwise, returns the highest `post_nms_topk` scoring proposals for each
    feature map.

    Args:
        proposals (list[Tensor]): A list of L tensors. Tensor i has shape (N, Hi*Wi*A, 5).
            All proposal predictions on the feature maps.
        pred_objectness_logits (list[Tensor]): A list of L tensors. Tensor i has shape (N, Hi*Wi*A).
        images (ImageList): Input images as an :class:`ImageList`.
        nms_thresh (float): IoU threshold to use for NMS
        pre_nms_topk (int): number of top k scoring proposals to keep before applying NMS.
            When RRPN is run on multiple feature maps (as in FPN) this number is per
            feature map.
        post_nms_topk (int): number of top k scoring proposals to keep after applying NMS.
            When RRPN is run on multiple feature maps (as in FPN) this number is total,
            over all feature maps.
        min_box_side_len (float): minimum proposal box side length in pixels (absolute units
            wrt input images).
        training (bool): True if proposals are to be used in training, otherwise False.
            This arg exists only to support a legacy bug; look for the "NB: Legacy bug ..."
            comment.

    Returns:
        proposals (list[Instances]): list of N Instances. The i-th Instances
            stores post_nms_topk object proposals for image i.
    """
    image_sizes = images.image_sizes  # in (h, w) order
    num_images = len(image_sizes)
    device = proposals[0].device

    # 1. Select top-k anchor for every level and every image
    topk_scores = []  # #lvl Tensor, each of shape N x topk
    topk_proposals = []
    level_ids = []  # #lvl Tensor, each of shape (topk,)
    batch_idx = torch.arange(num_images, device=device)
    for level_id, proposals_i, logits_i in zip(
        itertools.count(), proposals, pred_objectness_logits
    ):
        Hi_Wi_A = logits_i.shape[1]
        num_proposals_i = min(pre_nms_topk, Hi_Wi_A)

        # sort is faster than topk (https://github.com/pytorch/pytorch/issues/22812)
        # topk_scores_i, topk_idx = logits_i.topk(num_proposals_i, dim=1)
        logits_i, idx = logits_i.sort(descending=True, dim=1)
        topk_scores_i = logits_i[batch_idx, :num_proposals_i]
        topk_idx = idx[batch_idx, :num_proposals_i]

        # each is N x topk
        topk_proposals_i = proposals_i[batch_idx[:, None], topk_idx]  # N x topk x 5

        topk_proposals.append(topk_proposals_i)
        topk_scores.append(topk_scores_i)
        level_ids.append(torch.full((num_proposals_i,), level_id, dtype=torch.int64, device=device))

    # 2. Concat all levels together
    topk_scores = cat(topk_scores, dim=1)
    topk_proposals = cat(topk_proposals, dim=1)
    level_ids = cat(level_ids, dim=0)

    # 3. For each image, run a per-level NMS, and choose topk results.
    results = []
    for n, image_size in enumerate(image_sizes):
        boxes = RotatedBoxes(topk_proposals[n])
        scores_per_img = topk_scores[n]
        valid_mask = torch.isfinite(boxes.tensor).all(dim=1) & torch.isfinite(scores_per_img)
        if not valid_mask.all():
            boxes = boxes[valid_mask]
            scores_per_img = scores_per_img[valid_mask]
        boxes.clip(image_size)

        # filter empty boxes
        keep = boxes.nonempty(threshold=min_box_side_len)
        lvl = level_ids
        if keep.sum().item() != len(boxes):
            boxes, scores_per_img, lvl = (boxes[keep], scores_per_img[keep], level_ids[keep])

        keep = batched_nms_rotated(boxes.tensor, scores_per_img, lvl, nms_thresh)
        # In Detectron1, there was different behavior during training vs. testing.
        # (https://github.com/facebookresearch/Detectron/issues/459)
        # During training, topk is over the proposals from *all* images in the training batch.
        # During testing, it is over the proposals for each image separately.
        # As a result, the training behavior becomes batch-dependent,
        # and the configuration "POST_NMS_TOPK_TRAIN" end up relying on the batch size.
        # This bug is addressed in Detectron2 to make the behavior independent of batch size.
        keep = keep[:post_nms_topk]

        res = Instances(image_size)
        res.proposal_boxes = boxes[keep]
        res.objectness_logits = scores_per_img[keep]
        results.append(res)
    return results


@PROPOSAL_GENERATOR_REGISTRY.register()
class RRPN(RPN):
    """
    Rotated Region Proposal Network described in :paper:`RRPN`.
    """

    def __init__(self, cfg, input_shape: Dict[str, ShapeSpec]):
        super().__init__(cfg, input_shape)
        self.box2box_transform = Box2BoxTransformRotated(weights=cfg.MODEL.RPN.BBOX_REG_WEIGHTS)
        if self.boundary_threshold >= 0:
            raise NotImplementedError(
                "boundary_threshold is a legacy option not implemented for RRPN."
            )

    @torch.no_grad()
    def label_and_sample_anchors(self, anchors: List[RotatedBoxes], gt_instances: List[Instances]):
        """
        Args:
            anchors (list[RotatedBoxes]): anchors for each feature map.
            gt_instances: the ground-truth instances for each image.

        Returns:
            list[Tensor]:
                List of #img tensors. i-th element is a vector of labels whose length is
                the total number of anchors across feature maps. Label values are in {-1, 0, 1},
                with meanings: -1 = ignore; 0 = negative class; 1 = positive class.
            list[Tensor]:
                i-th element is a Nx5 tensor, where N is the total number of anchors across
                feature maps.  The values are the matched gt boxes for each anchor.
                Values are undefined for those anchors not labeled as 1.
        """
        anchors = RotatedBoxes.cat(anchors)

        gt_boxes = [x.gt_boxes for x in gt_instances]
        del gt_instances

        gt_labels = []
        matched_gt_boxes = []
        for gt_boxes_i in gt_boxes:
            """
            gt_boxes_i: ground-truth boxes for i-th image
            """
            match_quality_matrix = retry_if_cuda_oom(pairwise_iou_rotated)(gt_boxes_i, anchors)
            matched_idxs, gt_labels_i = retry_if_cuda_oom(self.anchor_matcher)(match_quality_matrix)
            # Matching is memory-expensive and may result in CPU tensors. But the result is small
            gt_labels_i = gt_labels_i.to(device=gt_boxes_i.device)

            # A vector of labels (-1, 0, 1) for each anchor
            gt_labels_i = self._subsample_labels(gt_labels_i)

            if len(gt_boxes_i) == 0:
                # These values won't be used anyway since the anchor is labeled as background
                matched_gt_boxes_i = torch.zeros_like(anchors.tensor)
            else:
                # TODO wasted indexing computation for ignored boxes
                matched_gt_boxes_i = gt_boxes_i[matched_idxs].tensor

            gt_labels.append(gt_labels_i)  # N,AHW
            matched_gt_boxes.append(matched_gt_boxes_i)
        return gt_labels, matched_gt_boxes

    def forward(self, images, features, gt_instances=None):
        # same signature as RPN.forward
        features = [features[f] for f in self.in_features]
        pred_objectness_logits, pred_anchor_deltas = self.rpn_head(features)
        anchors = self.anchor_generator(features)

        if self.training:
            gt_labels, gt_boxes = self.label_and_sample_anchors(anchors, gt_instances)
        else:
            gt_labels, gt_boxes = None, None

        outputs = RPNOutputs(
            self.box2box_transform,
            self.batch_size_per_image,
            images,
            pred_objectness_logits,
            pred_anchor_deltas,
            anchors,
            gt_labels,
            gt_boxes,
            self.smooth_l1_beta,
        )

        if self.training:
            losses = {k: v * self.loss_weight for k, v in outputs.losses().items()}
        else:
            losses = {}

        with torch.no_grad():
            # Find the top proposals by applying NMS and removing boxes that
            # are too small. The proposals are treated as fixed for approximate
            # joint training with roi heads. This approach ignores the derivative
            # w.r.t. the proposal boxes’ coordinates that are also network
            # responses, so is approximate.

            # Note: this line is the only difference v.s. RPN.forward
            proposals = find_top_rrpn_proposals(
                outputs.predict_proposals(),
                outputs.predict_objectness_logits(),
                images,
                self.nms_thresh,
                self.pre_nms_topk[self.training],
                self.post_nms_topk[self.training],
                self.min_box_side_len,
                self.training,
            )

        return proposals, losses
