# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import itertools
import logging
import torch

from detectron2.layers import batched_nms_rotated, cat
from detectron2.structures import Instances, RotatedBoxes, pairwise_iou_rotated

from .rpn_outputs import RPNOutputs

logger = logging.getLogger(__name__)

"""
Shape shorthand in this module:

    N: number of images in the minibatch
    L: number of feature maps per image on which RRPN is run
    A: number of cell anchors (must be the same for all feature maps)
    Hi, Wi: height and width of the i-th feature map
    5: size of the box parameterization

Naming convention:

    objectness: refers to the binary classification of an anchor as object vs. not
    object.

    deltas: refers to the 5-d (dx, dy, dw, dh, da) deltas that parameterize the rotated box2box
    transform (see :class:`box_regression.Box2BoxTransformRotated`).

    pred_objectness_logits: predicted objectness scores in [-inf, +inf]; use
        sigmoid(pred_objectness_logits) to estimate P(object).

    gt_objectness_logits: ground-truth binary classification labels for objectness

    pred_anchor_deltas: predicted rotated box2box transform deltas

    gt_anchor_deltas: ground-truth rotated box2box transform deltas
"""


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


class RRPNOutputs(RPNOutputs):
    def __init__(
        self,
        box2box_transform,
        anchor_matcher,
        batch_size_per_image,
        positive_fraction,
        images,
        pred_objectness_logits,
        pred_anchor_deltas,
        anchors,
        boundary_threshold=0,
        gt_boxes=None,
        smooth_l1_beta=0.0,
    ):
        """
        Args:
            box2box_transform (Box2BoxTransformRotated): :class:`Box2BoxTransformRotated`
                instance for anchor-proposal transformations.
            anchor_matcher (Matcher): :class:`Matcher` instance for matching anchors to
                ground-truth boxes; used to determine training labels.
            batch_size_per_image (int): number of proposals to sample when training
            positive_fraction (float): target fraction of sampled proposals that should be positive
            images (ImageList): :class:`ImageList` instance representing N input images
            pred_objectness_logits (list[Tensor]): A list of L elements.
                Element i is a tensor of shape (N, A, Hi, Wi) representing
                the predicted objectness logits for anchors.
            pred_anchor_deltas (list[Tensor]): A list of L elements. Element i is a tensor of shape
                (N, A*5, Hi, Wi) representing the predicted "deltas" used to transform anchors
                to proposals.
            anchors (list[list[RotatedBoxes]]): A list of N elements. Each element is a list of L
                RotatedBoxes. The RotatedBoxes at (n, l) stores the entire anchor array for
                feature map l in image n (i.e. the cell anchors repeated over all locations in
                feature map (n, l)).
            boundary_threshold (int): if >= 0, then anchors that extend beyond the image
                boundary by more than boundary_thresh are not used in training. Set to a very large
                number or < 0 to disable this behavior. Only needed in training.
            gt_boxes (list[RotatedBoxes], optional): A list of N elements. Element i a RotatedBoxes
                storing the ground-truth ("gt") rotated boxes for image i.
            smooth_l1_beta (float): The transition point between L1 and L2 loss in
                the smooth L1 loss function. When set to 0, the loss becomes L1. When
                set to +inf, the loss becomes constant 0.
        """
        super(RRPNOutputs, self).__init__(
            box2box_transform,
            anchor_matcher,
            batch_size_per_image,
            positive_fraction,
            images,
            pred_objectness_logits,
            pred_anchor_deltas,
            anchors,
            boundary_threshold,
            gt_boxes,
            smooth_l1_beta,
        )

    def _get_ground_truth(self):
        """
        Returns:
            gt_objectness_logits: list of N tensors. Tensor i is a vector whose length is the
                total number of anchors in image i (i.e., len(anchors[i])). Label values are
                in {-1, 0, 1}, with meanings: -1 = ignore; 0 = negative class; 1 = positive class.
            gt_anchor_deltas: list of N tensors. Tensor i has shape (len(anchors[i]), 5).
        """
        gt_objectness_logits = []
        gt_anchor_deltas = []
        # Concatenate anchors from all feature maps into a single RotatedBoxes per image
        anchors = [RotatedBoxes.cat(anchors_i) for anchors_i in self.anchors]
        for image_size_i, anchors_i, gt_boxes_i in zip(self.image_sizes, anchors, self.gt_boxes):
            """
            image_size_i: (h, w) for the i-th image
            anchors_i: anchors for i-th image
            gt_boxes_i: ground-truth boxes for i-th image
            """
            match_quality_matrix = pairwise_iou_rotated(gt_boxes_i, anchors_i)
            matched_idxs, gt_objectness_logits_i = self.anchor_matcher(match_quality_matrix)

            if self.boundary_threshold >= 0:
                # Discard anchors that go out of the boundaries of the image
                # NOTE: This is legacy functionality that is turned off by default in Detectron2
                anchors_inside_image = anchors_i.inside_box(image_size_i, self.boundary_threshold)
                gt_objectness_logits_i[~anchors_inside_image] = -1

            if len(gt_boxes_i) == 0:
                # These values won't be used anyway since the anchor is labeled as background
                gt_anchor_deltas_i = torch.zeros_like(anchors_i.tensor)
            else:
                # TODO wasted computation for ignored boxes
                matched_gt_boxes = gt_boxes_i[matched_idxs]
                gt_anchor_deltas_i = self.box2box_transform.get_deltas(
                    anchors_i.tensor, matched_gt_boxes.tensor
                )

            gt_objectness_logits.append(gt_objectness_logits_i)
            gt_anchor_deltas.append(gt_anchor_deltas_i)

        return gt_objectness_logits, gt_anchor_deltas
