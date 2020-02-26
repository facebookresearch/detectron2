# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import itertools
import logging
import numpy as np
import torch
import torch.nn.functional as F
from fvcore.nn import smooth_l1_loss

from detectron2.layers import batched_nms, cat
from detectron2.structures import Boxes, Instances, pairwise_iou
from detectron2.utils.events import get_event_storage
from detectron2.utils.memory import retry_if_cuda_oom

from ..sampling import subsample_labels

logger = logging.getLogger(__name__)

# TODO: comments for future refactoring of this module
#
# From @rbg:
# This code involves a significant amount of tensor reshaping and permuting. Look for
# ways to simplify this.

"""
Shape shorthand in this module:

    N: number of images in the minibatch
    L: number of feature maps per image on which RPN is run
    A: number of cell anchors (must be the same for all feature maps)
    Hi, Wi: height and width of the i-th feature map
    4: size of the box parameterization

Naming convention:

    objectness: refers to the binary classification of an anchor as object vs. not
    object.

    deltas: refers to the 4-d (dx, dy, dw, dh) deltas that parameterize the box2box
    transform (see :class:`box_regression.Box2BoxTransform`).

    pred_objectness_logits: predicted objectness scores in [-inf, +inf]; use
        sigmoid(pred_objectness_logits) to estimate P(object).

    gt_objectness_logits: ground-truth binary classification labels for objectness

    pred_anchor_deltas: predicted box2box transform deltas

    gt_anchor_deltas: ground-truth box2box transform deltas
"""


def find_top_rpn_proposals(
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
        proposals (list[Tensor]): A list of L tensors. Tensor i has shape (N, Hi*Wi*A, 4).
            All proposal predictions on the feature maps.
        pred_objectness_logits (list[Tensor]): A list of L tensors. Tensor i has shape (N, Hi*Wi*A).
        images (ImageList): Input images as an :class:`ImageList`.
        nms_thresh (float): IoU threshold to use for NMS
        pre_nms_topk (int): number of top k scoring proposals to keep before applying NMS.
            When RPN is run on multiple feature maps (as in FPN) this number is per
            feature map.
        post_nms_topk (int): number of top k scoring proposals to keep after applying NMS.
            When RPN is run on multiple feature maps (as in FPN) this number is total,
            over all feature maps.
        min_box_side_len (float): minimum proposal box side length in pixels (absolute units
            wrt input images).
        training (bool): True if proposals are to be used in training, otherwise False.
            This arg exists only to support a legacy bug; look for the "NB: Legacy bug ..."
            comment.

    Returns:
        proposals (list[Instances]): list of N Instances. The i-th Instances
            stores post_nms_topk object proposals for image i, sorted by their
            objectness score in descending order.
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
        topk_proposals_i = proposals_i[batch_idx[:, None], topk_idx]  # N x topk x 4

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
        boxes = Boxes(topk_proposals[n])
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
            boxes, scores_per_img, lvl = boxes[keep], scores_per_img[keep], level_ids[keep]

        keep = batched_nms(boxes.tensor, scores_per_img, lvl, nms_thresh)
        # In Detectron1, there was different behavior during training vs. testing.
        # (https://github.com/facebookresearch/Detectron/issues/459)
        # During training, topk is over the proposals from *all* images in the training batch.
        # During testing, it is over the proposals for each image separately.
        # As a result, the training behavior becomes batch-dependent,
        # and the configuration "POST_NMS_TOPK_TRAIN" end up relying on the batch size.
        # This bug is addressed in Detectron2 to make the behavior independent of batch size.
        keep = keep[:post_nms_topk]  # keep is already sorted

        res = Instances(image_size)
        res.proposal_boxes = boxes[keep]
        res.objectness_logits = scores_per_img[keep]
        results.append(res)
    return results


def rpn_losses(
    gt_objectness_logits,
    gt_anchor_deltas,
    pred_objectness_logits,
    pred_anchor_deltas,
    smooth_l1_beta,
):
    """
    Args:
        gt_objectness_logits (Tensor): shape (N,), each element in {-1, 0, 1} representing
            ground-truth objectness labels with: -1 = ignore; 0 = not object; 1 = object.
        gt_anchor_deltas (Tensor): shape (N, box_dim), row i represents ground-truth
            box2box transform targets (dx, dy, dw, dh) or (dx, dy, dw, dh, da) that map anchor i to
            its matched ground-truth box.
        pred_objectness_logits (Tensor): shape (N,), each element is a predicted objectness
            logit.
        pred_anchor_deltas (Tensor): shape (N, box_dim), each row is a predicted box2box
            transform (dx, dy, dw, dh) or (dx, dy, dw, dh, da)
        smooth_l1_beta (float): The transition point between L1 and L2 loss in
            the smooth L1 loss function. When set to 0, the loss becomes L1. When
            set to +inf, the loss becomes constant 0.

    Returns:
        objectness_loss, localization_loss, both unnormalized (summed over samples).
    """
    pos_masks = gt_objectness_logits == 1
    localization_loss = smooth_l1_loss(
        pred_anchor_deltas[pos_masks], gt_anchor_deltas[pos_masks], smooth_l1_beta, reduction="sum"
    )

    valid_masks = gt_objectness_logits >= 0
    objectness_loss = F.binary_cross_entropy_with_logits(
        pred_objectness_logits[valid_masks],
        gt_objectness_logits[valid_masks].to(torch.float32),
        reduction="sum",
    )
    return objectness_loss, localization_loss


class RPNOutputs(object):
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
            box2box_transform (Box2BoxTransform): :class:`Box2BoxTransform` instance for
                anchor-proposal transformations.
            anchor_matcher (Matcher): :class:`Matcher` instance for matching anchors to
                ground-truth boxes; used to determine training labels.
            batch_size_per_image (int): number of proposals to sample when training
            positive_fraction (float): target fraction of sampled proposals that should be positive
            images (ImageList): :class:`ImageList` instance representing N input images
            pred_objectness_logits (list[Tensor]): A list of L elements.
                Element i is a tensor of shape (N, A, Hi, Wi) representing
                the predicted objectness logits for anchors.
            pred_anchor_deltas (list[Tensor]): A list of L elements. Element i is a tensor of shape
                (N, A*4, Hi, Wi) representing the predicted "deltas" used to transform anchors
                to proposals.
            anchors (list[list[Boxes]]): A list of N elements. Each element is a list of L
                Boxes. The Boxes at (n, l) stores the entire anchor array for feature map l in image
                n (i.e. the cell anchors repeated over all locations in feature map (n, l)).
            boundary_threshold (int): if >= 0, then anchors that extend beyond the image
                boundary by more than boundary_thresh are not used in training. Set to a very large
                number or < 0 to disable this behavior. Only needed in training.
            gt_boxes (list[Boxes], optional): A list of N elements. Element i a Boxes storing
                the ground-truth ("gt") boxes for image i.
            smooth_l1_beta (float): The transition point between L1 and L2 loss in
                the smooth L1 loss function. When set to 0, the loss becomes L1. When
                set to +inf, the loss becomes constant 0.
        """
        self.box2box_transform = box2box_transform
        self.anchor_matcher = anchor_matcher
        self.batch_size_per_image = batch_size_per_image
        self.positive_fraction = positive_fraction
        self.pred_objectness_logits = pred_objectness_logits
        self.pred_anchor_deltas = pred_anchor_deltas

        self.anchors = anchors
        self.gt_boxes = gt_boxes
        self.num_feature_maps = len(pred_objectness_logits)
        self.num_images = len(images)
        self.image_sizes = images.image_sizes
        self.boundary_threshold = boundary_threshold
        self.smooth_l1_beta = smooth_l1_beta

    def _get_ground_truth(self):
        """
        Returns:
            gt_objectness_logits: list of N tensors. Tensor i is a vector whose length is the
                total number of anchors in image i (i.e., len(anchors[i])). Label values are
                in {-1, 0, 1}, with meanings: -1 = ignore; 0 = negative class; 1 = positive class.
            gt_anchor_deltas: list of N tensors. Tensor i has shape (len(anchors[i]), 4).
        """
        gt_objectness_logits = []
        gt_anchor_deltas = []
        # Concatenate anchors from all feature maps into a single Boxes per image
        anchors = [Boxes.cat(anchors_i) for anchors_i in self.anchors]
        for image_size_i, anchors_i, gt_boxes_i in zip(self.image_sizes, anchors, self.gt_boxes):
            """
            image_size_i: (h, w) for the i-th image
            anchors_i: anchors for i-th image
            gt_boxes_i: ground-truth boxes for i-th image
            """
            match_quality_matrix = retry_if_cuda_oom(pairwise_iou)(gt_boxes_i, anchors_i)
            matched_idxs, gt_objectness_logits_i = retry_if_cuda_oom(self.anchor_matcher)(
                match_quality_matrix
            )
            # Matching is memory-expensive and may result in CPU tensors. But the result is small
            gt_objectness_logits_i = gt_objectness_logits_i.to(device=gt_boxes_i.device)
            del match_quality_matrix

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

    def losses(self):
        """
        Return the losses from a set of RPN predictions and their associated ground-truth.

        Returns:
            dict[loss name -> loss value]: A dict mapping from loss name to loss value.
                Loss names are: `loss_rpn_cls` for objectness classification and
                `loss_rpn_loc` for proposal localization.
        """

        def resample(label):
            """
            Randomly sample a subset of positive and negative examples by overwriting
            the label vector to the ignore value (-1) for all elements that are not
            included in the sample.
            """
            pos_idx, neg_idx = subsample_labels(
                label, self.batch_size_per_image, self.positive_fraction, 0
            )
            # Fill with the ignore label (-1), then set positive and negative labels
            label.fill_(-1)
            label.scatter_(0, pos_idx, 1)
            label.scatter_(0, neg_idx, 0)
            return label

        gt_objectness_logits, gt_anchor_deltas = self._get_ground_truth()
        """
        gt_objectness_logits: list of N tensors. Tensor i is a vector whose length is the
            total number of anchors in image i (i.e., len(anchors[i]))
        gt_anchor_deltas: list of N tensors. Tensor i has shape (len(anchors[i]), B),
            where B is the box dimension
        """
        # Collect all objectness labels and delta targets over feature maps and images
        # The final ordering is L, N, H, W, A from slowest to fastest axis.
        num_anchors_per_map = [np.prod(x.shape[1:]) for x in self.pred_objectness_logits]
        num_anchors_per_image = sum(num_anchors_per_map)

        # Stack to: (N, num_anchors_per_image)
        gt_objectness_logits = torch.stack(
            [resample(label) for label in gt_objectness_logits], dim=0
        )

        # Log the number of positive/negative anchors per-image that's used in training
        num_pos_anchors = (gt_objectness_logits == 1).sum().item()
        num_neg_anchors = (gt_objectness_logits == 0).sum().item()
        storage = get_event_storage()
        storage.put_scalar("rpn/num_pos_anchors", num_pos_anchors / self.num_images)
        storage.put_scalar("rpn/num_neg_anchors", num_neg_anchors / self.num_images)

        assert gt_objectness_logits.shape[1] == num_anchors_per_image
        # Split to tuple of L tensors, each with shape (N, num_anchors_per_map)
        gt_objectness_logits = torch.split(gt_objectness_logits, num_anchors_per_map, dim=1)
        # Concat from all feature maps
        gt_objectness_logits = cat([x.flatten() for x in gt_objectness_logits], dim=0)

        # Stack to: (N, num_anchors_per_image, B)
        gt_anchor_deltas = torch.stack(gt_anchor_deltas, dim=0)
        assert gt_anchor_deltas.shape[1] == num_anchors_per_image
        B = gt_anchor_deltas.shape[2]  # box dimension (4 or 5)

        # Split to tuple of L tensors, each with shape (N, num_anchors_per_image)
        gt_anchor_deltas = torch.split(gt_anchor_deltas, num_anchors_per_map, dim=1)
        # Concat from all feature maps
        gt_anchor_deltas = cat([x.reshape(-1, B) for x in gt_anchor_deltas], dim=0)

        # Collect all objectness logits and delta predictions over feature maps
        # and images to arrive at the same shape as the labels and targets
        # The final ordering is L, N, H, W, A from slowest to fastest axis.
        pred_objectness_logits = cat(
            [
                # Reshape: (N, A, Hi, Wi) -> (N, Hi, Wi, A) -> (N*Hi*Wi*A, )
                x.permute(0, 2, 3, 1).flatten()
                for x in self.pred_objectness_logits
            ],
            dim=0,
        )
        pred_anchor_deltas = cat(
            [
                # Reshape: (N, A*B, Hi, Wi) -> (N, A, B, Hi, Wi) -> (N, Hi, Wi, A, B)
                #          -> (N*Hi*Wi*A, B)
                x.view(x.shape[0], -1, B, x.shape[-2], x.shape[-1])
                .permute(0, 3, 4, 1, 2)
                .reshape(-1, B)
                for x in self.pred_anchor_deltas
            ],
            dim=0,
        )

        objectness_loss, localization_loss = rpn_losses(
            gt_objectness_logits,
            gt_anchor_deltas,
            pred_objectness_logits,
            pred_anchor_deltas,
            self.smooth_l1_beta,
        )
        normalizer = 1.0 / (self.batch_size_per_image * self.num_images)
        loss_cls = objectness_loss * normalizer  # cls: classification loss
        loss_loc = localization_loss * normalizer  # loc: localization loss
        losses = {"loss_rpn_cls": loss_cls, "loss_rpn_loc": loss_loc}

        return losses

    def predict_proposals(self):
        """
        Transform anchors into proposals by applying the predicted anchor deltas.

        Returns:
            proposals (list[Tensor]): A list of L tensors. Tensor i has shape
                (N, Hi*Wi*A, B), where B is box dimension (4 or 5).
        """
        proposals = []
        # Transpose anchors from images-by-feature-maps (N, L) to feature-maps-by-images (L, N)
        anchors = list(zip(*self.anchors))
        # For each feature map
        for anchors_i, pred_anchor_deltas_i in zip(anchors, self.pred_anchor_deltas):
            B = anchors_i[0].tensor.size(1)
            N, _, Hi, Wi = pred_anchor_deltas_i.shape
            # Reshape: (N, A*B, Hi, Wi) -> (N, A, B, Hi, Wi) -> (N, Hi, Wi, A, B) -> (N*Hi*Wi*A, B)
            pred_anchor_deltas_i = (
                pred_anchor_deltas_i.view(N, -1, B, Hi, Wi).permute(0, 3, 4, 1, 2).reshape(-1, B)
            )
            # Concatenate all anchors to shape (N*Hi*Wi*A, B)
            # type(anchors_i[0]) is Boxes (B = 4) or RotatedBoxes (B = 5)
            anchors_i = type(anchors_i[0]).cat(anchors_i)
            proposals_i = self.box2box_transform.apply_deltas(
                pred_anchor_deltas_i, anchors_i.tensor
            )
            # Append feature map proposals with shape (N, Hi*Wi*A, B)
            proposals.append(proposals_i.view(N, -1, B))
        return proposals

    def predict_objectness_logits(self):
        """
        Return objectness logits in the same format as the proposals returned by
        :meth:`predict_proposals`.

        Returns:
            pred_objectness_logits (list[Tensor]): A list of L tensors. Tensor i has shape
                (N, Hi*Wi*A).
        """
        pred_objectness_logits = [
            # Reshape: (N, A, Hi, Wi) -> (N, Hi, Wi, A) -> (N, Hi*Wi*A)
            score.permute(0, 2, 3, 1).reshape(self.num_images, -1)
            for score in self.pred_objectness_logits
        ]
        return pred_objectness_logits
