from typing import List, Tuple, Union

import tensorflow as tf
import numpy as np

from pycocotools.mask import frPyObjects, merge, decode

from effizency.config import CONFIG
from effizency.utils.general_utils import pairwise_iou


def label_and_sample_anchors(anchors: tf.Tensor, gt_boxes: List[tf.Tensor]) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Keras' implementation of the label_and_sample_anchors function.

    Args:
        anchors (tf.Tensor): Anchors for each feature map, represented as tensor.
        gt_boxes (tf.Tensor): Ground-truth instances for each image, represented as tensor.

    Returns:
        tuple[list[tf.Tensor], list[tf.Tensor]]:
            - gt_labels: List of tensors containing anchor labels for each image.
            - matched_gt_boxes: List of tensors containing matched ground-truth boxes for each image.
    """

    gt_labels = []
    matched_gt_boxes = []
    for gt_boxes_i in gt_boxes:
        match_quality_matrix = pairwise_iou(gt_boxes_i, anchors)

        matched_indices, gt_labels_i = match_pred_to_gt(match_quality_matrix, type='anchors')

        gt_labels_i = _subsample_labels(gt_labels_i)

        if tf.size(gt_boxes_i) == 0:
            matched_gt_boxes_i = tf.zeros_like(anchors)
        else:
            matched_gt_boxes_i = tf.gather(gt_boxes_i, matched_indices)

        gt_labels.append(gt_labels_i)
        matched_gt_boxes.append(matched_gt_boxes_i)

    return tf.convert_to_tensor(gt_labels), tf.convert_to_tensor(matched_gt_boxes)


def label_and_sample_proposals(proposal_boxes: tf.Tensor, proposal_logits: tf.Tensor,
                               masks_features: tf.Tensor,
                               gt_boxes: tf.Tensor, gt_classes: tf.Tensor, scores: tf.Tensor,
                               proposal_deltas: tf.Tensor, gt_polygons: tf.Tensor):
    """
    Keras's implementation of label_and_sample_proposals.

    Args:
        proposal_boxes (tf.Tensor): Proposal boxes for each image, represented as tensor.
        proposal_logits (tf.Tensor): Proposal logits for each image, represented as tensor.
        masks_features (tf.Tensor): Mask features for each image, represented as tensor.
        gt_boxes (tf.Tensor): Ground-truth instances for each image, represented as tensor.
        gt_classes (tf.Tensor): Ground-truth classes for each image, represented as tensor.
        scores (tf.Tensor): Scores for each image, represented as tensor.
        proposal_deltas (tf.Tensor): Proposal deltas for each image, represented as tensor.
        gt_polygons (tf.Tensor): Ground-truth polygons for each image, represented as tensor.

    Returns:

    """

    box_proposals_with_gt, matched_gt_boxes, matched_gt_classes = [], [], []
    matched_scores, matched_deltas, matched_logits = [], [], []
    matched_masks, matched_masks_features = [], []
    for (
            box_proposals_per_image, logit_proposals_per_image, masks_features_per_image, gt_boxes_per_image,
            gt_classes_per_image, scores_per_image, proposal_deltas_per_image, gt_masks_per_image
    ) in zip(
        proposal_boxes, proposal_logits, masks_features, gt_boxes, gt_classes, scores, proposal_deltas, gt_polygons
    ):
        has_gt = tf.size(gt_classes_per_image) > 0
        match_quality_matrix = pairwise_iou(gt_boxes_per_image, box_proposals_per_image)
        matched_indices, matched_labels = match_pred_to_gt(match_quality_matrix, type='proposals')
        sampled_indices, gt_classes_per_image = _sample_proposals(matched_indices, matched_labels, gt_classes_per_image)

        matched_gt_proposals_i = tf.cond(
            tf.equal(tf.size(gt_boxes_per_image), 0),
            lambda: tf.zeros_like(box_proposals_per_image),  # Handle empty gt_boxes_i
            lambda: tf.gather(box_proposals_per_image, sampled_indices)  # Gather matched ground-truth boxes
        )

        if has_gt:
            sampled_targets = tf.gather(matched_indices, sampled_indices)
            matched_gt_boxes.append(tf.gather(gt_boxes_per_image, sampled_targets))
            matched_gt_classes.append(gt_classes_per_image)
            matched_scores.append(tf.gather(scores_per_image, sampled_targets))
            matched_deltas.append(tf.gather(proposal_deltas_per_image, sampled_targets))
            matched_logits.append(tf.gather(logit_proposals_per_image, sampled_targets))
            matched_masks.append(tf.gather(gt_masks_per_image, sampled_targets))
            matched_masks_features.append(tf.gather(masks_features_per_image, sampled_targets))
        else:
            matched_gt_classes.append(tf.constant(value=CONFIG['num_classes'] + 1,
                                                  shape=gt_classes_per_image.shape,
                                                  dtype=gt_classes_per_image.dtype))

        box_proposals_with_gt.append(matched_gt_proposals_i)
    outputs = [box_proposals_with_gt, matched_gt_boxes, matched_gt_classes, matched_scores,
               matched_deltas, matched_logits, matched_masks]
    tf_outputs = [tf.convert_to_tensor(output) for output in outputs]
    return tuple(tf_outputs)


def _subsample_labels(labels):
    """
    Randomly sample a subset of positive and negative examples, and create a new
    labels tensor where all elements that are not included in the sample are set
    to the ignore value (-1).

    Args:
        labels (Tensor): a vector of -1, 0, 1.

    Returns:
        Tensor: a new tensor with modified labels.
    """
    pos_idx, neg_idx = subsample_labels(
        labels, CONFIG['rpn_loss_batch_size_per_image'], CONFIG['sampling_positive_fraction'], 0
    )
    # Create a tensor filled with -1 (ignore label)
    new_labels = tf.cast(tf.fill(tf.shape(labels), -1), dtype=tf.int64)

    # Update positive and negative labels
    new_labels = tf.tensor_scatter_nd_update(new_labels, tf.expand_dims(pos_idx, 1), tf.ones_like(pos_idx))
    new_labels = tf.tensor_scatter_nd_update(new_labels, tf.expand_dims(neg_idx, 1), tf.zeros_like(neg_idx))

    return new_labels


def _sample_proposals(matched_idxs: tf.Tensor, matched_labels: tf.Tensor, gt_classes: tf.Tensor) -> Tuple[
    tf.Tensor, tf.Tensor]:
    """
    Samples proposals based on matching with groundtruth and sets classification labels.

    Args:
        matched_idxs: A vector of length N, each is the best-matched gt index in [0, M).
        matched_labels: A vector of length N, the matcher's label for each proposal.
        gt_classes: A vector of length M.

    Returns:
        A tuple of:
        - A vector of indices of sampled proposals.
        - A vector of classification labels for each sampled proposal.
    """

    has_gt = tf.greater(tf.size(gt_classes), 0)
    gt_classes = tf.cast(gt_classes, dtype=tf.int32)
    if has_gt:
        gt_classes = tf.gather(gt_classes, matched_idxs)
        gt_classes = tf.where(
            tf.equal(matched_labels, 0),
            tf.fill(tf.shape(matched_labels), CONFIG['num_classes']),
            gt_classes
        )
        gt_classes = tf.where(
            tf.equal(matched_labels, -1),
            -tf.ones_like(matched_labels),
            gt_classes
        )
    else:
        gt_classes = tf.zeros_like(matched_idxs) + CONFIG['num_classes']

    sampled_fg_idxs, sampled_bg_idxs = subsample_labels(
        gt_classes, CONFIG['roi_loss_batch_size_per_image'], CONFIG['sampling_positive_fraction'], CONFIG['num_classes']
    )

    sampled_idxs = tf.concat([sampled_fg_idxs, sampled_bg_idxs], axis=0)
    return sampled_idxs, tf.gather(gt_classes, sampled_idxs)


def subsample_labels(
        labels: tf.Tensor, num_samples: int, positive_fraction: float, bg_label: int
) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Samples positives and negatives from labels, respecting positive_fraction.

    Args:
        labels: (N, ) label vector with values: -1 (ignore), bg_label (background), or others (foreground).
        num_samples: The total number of labels to return.
        positive_fraction: The fraction of positives to sample.
        bg_label: Index of the background class.

    Returns:
        A tuple of (pos_idx, neg_idx), where:
        - pos_idx: Indices of sampled positive labels.
        - neg_idx: Indices of sampled negative labels.
    """

    positive = tf.where(tf.logical_and(tf.not_equal(labels, -1), tf.not_equal(labels, bg_label)))[:, 0]
    negative = tf.where(tf.equal(labels, bg_label))[:, 0]

    num_pos = tf.minimum(tf.cast(num_samples * positive_fraction, tf.int32), tf.size(positive))
    num_neg = num_samples - num_pos
    num_neg = tf.minimum(num_neg, tf.size(negative))

    pos_idx = tf.random.shuffle(positive)[:num_pos]
    neg_idx = tf.random.shuffle(negative)[:num_neg]

    return pos_idx, neg_idx


def match_pred_to_gt(match_quality_matrix, type: str):
    """
    Keras' implementation of the anchor matcher's __call__ function of detectron2.

    Args:
        match_quality_matrix (tf.Tensor): An MxN tensor of pairwise quality scores.
        type (str): a string that specifies the matcher input [anchors, proposals]

    Returns:
        tuple[tf.Tensor, tf.Tensor]:
            - matches: A vector of length N, containing matched ground-truth indices.
            - match_labels: A vector of length N, indicating true/false positives or ignored predictions.
    """

    assert len(match_quality_matrix.shape) == 2
    if type == 'anchors':
        matcher_config = CONFIG['anchor_matcher']
    elif type == 'proposals':
        matcher_config = CONFIG['proposal_matcher']
    # Handle empty match quality matrix
    matches = tf.cond(
        tf.equal(tf.size(match_quality_matrix), 0),
        lambda: tf.zeros_like(match_quality_matrix[0], dtype=tf.int64),
        lambda: tf.argmax(match_quality_matrix, axis=0),  # Find best match along first axis
    )
    matched_values = tf.reduce_max(match_quality_matrix, axis=0)
    match_labels = tf.fill(matches.shape, matcher_config['labels'][0],
                           name='match_labels')  # Initialize with default label

    # Assign labels based on thresholds
    for l, low, high in zip(matcher_config['labels'], matcher_config['thresholds'][:-1],
                            matcher_config['thresholds'][1:]):
        low_high = tf.logical_and(
            tf.greater_equal(matched_values, low),
            tf.less(matched_values, high)
        )
        match_labels = tf.where(low_high, l, match_labels)

    # Handle low-quality matches
    if matcher_config['allow_low_quality_matches']:
        match_labels = set_low_quality_matches(match_labels, match_quality_matrix)

    return matches, match_labels


def set_low_quality_matches(match_labels, match_quality_matrix):
    """
    Keras' implementation of set_low_quality_matches_.

    Args:
        match_labels (tf.Tensor): A vector of length N, representing match labels.
        match_quality_matrix (tf.Tensor): An MxN tensor of pairwise quality scores.
    """

    # Find the highest quality match for each ground-truth
    highest_quality_foreach_gt = tf.reduce_max(match_quality_matrix, axis=1)

    # Find predictions with the highest quality matches (including ties)
    _, pred_inds_with_highest_quality = nonzero(tf.equal(match_quality_matrix,
                                                         tf.expand_dims(highest_quality_foreach_gt, axis=1)),
                                                as_tuple=True)

    # Assign positive labels to those predictions
    match_labels = tf.tensor_scatter_nd_update(
        match_labels, tf.expand_dims(pred_inds_with_highest_quality, axis=1),
        tf.ones_like(pred_inds_with_highest_quality)
    )
    return match_labels


def nonzero(tensor, as_tuple: bool = False):
    indices = tf.cast(tf.where(tf.math.not_equal(tensor, False)), dtype=tf.int32)
    if as_tuple:
        return indices[:, 0], indices[:, 1]
    else:
        return indices


def get_deltas(src_boxes, target_boxes):
    """
    Keras' implementation of get_deltas for box regression.

    Args:
        src_boxes: (tf.Tensor) source boxes, e.g., object proposals
        target_boxes: (tf.Tensor) target of the transformation, e.g., ground-truth boxes

    Returns:
        tf.Tensor: box regression transformation deltas (dx, dy, dw, dh)
    """

    src_widths = src_boxes[:, 2] - src_boxes[:, 0]
    src_heights = src_boxes[:, 3] - src_boxes[:, 1]
    src_ctr_x = src_boxes[:, 0] + 0.5 * src_widths
    src_ctr_y = src_boxes[:, 1] + 0.5 * src_heights

    target_widths = target_boxes[:, 2] - target_boxes[:, 0]
    target_heights = target_boxes[:, 3] - target_boxes[:, 1]
    target_ctr_x = target_boxes[:, 0] + 0.5 * target_widths
    target_ctr_y = target_boxes[:, 1] + 0.5 * target_heights

    wx, wy, ww, wh = CONFIG['box_transform_weights']  # Assuming weights are available in self
    dx = wx * (target_ctr_x - src_ctr_x) / tf.where(tf.equal(src_widths, 0), 1, src_widths)  # Prevent division by zero
    dy = wy * (target_ctr_y - src_ctr_y) / tf.where(tf.equal(src_heights, 0), 1, src_heights)
    dw = ww * tf.math.log(target_widths / tf.where(tf.equal(src_widths, 0), 1, src_widths))
    dh = wh * tf.math.log(target_heights / tf.where(tf.equal(src_heights, 0), 1, src_heights))

    deltas = tf.stack([dx, dy, dw, dh], axis=1)
    return deltas


def crop_and_resize(boxes: tf.Tensor, polygons: tf.Tensor, mask_size: int) -> tf.Tensor:
    """
    Crops each mask by the given box and resizes results to (mask_size, mask_size).

    Args:
        boxes: Nx4 tensor storing the boxes for each mask.
        mask_size: The size of the rasterized mask.

    Returns:
        A bool tensor of shape (N, mask_size, mask_size).
    """
    device = boxes.device
    boxes = boxes.cpu()  # Move boxes to CPU

    results = [
        rasterize_polygons_within_box(poly, box.numpy(), mask_size)
        for poly, box in zip(polygons, boxes)
    ]

    if not results:
        return tf.zeros((0, mask_size, mask_size), dtype=tf.bool, device=device)
    return tf.stack(results, axis=0)


def rasterize_polygons_within_box(polygons: List[tf.Tensor], box: tf.Tensor, mask_size: int) -> tf.Tensor:
    """
    Rasterizes polygons into a mask, crops within the box, and resizes to (mask_size, mask_size).

    Args:
        polygons: A list of polygons, each represented as a 2D tensor of shape (N, 2).
        box: A 4-element tensor representing the bounding box.
        mask_size: The desired size of the output mask.

    Returns:
        A boolean tensor of shape (mask_size, mask_size).
    """

    w = box[2] - box[0]
    h = box[3] - box[1]
    mask_size = tf.cast(mask_size, tf.float32)
    polygons = [tf.identity(polygons[polygons >= 0])]  # Create independent copies
    for p in polygons:
        even_indices = tf.expand_dims(tf.range(0, tf.shape(p)[0], 2), axis=1)
        even_values = tf.squeeze(tf.gather(p, even_indices))
        p = tf.tensor_scatter_nd_update(p, even_indices, even_values - box[0])

        odd_indices = tf.expand_dims(tf.range(1, tf.shape(p)[0], 2), axis=1)
        odd_values = tf.squeeze(tf.gather(p, odd_indices))
        p = tf.tensor_scatter_nd_update(p, odd_indices, odd_values - box[1])

    ratio_h = mask_size / tf.maximum(h, 0.1)
    ratio_w = mask_size / tf.maximum(w, 0.1)

    if tf.equal(ratio_h, ratio_w):
        for p in polygons:
            p *= ratio_h
    else:
        for p in polygons:
            even_indices = tf.expand_dims(tf.range(0, tf.shape(p)[0], 2), axis=1)
            even_values = tf.squeeze(tf.gather(p, even_indices))
            p = tf.tensor_scatter_nd_update(p, even_indices, tf.cast(even_values, dtype=tf.float32) * ratio_w)

            odd_indices = tf.expand_dims(tf.range(1, tf.shape(p)[0], 2), axis=1)
            odd_values = tf.squeeze(tf.gather(p, odd_indices))
            p = tf.tensor_scatter_nd_update(p, odd_indices, odd_values * ratio_h)

    # Placeholder for TensorFlow polygon rasterization
    mask = polygons_to_bitmask(polygons, mask_size, mask_size)
    return tf.cast(mask, tf.bool)


def polygons_to_bitmask(polygons: List[np.ndarray], height: int, width: int) -> np.ndarray:
    """
    Args:
        polygons (list[ndarray]): each array has shape (Nx2,)
        height, width (int)

    Returns:
        ndarray: a bool mask of shape (height, width)
    """
    if len(polygons) == 0:
        # COCOAPI does not support empty polygons
        return np.zeros((height, width)).astype(bool)
    rles = frPyObjects(polygons, height, width)
    rle = merge(rles)
    return decode(rle).astype(bool)
