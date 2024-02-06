from typing import List, Dict, Union
import tensorflow as tf
import numpy as np
from code_loader.helpers.detection.utils import xywh_to_xyxy_format
from keras.backend import binary_crossentropy
from effizency.config import CONFIG
from effizency.utils.loss_utils import label_and_sample_anchors, get_deltas, label_and_sample_proposals, \
    crop_and_resize, nonzero


def calc_detectron2_loss(gt_boxes: tf.Tensor, anchors: tf.Tensor,
                         pred_objectness_logits: tf.Tensor, pred_anchor_deltas: tf.Tensor,
                         cls_loss_predictions: tf.Tensor, box_loss_predictions: tf.Tensor,
                         proposal_boxes: tf.Tensor, proposal_logits: tf.Tensor, mask_features: tf.Tensor,
                         gt_polygons: tf.Tensor) -> tf.Tensor:
    """
    Keras' implementation of Detectron2 loss function.
        Args:
            gt_boxes: Tensor of shape (N, 5) where N is the number of ground-truth instances.
            anchors: Tensor of shape (R, 4) where N is the number of anchors.
            pred_objectness_logits: Tensor of shape (R, 2) where R is the number of anchors.
            pred_anchor_deltas: Tensor of shape (R, 4) where R is the number of anchors.
            cls_loss_predictions: Tensor of shape (N, 2) where N is the number of ground-truth instances.
            box_loss_predictions: Tensor of shape (N, 4) where N is the number of ground-truth instances.
            proposal_boxes: Tensor of shape (R, 4) where R is the number of anchors.
            proposal_logits: Tensor of shape (R, 2) where R is the number of anchors.
            mask_features: Tensor of shape (R, 1, 28, 28) where R is the number of anchors.
            gt_polygons: Tensor of shape (N, P, 2) where N is the number of ground-truth instances and P is the number
                of points.
        Returns:
            tf.Tensor: The calculated loss.
    """
    boxes_coor = gt_boxes[..., :-1] * CONFIG['image_size'][0]
    gt_boxes = tf.concat([xywh_to_xyxy_format(boxes_coor), gt_boxes[..., -1, tf.newaxis]], axis=-1)
    rpn_losses = calc_rpn_loss(gt_boxes, anchors, pred_objectness_logits, pred_anchor_deltas)
    roi_losses = calc_roi_losses(cls_loss_predictions, box_loss_predictions, proposal_boxes, proposal_logits,
                                 mask_features, gt_boxes, gt_polygons)
    losses = {}
    losses.update(rpn_losses)
    losses.update(roi_losses)
    return sum(losses.values())


def calc_rpn_loss(gt_boxes: tf.Tensor, anchors: tf.Tensor,
                  pred_objectness_logits: tf.Tensor, pred_anchor_deltas: tf.Tensor) -> Dict[str, tf.Tensor]:
    """
    Keras' implementation of RPN loss function.
    Args:
        gt_boxes: Tensor of shape (N, 5) where N is the number of ground-truth instances.
        anchors: Tensor of shape (R, 4) where N is the number of anchors.
        pred_objectness_logits: Tensor of shape (R, 2) where R is the number of anchors.
        pred_anchor_deltas: Tensor of shape (R, 4) where R is the number of anchors.
    Returns:
        dict[str, tf.Tensor]: A dict of loss values.
    """
    boxes = gt_boxes[..., :-1]
    gt_labels, gt_boxes = label_and_sample_anchors(anchors, boxes)
    losses = calc_rpn_losses(anchors, pred_objectness_logits, gt_labels, pred_anchor_deltas, gt_boxes)
    return losses


def calc_roi_losses(cls_loss_predictions: tf.Tensor, box_loss_predictions: tf.Tensor,
                    proposal_boxes: tf.Tensor, proposal_logits: tf.Tensor, mask_features: tf.Tensor,
                    gt_boxes: tf.Tensor, gt_polygons: tf.Tensor) -> Dict[str, tf.Tensor]:
    """
    Keras' implementation of ROI loss function.
    Args:
        cls_loss_predictions: Tensor of shape (N, 2) where N is the number of ground-truth instances.
        box_loss_predictions: Tensor of shape (N, 4) where N is the number of ground-truth instances.
        proposal_boxes: Tensor of shape (R, 4) where R is the number of anchors.
        proposal_logits: Tensor of shape (R, 2) where R is the number of anchors.
        mask_features: Tensor of shape (R, 1, 28, 28) where R is the number of anchors.
        gt_boxes: Tensor of shape (N, 4) where N is the number of ground-truth instances.
        gt_polygons: Tensor of shape (N, P, 2) where N is the number of ground-truth instances and P is the number of
            points.
    Returns:
        dict[str, tf.Tensor]: A dict of loss values.
    """
    gt_classes = gt_boxes[..., -1]
    gt_boxes = gt_boxes[..., :-1]
    batch = cls_loss_predictions.shape[0] // 1000
    cls_loss_predictions = tf.reshape(cls_loss_predictions, (batch, 1000, 2))
    box_loss_predictions = tf.reshape(box_loss_predictions, (batch, 1000, 4))
    proposal_boxes = tf.reshape(proposal_boxes, (batch, 1000, 4))
    proposal_logits = tf.reshape(proposal_logits, (batch, 1000))
    mask_features = tf.reshape(mask_features, (batch, -1, 1, 28, 28))
    proposal_boxes, gt_boxes, gt_classes, scores, proposal_deltas, \
        proposal_logits, gt_polygons = \
        label_and_sample_proposals(
            proposal_boxes=proposal_boxes, proposal_logits=proposal_logits, proposal_deltas=box_loss_predictions,
            masks_features=mask_features,
            gt_boxes=gt_boxes, gt_classes=gt_classes, gt_polygons=gt_polygons,
            scores=cls_loss_predictions
        )
    losses = {}
    box_losses = calc_box_losses(proposal_deltas=box_loss_predictions, scores=scores,
                                 proposals_boxes=proposal_boxes,
                                 gt_boxes=gt_boxes, gt_classes=gt_classes)
    fg_instances = select_foreground_proposals(gt_classes=gt_classes)

    mask_loss = calc_mask_loss(pred_mask_logits=mask_features, gt_classes=gt_classes, gt_polygons=gt_polygons,
                               proposal_boxes=proposal_boxes, fg_indices=fg_instances)
    losses.update(box_losses)
    losses.update({'mask_loss': mask_loss})
    return losses


def calc_box_losses(proposal_deltas: tf.Tensor, scores: tf.Tensor,
                    proposals_boxes: tf.Tensor,
                    gt_classes: tf.Tensor, gt_boxes: tf.Tensor):
    """
    Keras' implementation of box losses.

    Args:
        proposal_deltas: Tensor of shape (R, 4 or 5) or (R, num_classes * (4 or 5)).
        scores: Tensor of shape (R, 1) or (R, num_classes).
        proposals_boxes: Tensor of shape (R, 4 or 5).
        gt_classes: Tensor of shape R, containing ground-truth class labels.
        gt_boxes: Tensor of shape (R, 4 or 5).

    Returns:
        Dict[str, tf.Tensor]: Dictionary of calculated losses.
    """

    # Parse classification outputs
    gt_classes = tf.concat(gt_classes, axis=0)
    scores = tf.concat(scores, axis=0)
    # Parse box regression outputs
    if len(proposals_boxes) > 0:
        proposal_boxes = tf.concat(proposals_boxes, axis=0)  # Nx4
        gt_boxes = tf.concat(gt_boxes, axis=0)
    else:
        proposal_boxes = gt_boxes = tf.zeros((0, 4), dtype=proposal_deltas.dtype)

    if CONFIG['roi_loss']['use_sigmoid_ce']:
        loss_cls = tf.keras.losses.BinaryCrossentropy(from_logits=False)(scores[:, 0], gt_classes)
    else:
        loss_cls = tf.keras.losses.BinaryCrossentropy(from_logits=True)(tf.cast(gt_classes, tf.float32),
                                                                        scores[:, :, 0])

    losses = {
        "loss_cls": loss_cls,
        "loss_box_reg": box_reg_loss(proposal_boxes, gt_boxes, proposal_deltas, gt_classes),
    }
    return {k: v * CONFIG['roi_loss'].get(k, 1.0) for k, v in losses.items()}


def box_reg_loss(proposal_boxes, gt_boxes, pred_deltas, gt_classes):
    """
    Keras' implementation of box_reg_loss.

    Args:
        proposal_boxes: Tensor of shape (R, 4 or 5).
        gt_boxes: Tensor of shape (R, 4 or 5).
        pred_deltas: Tensor of shape (R, 4 or 5) or (R, num_classes * (4 or 5)).
        gt_classes: Tensor of shape R, containing ground-truth class labels.

    Returns:
        tf.Tensor: Calculated box regression loss.
    """

    box_dim = proposal_boxes.shape[-1]  # 4 or 5

    # Select foreground indices (proposals matched to GT boxes)
    fg_inds = tf.where((gt_classes >= 0) & (gt_classes < CONFIG['num_classes']))[:, 1]
    # Handle class-agnostic and class-specific regression
    if pred_deltas.shape[-1] == box_dim:  # Class-agnostic
        fg_pred_deltas = tf.gather(pred_deltas, fg_inds, axis=1)
    else:  # Class-specific
        fg_pred_deltas = tf.gather(
            tf.reshape(pred_deltas, (-1, CONFIG['num_classes'], box_dim)),
            tf.stack([fg_inds, tf.gather(tf.cast(gt_classes, tf.int64), fg_inds)], axis=1),
        )

    # Calculate box regression loss using _dense_box_regression_loss
    loss_box_reg = dense_box_regression_loss(
        tf.gather(proposal_boxes, fg_inds, axis=1)[0, ...],
        fg_pred_deltas,
        tf.gather(gt_boxes, fg_inds, axis=1),
        ...,
        CONFIG['smooth_l1_beta'],
    )

    # Normalize by the total number of regions (R) for equal influence
    return loss_box_reg / tf.maximum(tf.cast(tf.size(gt_classes), tf.float32), 1.0)  # Return 0 if empty


def calc_mask_loss(pred_mask_logits: tf.Tensor, gt_classes: tf.Tensor, gt_polygons: tf.Tensor,
                   proposal_boxes: tf.Tensor, fg_indices: tf.Tensor) -> tf.Tensor:
    """
    Computes the mask prediction loss defined in the Mask R-CNN paper.

    Args:
        pred_mask_logits: A tensor of shape (R, C, M, M) or (R, 1, M, M) for class-specific or class-agnostic, where R
            is the total number of predicted masks in all images, C is the number of foreground classes, and M is the
            resolution of the mask predictions.
        gt_classes: A tensor of shape (R,) where R is the total number of predicted masks in all images.
        gt_polygons: A list of N tensors, where N is the number of images in the batch. The i-th element represents
            the ground-truth polygons for all instances in the i-th image, in the format of (N, P, 2), where N is the
            number of instances and P is the number of points.
        proposal_boxes: A list of N tensors, where N is the number of images in the batch. The i-th element represents
            the proposal boxes for all instances in the i-th image, in the format of (N, 4), where N is the number of
            instances.
        fg_indices: A list of N tensors, where N is the number of images in the batch. The i-th element represents
            the foreground indices of the i-th image.

    Returns:
        mask_loss: A scalar tensor containing the loss.
    """

    cls_agnostic_mask = tf.equal(tf.shape(pred_mask_logits)[2], 1)
    total_num_masks = tf.shape(pred_mask_logits)[1]
    mask_side_len = tf.shape(pred_mask_logits)[3]

    gt_classes_res = []
    gt_masks_res = []
    pred_mask_logits_res = []
    for (
            pred_mask_logits_per_image, proposal_boxes_per_image, gt_classes_per_image,
            gt_poly_per_image, fg_indices_per_image
    ) in zip(
            pred_mask_logits,
            proposal_boxes,
            gt_classes,
            gt_polygons,
            fg_indices):
        if gt_classes_per_image.shape[0] == 0:
            continue

        proposal_boxes_per_image = tf.gather(proposal_boxes_per_image, fg_indices_per_image)
        gt_masks_per_image = tf.gather(gt_poly_per_image, fg_indices_per_image)
        if not cls_agnostic_mask:
            gt_classes_per_image = tf.cast(gt_classes, tf.int64)
            gt_classes_res.append(gt_classes_per_image)

        gt_masks_per_image = crop_and_resize(boxes=proposal_boxes_per_image,
                                             polygons=gt_masks_per_image,
                                             mask_size=mask_side_len)
        gt_masks_res.append(gt_masks_per_image)
        pred_mask_logits_res.append(tf.gather(pred_mask_logits_per_image, fg_indices_per_image))
    if not gt_masks_res:
        return pred_mask_logits.sum() * 0

    gt_masks_res = tf.concat(gt_masks_res, axis=0)
    pred_mask_logits_res = tf.concat(pred_mask_logits_res, axis=0)

    if cls_agnostic_mask:
        pred_mask_logits = pred_mask_logits_res[:, 0]
    else:
        indices = tf.range(total_num_masks)
        gt_classes_res = tf.concat(gt_classes_res, axis=0)
        pred_mask_logits = tf.gather(pred_mask_logits, indices, gt_classes_res)

    gt_masks_res = tf.cast(gt_masks_res, tf.float32)

    mask_loss = tf.keras.losses.binary_crossentropy(gt_masks_res, pred_mask_logits, from_logits=True, axis=0)
    return tf.reduce_mean(mask_loss)


def select_foreground_proposals(gt_classes: List[tf.Tensor]) -> tf.Tensor:
    """
    Given a list of N Instances (for N images), each containing a `gt_classes` field,
    return a list of Instances that contain only instances with `gt_classes != -1 &&
    gt_classes != bg_label`.

    Args:
        gt_classes: A list of N tensors, where N is the number of images in the batch.

    Returns:
        A list of N tensors, where N is the number of images in the batch. The i-th
        element represents the foreground indices of the i-th image.
    """
    fg_indices = []
    for gt_classes_per_image in gt_classes:
        fg_selection_mask = (gt_classes_per_image != -1) & (gt_classes_per_image != CONFIG['num_classes'])
        fg_idxs = tf.squeeze(nonzero(fg_selection_mask), 1)
        fg_indices.append(fg_idxs)
    return tf.convert_to_tensor(fg_indices)


def calc_rpn_losses(anchors: tf.Tensor,
                    pred_objectness_logits: tf.Tensor,
                    gt_labels: tf.Tensor,
                    pred_anchor_deltas: tf.Tensor,
                    gt_boxes: tf.Tensor,
                    ) -> Dict[str, tf.Tensor]:
    """
    Keras' implementation of RPN losses.

    Args:
      anchors: Tensor of shape (N, 4) or (N, A, 4) for class-specific or class-agnostic, where N is the number of images
            in the batch and A is the number of anchors per image.
      pred_objectness_logits: Tensor of shape (N, A, 2), where N is the number of images in the batch and A is the
            number of anchors per image.
      gt_labels: Tensor of shape (N, R) or (N, A, R) for class-specific or class-agnostic, where R is the number of
            ground-truth instances.
      pred_anchor_deltas: Tensor of shape (N, A, 4), where N is the number of images in the batch and A is the number
            of anchors per image.
      gt_boxes: Tensor of shape (N, R, 4) or (N, A, R, 4) for class-specific or class-agnostic, where R is the number
            of ground-truth instances.

    Returns:
        dict[str, tf.Tensor]: A dict of loss values.
    """

    num_images = len(gt_labels)
    gt_labels = tf.stack(gt_labels)  # (N, sum(Hi*Wi*Ai))

    # Create masks for positive and valid anchors
    pos_mask = tf.equal(gt_labels, 1)
    valid_mask = tf.greater_equal(gt_labels, 0)

    localization_loss = dense_box_regression_loss(
        anchors,
        pred_anchor_deltas,
        gt_boxes,
        pos_mask,
        smooth_l1_beta=CONFIG['smooth_l1_beta'],
    )

    # Calculate objectness loss using Keras binary crossentropy
    objectness_loss = tf.reduce_sum(binary_crossentropy(
        tf.concat(pred_objectness_logits, axis=1)[valid_mask],
        tf.cast(gt_labels[valid_mask], tf.float32),
        from_logits=True
    ))

    normalizer = CONFIG['rpn_loss_batch_size_per_image'] * num_images
    losses = {
        "loss_rpn_cls": objectness_loss / normalizer,
        "loss_rpn_loc": localization_loss / normalizer,
    }
    losses = {k: v * CONFIG['rpn_loss_weights'].get(k, 1.0) for k, v in losses.items()}
    return losses


def dense_box_regression_loss(anchors: tf.Tensor,
                              pred_anchor_deltas: tf.Tensor,
                              gt_boxes: tf.Tensor,
                              fg_mask: tf.Tensor,
                              smooth_l1_beta=0.0,
                              ):
    """
    Keras' implementation of dense multi-level box regression loss.

    Args:
        anchors: Tensor of shape (N, 4) or (N, A, 4) for class-specific or class-agnostic, where N is the number of
            images in the batch and A is the number of anchors per image.
        pred_anchor_deltas: Tensor of shape (N, A, 4), where N is the number of images in the batch and A is the number
            of anchors per image.
        gt_boxes: Tensor of shape (N, R, 4) or (N, A, R, 4) for class-specific or class-agnostic, where R is the number
            of ground-truth instances.
        fg_mask: Tensor of shape (N, A), where N is the number of images in the batch and A is the number of anchors per
            image.
        smooth_l1_beta: float, the transition point between L1 and L2 loss. When set to 0, the loss becomes L1. When
            set to +inf, the loss becomes constant 0.

    Returns:
        tf.Tensor: The calculated loss.
    """

    anchors = tf.concat(anchors, axis=0)  # Concatenate anchors

    gt_anchor_deltas = tf.stack([
        get_deltas(anchors, k) for k in gt_boxes
    ])
    loss_box_reg = smooth_l1_loss(
        pred_anchor_deltas[fg_mask],
        gt_anchor_deltas[fg_mask],
        beta=smooth_l1_beta,
        reduction='sum'
    )
    return loss_box_reg


def smooth_l1_loss(input: tf.Tensor, target: tf.Tensor, beta: float, reduction: str = "none") -> tf.Tensor:
    """
    Keras' implementation of smooth L1 loss.

    Args:
        input: (tf.Tensor) input tensor of any shape
        target: (tf.Tensor) target value tensor with the same shape as input
        beta: (float) L1 to L2 change point
        reduction: (str) 'none' | 'mean' | 'sum'

    Returns:
        (tf.Tensor) The loss with the reduction option applied.
    """

    if beta < 1e-5:
        loss = tf.abs(input - target)
    else:
        n = tf.abs(input - target)
        cond = tf.less(n, beta)
        loss = tf.where(cond, 0.5 * tf.square(n) / beta, n - 0.5 * beta)

    if reduction == "mean":
        loss = tf.reduce_mean(loss)
    elif reduction == "sum":
        loss = tf.reduce_sum(loss)
    return loss
