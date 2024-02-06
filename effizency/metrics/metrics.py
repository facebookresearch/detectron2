import tensorflow as tf
import numpy as np

from effizency.utils.general_utils import resize_predicted_masks_to_original_image_shape


def calculate_iou_matrix(gt_masks: tf.Tensor, pred_masks: tf.Tensor) -> tf.Tensor:
    """
    Compute the Intersection over Union (IoU) matrix between predicted masks and ground truth masks.

    Args:
        gt_masks: Tensor of shape [M, H, W] containing ground truth masks
        pred_masks: Tensor of shape [N, H, W] containing predicted masks
    Returns:
        iou_matrix: Tensor of shape [N, M] containing the IoU scores between each pair of predicted and GT masks
    """
    # Expand dimensions to [N, 1, H, W] for pred_masks and [1, M, H, W] for gt_masks to prepare for broadcasting
    pred_masks_exp = tf.expand_dims(pred_masks, 1)
    gt_masks_exp = tf.expand_dims(gt_masks, 0)

    # Calculate intersections and unions
    intersections = tf.reduce_sum(tf.cast(pred_masks_exp, tf.float32) * tf.cast(gt_masks_exp, tf.float32), axis=[2, 3])
    pred_area = tf.reduce_sum(tf.cast(pred_masks_exp, tf.float32), axis=[2, 3])
    gt_area = tf.reduce_sum(tf.cast(gt_masks_exp, tf.float32), axis=[2, 3])

    unions = pred_area + gt_area - intersections

    # Calculate IoU scores
    iou_scores = tf.math.divide_no_nan(intersections, unions)

    return iou_scores


def calc_mean_mask_iou(gt_masks: tf.Tensor, pred_masks: tf.Tensor, pred_bboxes: tf.Tensor) -> tf.Tensor:
    """
    Compute the Intersection over Union (IoU) for masks.

    Args:
        gt_masks: Tensor of shape [M, H, W] containing ground truth masks
        pred_masks: Tensor of shape [N, 1, H, W] containing predicted masks
        pred_bboxes: Tensor of shape [N, 4] containing predicted bounding boxes
    Returns:
        Tensor of shape [] containing the mean IoU score.
    """

    y_pred = tf.convert_to_tensor(resize_predicted_masks_to_original_image_shape(pred_masks, pred_bboxes))
    gt_masks = tf.squeeze(gt_masks, axis=0)
    # Calculate the IoU matrix
    iou_matrix = calculate_iou_matrix(gt_masks, y_pred)

    # Find the maximum IoU score for each GT mask
    best_iou_scores_per_gt_mask = tf.reduce_max(iou_matrix, axis=0)

    # Calculate mean IoU score based on the best matches
    mean_iou_score = tf.reduce_mean(best_iou_scores_per_gt_mask)

    return mean_iou_score
