from typing import List, Dict

import cv2
import scipy
import tensorflow as tf
import numpy as np
from code_loader.helpers.detection.utils import xyxy_to_xywh_format

from effizency.config import CONFIG


def get_bboxes(annotations: Dict) -> np.ndarray:
    """
    This function returns a numpy array of shape (n, 5) where n is the number of bounding boxes in the image.
    The 5 columns are: xc, yc, w, h, label_id
    the coordinates are normalized to the original image size
    """
    image_width, image_height = annotations['imageWidth'], annotations['imageHeight']
    polygons = [ann['points'] for ann in annotations['shapes']]
    labels = np.asarray([CONFIG['label_name_to_id'].get(ann['label']) for ann in annotations['shapes']])
    bboxes = np.asarray([polygon2bbox(polygon) for polygon in polygons])
    bboxes_fixed = xyxy_to_xywh_format(bboxes[:, :4])
    normalized_bboxes = np.stack([
        bboxes_fixed[:, 0] / image_width,  # Normalized (x1)
        bboxes_fixed[:, 1] / image_height,  # Normalized (y1)
        bboxes_fixed[:, 2] / image_width,  # Normalized (w)
        bboxes_fixed[:, 3] / image_height  # Normalized (h)
    ], axis=1)
    bboxes = np.concatenate([normalized_bboxes, labels[:, np.newaxis]], axis=1)
    return bboxes


def polygon2bbox(polygon: List[List[int]]) -> np.ndarray:
    """
    Converts a polygon to a bounding box.

    Args:
        polygon (List[List[int]]): A list of (x, y) coordinates representing a polygon.

    Returns:
        np.ndarray: A numpy array of shape (4,) representing the bounding box in (x1, y1, x2, y2) format.
    """

    # Convert polygon to numpy array
    polygon = np.array(polygon)

    # Calculate bounding box coordinates
    x1 = np.min(polygon[:, 0])
    y1 = np.min(polygon[:, 1])
    x2 = np.max(polygon[:, 0])
    y2 = np.max(polygon[:, 1])

    return np.array([x1, y1, x2, y2])


def pairwise_intersection(boxes1: tf.Tensor, boxes2: tf.Tensor) -> tf.Tensor:
    """
    Given two lists of boxes of size N and M,
    compute the intersection area between __all__ N x M pairs of boxes.
    The box order must be (xmin, ymin, xmax, ymax).

    Args:
        boxes1, boxes2: two tensors of shape [N, 4] and [M, 4] respectively.

    Returns:
        Tensor: intersection, sized [N, M].
    """
    # Expand dimensions to support broadcasting for pairwise computation: [N, 1, 4] and [1, M, 4]
    boxes1 = tf.expand_dims(boxes1, 1)
    boxes2 = tf.expand_dims(boxes2, 0)

    # Compute the intersection in the xy plane: [N, M, 2]
    min_xy = tf.maximum(boxes1[:, :, :2], boxes2[:, :, :2])
    max_xy = tf.minimum(boxes1[:, :, 2:], boxes2[:, :, 2:])
    width_height = tf.maximum(max_xy - min_xy, 0.0)

    # Compute the area of intersection: width * height
    intersection = width_height[:, :, 0] * width_height[:, :, 1]
    return intersection


def pairwise_iou(boxes1: tf.Tensor, boxes2: tf.Tensor) -> tf.Tensor:
    """
    Given two lists of boxes of size N and M, compute the IoU
    (intersection over union) between **all** N x M pairs of boxes.
    The box order must be (xmin, ymin, xmax, ymax).

    Args:
        boxes1, boxes2: two tensors of shape [N, 4] and [M, 4] respectively.

    Returns:
        Tensor: IoU, sized [N, M].
    """
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])  # [N]
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])  # [M]

    inter = pairwise_intersection(boxes1, boxes2)  # [N, M] - Use the pairwise_intersection function

    union = tf.expand_dims(area1, 1) + area2 - inter

    iou = tf.where(
        inter > 0,
        inter / union,
        tf.zeros(1, dtype=inter.dtype)
    )
    return iou


def resize_predicted_masks_to_original_image_shape(predicted_masks: tf.Tensor,
                                                   predicted_bboxes: tf.Tensor) -> np.ndarray:
    """
    This function resizes the predicted masks to the original image shape.

    Args:
        predicted_masks (tf.Tensor): The predicted masks of shape (N, 1, H, W).
        predicted_bboxes (tf.Tensor): The predicted bounding boxes of shape (N, 4).

    Returns:
        tf.Tensor: The projected masks of shape (N, H', W'), where H' and W' are the height and width of the original image.
    """
    # Initialize the final mask with zeros
    n_predictions = predicted_masks.shape[0]
    final_shape = CONFIG['original_image_size']
    final_mask = np.zeros((n_predictions, *final_shape), dtype=np.uint8)

    # Calculate the scale factors for x and y dimensions
    scale_x = final_shape[1] / CONFIG['image_size'][1]
    scale_y = final_shape[0] / CONFIG['image_size'][0]

    for i, (mask, bbox) in enumerate(zip(predicted_masks, predicted_bboxes)):
        # Scale and convert bounding box coordinates from float to int
        x1, y1, x2, y2 = map(int, [bbox[0] * scale_x, bbox[1] * scale_y, bbox[2] * scale_x, bbox[3] * scale_y])

        # Adjust bbox dimensions to ensure it's within the final mask bounds
        x1, y1 = max(x1, 0), max(y1, 0)
        x2, y2 = min(x2, final_shape[1]), min(y2, final_shape[0])

        # Handle cases where the bbox may have zero or negative dimensions after conversion
        if x2 <= x1 or y2 <= y1:
            continue

        # Resize mask to match the bounding box size
        mask_resized = cv2.resize(mask[0].numpy(), (x2 - x1, y2 - y1))
        mask_resized = (mask_resized > CONFIG['mask_threshold']).astype(np.uint8)

        # Assign an integer i to the mask for instance_i
        final_mask[i][y1:y2, x1:x2][mask_resized == 1] = 1

    return final_mask


def is_faulty_mask(mask):
    """
    Check if a mask is faulty (e.g. contains multiple disconnected objects).
        Args:
            mask (np.ndarray): The mask to check.
        Returns:
            bool: True if the mask is faulty, False otherwise.
    """

    # Check for continuity (combined approach)
    labeled_mask, num_objects = scipy.ndimage.label(mask)
    if num_objects > 1:
        return True  # Multiple disconnected objects indicate discontinuity

    return False
