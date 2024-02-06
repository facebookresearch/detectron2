from typing import List

import numpy as np
import cv2

from code_loader.contract.responsedataclasses import BoundingBox
from code_loader.contract.visualizer_classes import LeapImageWithBBox, LeapImageMask
from code_loader.helpers.detection.utils import xyxy_to_xywh_format

from effizency.config import CONFIG
from effizency.utils.visualization_utils import build_final_mask_from_predictions


def bb_gt_visualizer(image: np.ndarray, bboxes: np.ndarray) -> LeapImageWithBBox:
    """
    This function overlays ground truth bounding boxes (BBs) on the input image.

    Parameters:
    image (np.ndarray): The input image for which the ground truth bounding boxes need to be overlaid.
    bboxes (np.ndarray): The ground truth bounding box array for the input image.

    Returns:
    An instance of LeapImageWithBBox containing the input image with ground truth bounding boxes overlaid.
    """
    bb_object = [BoundingBox(x=bb[0], y=bb[1], width=bb[2], height=bb[3],
                             label=CONFIG['id_to_label_name'].get(int(bb[4])),
                             confidence=1.0)
                 for bb in bboxes]
    return LeapImageWithBBox(data=image.astype(np.float32), bounding_boxes=bb_object)


def prediction_bb_visualizer(image: np.ndarray, bboxes: np.ndarray, scores: np.ndarray,
                             class_ids: np.ndarray) -> LeapImageWithBBox:
    """
    This function overlays predicted bounding boxes (BBs) on the input image.

    Parameters:
    image (np.ndarray): The input image for which the predicted bounding boxes need to be overlaid.
    bboxes (np.ndarray): The predicted bounding box array for the input image.

    Returns:
    An instance of LeapImageWithBBox containing the input image with predicted bounding boxes overlaid.
    """

    bboxes_fixed = xyxy_to_xywh_format(bboxes)
    normalized_bboxes = np.concatenate([
        bboxes_fixed[:, :2] / CONFIG['image_size'],  # Normalized (x1, y1)
        bboxes_fixed[:, 2:] / CONFIG['image_size']  # Normalized (w, h)
    ], axis=1)
    bboxes = np.concatenate([normalized_bboxes, class_ids[:, np.newaxis], scores[:, np.newaxis]], axis=1)
    bb_object = [BoundingBox(x=bb[0], y=bb[1], width=bb[2], height=bb[3],
                             label=CONFIG['id_to_label_name'].get(int(bb[4])),
                             confidence=bb[5])
                 for bb in
                 bboxes]
    return LeapImageWithBBox(data=image.astype(np.float32), bounding_boxes=bb_object)


def gt_mask_visualizer(image: np.ndarray, mask: np.ndarray) -> LeapImageMask:
    """
    This function overlays ground truth masks on the input image.

    Parameters:
    image (np.ndarray): The input image for which the ground truth masks need to be overlaid.
    masks (np.ndarray): The ground truth mask array for the input image.

    Returns:
    An instance of LeapImageMask containing the input image with ground truth masks overlaid.
    """
    n_instances = mask.shape[0]
    combined_mask = np.zeros(mask.shape[1:], dtype=np.uint8)
    for i in range(n_instances):
        combined_mask[mask[i] == 1] = i + 1
    return LeapImageMask(image=image.astype(np.float32), mask=combined_mask, labels=[f'roof_{i}' for i in range(n_instances)])


def pred_mask_visualizer(image: np.ndarray, masks: np.ndarray, bboxes: np.ndarray) -> LeapImageMask:
    """
    This function overlays predicted masks on the input image.

    Parameters:
    image (np.ndarray): The input image for which the predicted masks need to be overlaid.
    masks (np.ndarray): The predicted mask array for the input image.
    bboxes (np.ndarray): The predicted bounding box array for the input image.

    Returns:
    An instance of LeapImageMask containing the input image with predicted masks overlaid.
    """
    final_mask = build_final_mask_from_predictions(masks, bboxes, image.shape[:-1])
    return LeapImageMask(image=image.astype(np.float32), mask=final_mask,
                         labels=[f'roof_{i}' for i in range(masks.shape[0])])
