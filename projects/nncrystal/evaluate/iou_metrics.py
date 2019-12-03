from typing import List
from itertools import product
import numpy as np
from detectron2.utils.visualizer import GenericMask


def compute_iou(a: np.ndarray, b: np.ndarray):
    a.dtype = np.uint8
    b.dtype = np.uint8
    u = np.sum(np.bitwise_or(a, b))
    i = np.sum(np.bitwise_and(a, b))
    return i / u


def bbox_iou(box_1, box_2):
    # determine the (x, y)-coordinates of the intersection rectangle
    x_a = max(box_1[0], box_2[0])
    y_a = max(box_1[1], box_2[1])
    x_b = min(box_1[2], box_2[2])
    y_b = min(box_1[3], box_2[3])

    # compute the area of intersection rectangle
    inter_area = max(0, x_b - x_a + 1) * max(0, y_b - y_a + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    box_1_area = (box_1[2] - box_1[0] + 1) * (box_1[3] - box_1[1] + 1)
    box_2_area = (box_2[2] - box_2[0] + 1) * (box_2[3] - box_2[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    return inter_area / float(box_1_area + box_2_area - inter_area)


def iou_metrics(det_masks: List[np.ndarray], gt_masks: List[np.ndarray], tp_threshold=0.5, det_bboxes=None,
                gt_bboxes=None):
    """
    iou metric in one frame
    :param dets:
    :param gts:
    :param tp_threshold:
    :return:
    """
    FNs = []
    FPs = []
    TPs = []

    # create iou mapping. only above threshold objects are recorded
    iou_map = {}  # [i_det][i_gt]=iou
    iou_rev_map = {}  # [i_gt][i_det]=iou

    # create bboxes if bboxes are not provided
    if gt_bboxes is None:
        gt_bboxes = [GenericMask(m, *m.shape).bbox() for m in gt_masks]
    if det_bboxes is None:
        det_bboxes = [GenericMask(m, *m.shape).bbox() for m in det_masks]

    for i_gt, i_det in product(range(len(gt_masks)), range(len(det_masks))):
        gt_mask = gt_masks[i_gt]
        det_mask = det_masks[i_det]

        gt_bbox = gt_bboxes[i_gt]
        det_bbox = det_bboxes[i_det]

        # use bbox to determine if iou should be calculated
        if bbox_iou(gt_bbox, det_bbox) < tp_threshold:
            continue

        iou_value = compute_iou(gt_mask, det_mask)
        if iou_value >= tp_threshold:
            iou_map.setdefault(i_det, {})[i_gt] = iou_value
            iou_rev_map.setdefault(i_gt, {})[i_det] = iou_value

    for i_gt, _ in enumerate(gt_masks):
        # False Negative: gt no det
        if i_gt not in iou_rev_map:
            FNs.append(i_gt)
        else:
            dets_of_gt: dict = iou_rev_map[i_gt]
            if len(dets_of_gt) == 1:
                # single TP
                TPs.append(
                    (list(dets_of_gt.keys())[0], i_gt)
                )
            else:
                # multiple det on same GT
                dets_sorted = sorted(dets_of_gt.items(), key=lambda x: x[1])
                TPs.append(
                    (dets_sorted[0][0], i_gt)
                )
                for det_lower_iou, _ in enumerate(dets_sorted[1:]):
                    FPs.append(det_lower_iou)

    for i_det, _ in enumerate(det_masks):
        # False Positive: det no gt
        if i_det not in iou_map:
            FPs.append(i_det)

    tp_ious = [iou_map[tp[0]][tp[1]] for tp in TPs]

    # precision:
    precision_ious = [0] * len(FPs) + tp_ious
    precision = np.mean(precision_ious)

    # recall:
    common_recall_ious = [0] * len(FNs) + [1 if x > tp_threshold else 0 for x in tp_ious]
    common_recall = np.mean(common_recall_ious)
    recall_ious = [0] * len(FNs) + tp_ious
    recall = np.mean(recall_ious)

    return {
        "FPs": FPs,
        "TPs": TPs,
        "FNs": FNs,
        "iou_map": iou_map,
        "iou_rev_map": iou_rev_map,
        "precision": precision,
        "recall": recall,
        "common_recall": common_recall,
        "precision_ious": precision_ious,
        "recall_ious": recall_ious,
        "common_recall_ious": common_recall_ious,
    }
