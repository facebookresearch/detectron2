import torch

from detectron2.structures import Boxes, RotatedBoxes, pairwise_iou, pairwise_iou_rotated


def soft_nms(boxes, scores, method, sigma, threshold):
    """
    Peforms soft non-maximum suppression algorithm on axis aligned boxes

    Args:
        boxes (Tensor[N, 5]):
           boxes where NMS will be performed. They
           are expected to be in (x_ctr, y_ctr, width, height, angle_degrees) format
        scores (Tensor[N]):
           scores for each one of the boxes
        method (str):
           one of ['gaussian', 'linear', 'hard']
        sigma (float):
           param for gaussian
        threshold (float):
           discards all boxes with decayed confidence < threshold

    Returns:
        Tensor:
            int64 tensor with the indices of the elements that have been kept
            by NMS, sorted in decreasing order of scores
    """
    return _soft_nms(Boxes, pairwise_iou, boxes, scores, method, sigma, threshold)


def soft_nms_rotated(boxes, scores, method, sigma, threshold):
    """
    Performs soft non-maximum suppression algorithm on rotated boxes

    Args:
        boxes (Tensor[N, 5]):
           boxes where NMS will be performed. They
           are expected to be in (x_ctr, y_ctr, width, height, angle_degrees) format
        scores (Tensor[N]):
           scores for each one of the boxes
        method (str):
           one of ['gaussian', 'linear', 'hard']
        sigma (float):
           param for gaussian
        threshold (float):
           discards all boxes with decayed confidence < threshold

    Returns:
        Tensor:
            int64 tensor with the indices of the elements that have been kept
            by NMS, sorted in decreasing order of scores
    """
    return _soft_nms(RotatedBoxes, pairwise_iou_rotated, boxes, scores, method, sigma, threshold)


def batched_soft_nms(boxes, scores, idxs, method, sigma, threshold):
    """
    Performs soft non-maximum suppression in a batched fashion.

    Each index value correspond to a category, and NMS
    will not be applied between elements of different categories.

    Args:
        boxes (Tensor[N, 4]):
           boxes where NMS will be performed. They
           are expected to be in (x1, y1, x2, y2) format
        scores (Tensor[N]):
           scores for each one of the boxes
        idxs (Tensor[N]):
           indices of the categories for each one of the boxes.
        method (str):
           one of ['gaussian', 'linear', 'hard']
        sigma (float):
           param for gaussian
        threshold (float):
           discards all boxes with decayed confidence < threshold

    Returns:
        Tensor:
            int64 tensor with the indices of the elements that have been kept
            by NMS, sorted in decreasing order of scores
    """
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.int64, device=boxes.device)
    # strategy: in order to perform NMS independently per class.
    # we add an offset to all the boxes. The offset is dependent
    # only on the class idx, and is large enough so that boxes
    # from different classes do not overlap
    max_coordinate = boxes.max()
    offsets = idxs.to(boxes) * (max_coordinate + 1)
    boxes_for_nms = boxes + offsets[:, None]
    keep = soft_nms(boxes_for_nms, scores, method, sigma, threshold)
    return keep


def batched_soft_nms_rotated(boxes, scores, idxs, method, sigma, threshold):
    """
    Performs soft non-maximum suppression in a batched fashion on rotated bounding boxes.

    Each index value correspond to a category, and NMS
    will not be applied between elements of different categories.

    Args:
        boxes (Tensor[N, 5]):
           boxes where NMS will be performed. They
           are expected to be in (x_ctr, y_ctr, width, height, angle_degrees) format
        scores (Tensor[N]):
           scores for each one of the boxes
        idxs (Tensor[N]):
           indices of the categories for each one of the boxes.
        method (str):
           one of ['gaussian', 'linear', 'hard']
        sigma (float):
           param for gaussian
        threshold (float):
           discards all boxes with decayed confidence < threshold

    Returns:
        Tensor:
            int64 tensor with the indices of the elements that have been kept
            by NMS, sorted in decreasing order of scores
    """
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.int64, device=boxes.device)
    # strategy: in order to perform NMS independently per class.
    # we add an offset to all the boxes. The offset is dependent
    # only on the class idx, and is large enough so that boxes
    # from different classes do not overlap
    max_coordinate = boxes[:, :2].max() + torch.norm(boxes[:, 2:4], 2, dim=1).max()
    offsets = idxs.to(boxes) * (max_coordinate + 1)
    boxes_for_nms = boxes.clone()
    boxes_for_nms[:, :2] += offsets[:, None]
    keep = soft_nms_rotated(boxes_for_nms, scores, method, sigma, threshold)
    return keep


def _soft_nms(box_class, pairwise_iou_func, boxes, scores, method, sigma, threshold):
    """
    Soft non-max suppression algorithm.

    Implementation of [Soft-NMS -- Improving Object Detection With One Line of Codec]
    (https://arxiv.org/abs/1704.04503)

    Each index value correspond to a category, and NMS
    will not be applied between elements of different categories.

    Args:
        box_class (cls): one of Box, RotatedBoxes
        pairwise_iou_func (func): one of pairwise_iou, pairwise_iou_rotated
        boxes (Tensor[N, ?]):
           boxes where NMS will be performed
           if Boxes, in (x1, y1, x2, y2) format
           if RotatedBoxes, in (x_ctr, y_ctr, width, height, angle_degrees) format
        scores (Tensor[N]):
           scores for each one of the boxes
        method (str):
           one of ['gaussian', 'linear', 'hard']
        sigma (float):
           param for gaussian
        threshold (float):
           discards all boxes with decayed confidence < threshold

    Returns:
        Tensor:
            int64 tensor with the indices of the elements that have been kept
            by Soft NMS, sorted in decreasing order of scores
    """

    out = []

    boxes = boxes.clone()
    scores = scores.clone()
    idxs = torch.arange(scores.size()[0])

    while scores.numel() > 0:
        top_idx = torch.argmax(scores)
        top_box = boxes[top_idx]
        ious = pairwise_iou_func(box_class(top_box.unsqueeze(0)), box_class(boxes))[0]

        if method == "linear":
            decay = torch.ones_like(ious)
            decay_mask = ious > threshold
            decay[decay_mask] = 1 - ious[decay_mask]
        elif method == "gaussian":
            decay = torch.exp(-torch.pow(ious, 2) / sigma)
        elif method == "hard":  # standard NMS
            decay = (ious < threshold).float()
        else:
            raise NotImplementedError("{} soft nms method not implemented.".format(method))

        scores *= decay
        keep = scores > threshold
        keep[top_idx] = False

        out.append(idxs[top_idx])

        boxes = boxes[keep]
        scores = scores[keep]
        idxs = idxs[keep]

    return torch.tensor(out)
