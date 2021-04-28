# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch

from detectron2.layers import cat


def get_point_coords_from_point_annotation(instances):
    """
    Load point coords and their corresponding labels from point annotation.

    Args:
        instances (list[Instances]): A list of N Instances, where N is the number of images
            in the batch. These instances are in 1:1
            correspondence with the pred_mask_logits. The ground-truth labels (class, box, mask,
            ...) associated with each instance are stored in fields.
    Returns:
        point_coords (Tensor): A tensor of shape (N, P, 2) that contains the coordinates of P
            sampled points.
        point_labels (Tensor): A tensor of shape (N, P) that contains the labels of P
            sampled points. `point_labels` takes 3 possible values:
            - 0: the point belongs to background
            - 1: the point belongs to the object
            - -1: the point is ignored during training
    """
    point_coords_list = []
    point_labels_list = []
    for instances_per_image in instances:
        if len(instances_per_image) == 0:
            continue
        point_coords = instances_per_image.gt_point_coords.to(torch.float32)
        point_labels = instances_per_image.gt_point_labels.to(torch.float32).clone()
        proposal_boxes_per_image = instances_per_image.proposal_boxes.tensor

        # Convert point coordinate system, ground truth points are in image coord.
        point_coords_wrt_box = get_point_coords_wrt_box(proposal_boxes_per_image, point_coords)

        # Ignore points that are outside predicted boxes.
        point_ignores = (
            (point_coords_wrt_box[:, :, 0] < 0)
            | (point_coords_wrt_box[:, :, 0] > 1)
            | (point_coords_wrt_box[:, :, 1] < 0)
            | (point_coords_wrt_box[:, :, 1] > 1)
        )
        point_labels[point_ignores] = -1

        point_coords_list.append(point_coords_wrt_box)
        point_labels_list.append(point_labels)

    return (
        cat(point_coords_list, dim=0),
        cat(point_labels_list, dim=0),
    )


def get_point_coords_wrt_box(boxes_coords, point_coords):
    """
    Convert image-level absolute coordinates to box-normalized [0, 1] x [0, 1] point cooordinates.
    Args:
        boxes_coords (Tensor): A tensor of shape (R, 4) that contains bounding boxes.
            coordinates.
        point_coords (Tensor): A tensor of shape (R, P, 2) that contains
            image-normalized coordinates of P sampled points.
    Returns:
        point_coords_wrt_box (Tensor): A tensor of shape (R, P, 2) that contains
            [0, 1] x [0, 1] box-normalized coordinates of the P sampled points.
    """
    with torch.no_grad():
        point_coords_wrt_box = point_coords.clone()
        point_coords_wrt_box[:, :, 0] -= boxes_coords[:, None, 0]
        point_coords_wrt_box[:, :, 1] -= boxes_coords[:, None, 1]
        point_coords_wrt_box[:, :, 0] = point_coords_wrt_box[:, :, 0] / (
            boxes_coords[:, None, 2] - boxes_coords[:, None, 0]
        )
        point_coords_wrt_box[:, :, 1] = point_coords_wrt_box[:, :, 1] / (
            boxes_coords[:, None, 3] - boxes_coords[:, None, 1]
        )
    return point_coords_wrt_box
