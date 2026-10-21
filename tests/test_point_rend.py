# Copyright (c) Facebook, Inc. and its affiliates.
import torch

from detectron2.projects.point_rend.point_features import (
    point_sample_fine_grained_features,
)
from detectron2.structures import Boxes


def test_point_sample_fine_grained_features_empty_boxes():
    features = [torch.zeros(1, 4, 8, 8), torch.zeros(1, 2, 4, 4)]
    boxes = [Boxes(torch.empty((0, 4)))]
    point_coords = torch.empty((0, 5, 2))

    point_features, image_coords = point_sample_fine_grained_features(
        features, [1.0, 0.5], boxes, point_coords
    )

    assert point_features.shape == (0, 6, 5)
    assert image_coords.shape == (0, 5, 2)
