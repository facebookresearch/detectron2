# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import numpy as np
from typing import Dict
import torch
from torch import nn
from torch.nn import functional as F

from detectron2.layers import ShapeSpec, cat
from detectron2.modeling import SEM_SEG_HEADS_REGISTRY

from .point_features import (
    get_uncertain_point_coords_on_grid,
    get_uncertain_point_coords_with_randomness,
    point_sample,
)
from .point_head import build_point_head


def calculate_uncertainty(sem_seg_logits):
    """
    For each location of the prediction `sem_seg_logits` we estimate uncerainty as the
        difference between top first and top second predicted logits.

    Args:
        mask_logits (Tensor): A tensor of shape (N, C, ...), where N is the minibatch size and
            C is the number of foreground classes. The values are logits.

    Returns:
        scores (Tensor): A tensor of shape (N, 1, ...) that contains uncertainty scores with
            the most uncertain locations having the highest uncertainty score.
    """
    top2_scores = torch.topk(sem_seg_logits, k=2, dim=1)[0]
    return (top2_scores[:, 1] - top2_scores[:, 0]).unsqueeze(1)


@SEM_SEG_HEADS_REGISTRY.register()
class PointRendSemSegHead(nn.Module):
    """
    A semantic segmentation head that combines a head set in `POINT_HEAD.COARSE_SEM_SEG_HEAD_NAME`
        and a point head set in `MODEL.POINT_HEAD.NAME`.
    """

    def __init__(self, cfg, input_shape: Dict[str, ShapeSpec]):
        super().__init__()

        self.ignore_value = cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE

        self.coarse_sem_seg_head = SEM_SEG_HEADS_REGISTRY.get(
            cfg.MODEL.POINT_HEAD.COARSE_SEM_SEG_HEAD_NAME
        )(cfg, input_shape)
        self._init_point_head(cfg, input_shape)

    def _init_point_head(self, cfg, input_shape: Dict[str, ShapeSpec]):
        # fmt: off
        assert cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES == cfg.MODEL.POINT_HEAD.NUM_CLASSES
        feature_channels             = {k: v.channels for k, v in input_shape.items()}
        self.in_features             = cfg.MODEL.POINT_HEAD.IN_FEATURES
        self.train_num_points        = cfg.MODEL.POINT_HEAD.TRAIN_NUM_POINTS
        self.oversample_ratio        = cfg.MODEL.POINT_HEAD.OVERSAMPLE_RATIO
        self.importance_sample_ratio = cfg.MODEL.POINT_HEAD.IMPORTANCE_SAMPLE_RATIO
        self.subdivision_steps       = cfg.MODEL.POINT_HEAD.SUBDIVISION_STEPS
        self.subdivision_num_points  = cfg.MODEL.POINT_HEAD.SUBDIVISION_NUM_POINTS
        # fmt: on

        in_channels = np.sum([feature_channels[f] for f in self.in_features])
        self.point_head = build_point_head(cfg, ShapeSpec(channels=in_channels, width=1, height=1))

    def forward(self, features, targets=None):
        coarse_sem_seg_logits = self.coarse_sem_seg_head.layers(features)

        if self.training:
            losses = self.coarse_sem_seg_head.losses(coarse_sem_seg_logits, targets)

            with torch.no_grad():
                point_coords = get_uncertain_point_coords_with_randomness(
                    coarse_sem_seg_logits,
                    calculate_uncertainty,
                    self.train_num_points,
                    self.oversample_ratio,
                    self.importance_sample_ratio,
                )
            coarse_features = point_sample(coarse_sem_seg_logits, point_coords, align_corners=False)

            fine_grained_features = cat(
                [
                    point_sample(features[in_feature], point_coords, align_corners=False)
                    for in_feature in self.in_features
                ]
            )
            point_logits = self.point_head(fine_grained_features, coarse_features)
            point_targets = (
                point_sample(
                    targets.unsqueeze(1).to(torch.float),
                    point_coords,
                    mode="nearest",
                    align_corners=False,
                )
                .squeeze(1)
                .to(torch.long)
            )
            losses["loss_sem_seg_point"] = F.cross_entropy(
                point_logits, point_targets, reduction="mean", ignore_index=self.ignore_value
            )
            return None, losses
        else:
            sem_seg_logits = coarse_sem_seg_logits.clone()
            for _ in range(self.subdivision_steps):
                sem_seg_logits = F.interpolate(
                    sem_seg_logits, scale_factor=2, mode="bilinear", align_corners=False
                )
                uncertainty_map = calculate_uncertainty(sem_seg_logits)
                point_indices, point_coords = get_uncertain_point_coords_on_grid(
                    uncertainty_map, self.subdivision_num_points
                )
                fine_grained_features = cat(
                    [
                        point_sample(features[in_feature], point_coords, align_corners=False)
                        for in_feature in self.in_features
                    ]
                )
                coarse_features = point_sample(
                    coarse_sem_seg_logits, point_coords, align_corners=False
                )
                point_logits = self.point_head(fine_grained_features, coarse_features)

                # put sem seg point predictions to the right places on the upsampled grid.
                N, C, H, W = sem_seg_logits.shape
                point_indices = point_indices.unsqueeze(1).expand(-1, C, -1)
                sem_seg_logits = (
                    sem_seg_logits.reshape(N, C, H * W)
                    .scatter_(2, point_indices, point_logits)
                    .view(N, C, H, W)
                )
            return sem_seg_logits, {}
