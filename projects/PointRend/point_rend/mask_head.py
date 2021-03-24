# Copyright (c) Facebook, Inc. and its affiliates.
import numpy as np
from typing import Dict, List
import fvcore.nn.weight_init as weight_init
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from detectron2.layers import Conv2d, ShapeSpec, cat, interpolate
from detectron2.modeling import ROI_MASK_HEAD_REGISTRY
from detectron2.modeling.roi_heads.mask_head import mask_rcnn_inference, mask_rcnn_loss
from detectron2.structures import Boxes

from .point_features import (
    generate_regular_grid_point_coords,
    get_uncertain_point_coords_on_grid,
    get_uncertain_point_coords_with_randomness,
    point_sample,
    point_sample_fine_grained_features,
)
from .point_head import build_point_head, roi_mask_point_loss


def calculate_uncertainty(logits, classes):
    """
    We estimate uncerainty as L1 distance between 0.0 and the logit prediction in 'logits' for the
        foreground class in `classes`.

    Args:
        logits (Tensor): A tensor of shape (R, C, ...) or (R, 1, ...) for class-specific or
            class-agnostic, where R is the total number of predicted masks in all images and C is
            the number of foreground classes. The values are logits.
        classes (list): A list of length R that contains either predicted of ground truth class
            for eash predicted mask.

    Returns:
        scores (Tensor): A tensor of shape (R, 1, ...) that contains uncertainty scores with
            the most uncertain locations having the highest uncertainty score.
    """
    if logits.shape[1] == 1:
        gt_class_logits = logits.clone()
    else:
        gt_class_logits = logits[
            torch.arange(logits.shape[0], device=logits.device), classes
        ].unsqueeze(1)
    return -(torch.abs(gt_class_logits))


class CoarseMaskHead(nn.Module):
    """
    A mask head with fully connected layers. Given pooled features it first reduces channels and
    spatial dimensions with conv layers and then uses FC layers to predict coarse masks analogously
    to the standard box head.
    """

    def __init__(self, cfg, input_shape: ShapeSpec):
        """
        The following attributes are parsed from config:
            conv_dim: the output dimension of the conv layers
            fc_dim: the feature dimenstion of the FC layers
            num_fc: the number of FC layers
            output_side_resolution: side resolution of the output square mask prediction
        """
        super().__init__()

        # fmt: off
        self.num_classes            = cfg.MODEL.ROI_HEADS.NUM_CLASSES
        conv_dim                    = cfg.MODEL.ROI_MASK_HEAD.CONV_DIM
        self.fc_dim                 = cfg.MODEL.ROI_MASK_HEAD.FC_DIM
        num_fc                      = cfg.MODEL.ROI_MASK_HEAD.NUM_FC
        self.output_side_resolution = cfg.MODEL.ROI_MASK_HEAD.OUTPUT_SIDE_RESOLUTION
        self.input_channels         = input_shape.channels
        self.input_h                = input_shape.height
        self.input_w                = input_shape.width
        # fmt: on

        self.conv_layers = []
        if self.input_channels > conv_dim:
            self.reduce_channel_dim_conv = Conv2d(
                self.input_channels,
                conv_dim,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True,
                activation=F.relu,
            )
            self.conv_layers.append(self.reduce_channel_dim_conv)

        self.reduce_spatial_dim_conv = Conv2d(
            conv_dim, conv_dim, kernel_size=2, stride=2, padding=0, bias=True, activation=F.relu
        )
        self.conv_layers.append(self.reduce_spatial_dim_conv)

        input_dim = conv_dim * self.input_h * self.input_w
        input_dim //= 4

        self.fcs = []
        for k in range(num_fc):
            fc = nn.Linear(input_dim, self.fc_dim)
            self.add_module("coarse_mask_fc{}".format(k + 1), fc)
            self.fcs.append(fc)
            input_dim = self.fc_dim

        output_dim = self.num_classes * self.output_side_resolution * self.output_side_resolution

        self.prediction = nn.Linear(self.fc_dim, output_dim)
        # use normal distribution initialization for mask prediction layer
        nn.init.normal_(self.prediction.weight, std=0.001)
        nn.init.constant_(self.prediction.bias, 0)

        for layer in self.conv_layers:
            weight_init.c2_msra_fill(layer)
        for layer in self.fcs:
            weight_init.c2_xavier_fill(layer)

    def forward(self, x):
        N = x.shape[0]
        for layer in self.conv_layers:
            x = layer(x)
        x = torch.flatten(x, start_dim=1)
        for layer in self.fcs:
            x = F.relu(layer(x))
        return self.prediction(x).view(
            N, self.num_classes, self.output_side_resolution, self.output_side_resolution
        )


@ROI_MASK_HEAD_REGISTRY.register()
class PointRendMaskHead(nn.Module):
    def __init__(self, cfg, input_shape: Dict[str, ShapeSpec]):
        super().__init__()
        self.roi_pooler_in_features = cfg.MODEL.ROI_MASK_HEAD.IN_FEATURES
        self.roi_pooler_size = cfg.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION
        self._feature_scales = {k: 1.0 / v.stride for k, v in input_shape.items()}
        in_channels = np.sum([input_shape[f].channels for f in self.roi_pooler_in_features])
        # coarse mask head
        self.coarse_head = CoarseMaskHead(
            cfg,
            ShapeSpec(
                channels=in_channels,
                width=self.roi_pooler_size,
                height=self.roi_pooler_size,
            ),
        )

        # point head
        self._init_point_head(cfg, input_shape)

    def _init_point_head(self, cfg, input_shape):
        # fmt: off
        self.mask_point_on                      = cfg.MODEL.ROI_MASK_HEAD.POINT_HEAD_ON
        if not self.mask_point_on:
            return
        assert cfg.MODEL.ROI_HEADS.NUM_CLASSES == cfg.MODEL.POINT_HEAD.NUM_CLASSES
        self.mask_point_in_features             = cfg.MODEL.POINT_HEAD.IN_FEATURES
        self.mask_point_train_num_points        = cfg.MODEL.POINT_HEAD.TRAIN_NUM_POINTS
        self.mask_point_oversample_ratio        = cfg.MODEL.POINT_HEAD.OVERSAMPLE_RATIO
        self.mask_point_importance_sample_ratio = cfg.MODEL.POINT_HEAD.IMPORTANCE_SAMPLE_RATIO
        # next three parameters are use in the adaptive subdivions inference procedure
        self.mask_point_subdivision_init_resolution = cfg.MODEL.ROI_MASK_HEAD.OUTPUT_SIDE_RESOLUTION
        self.mask_point_subdivision_steps       = cfg.MODEL.POINT_HEAD.SUBDIVISION_STEPS
        self.mask_point_subdivision_num_points  = cfg.MODEL.POINT_HEAD.SUBDIVISION_NUM_POINTS
        # fmt: on

        in_channels = np.sum([input_shape[f].channels for f in self.mask_point_in_features])
        self.point_head = build_point_head(cfg, ShapeSpec(channels=in_channels, width=1, height=1))

        # An optimization to skip unused subdivision steps: if after subdivision, all pixels on
        # the mask will be selected and recomputed anyway, we should just double our init_resolution
        while (
            4 * self.mask_point_subdivision_init_resolution ** 2
            <= self.mask_point_subdivision_num_points
        ):
            self.mask_point_subdivision_init_resolution *= 2
            self.mask_point_subdivision_steps -= 1

    def forward(self, features, instances):
        """
        Args:
            features (dict[str, Tensor]): a dict of image-level features
            instances (list[Instances]): proposals in training; detected
                instances in inference
        """
        if self.training:
            proposal_boxes = [x.proposal_boxes for x in instances]
            mask_coarse_logits = self.coarse_head(self._roi_pooler(features, proposal_boxes))

            losses = {"loss_mask": mask_rcnn_loss(mask_coarse_logits, instances)}
            losses.update(self._forward_mask_point(features, mask_coarse_logits, instances))
            return losses
        else:
            pred_boxes = [x.pred_boxes for x in instances]
            mask_coarse_logits = self.coarse_head(self._roi_pooler(features, pred_boxes))

            mask_logits = self._forward_mask_point(features, mask_coarse_logits, instances)
            mask_rcnn_inference(mask_logits, instances)
            return instances

    def _roi_pooler(self, features: List[Tensor], boxes: List[Boxes]):
        """
        Extract per-box feature. This is similar to RoIAlign(sampling_ratio=1) except:
        1. It's implemented by point_sample
        2. It pools features across all levels and concat them, while typically
           RoIAlign select one level for every box. However in the config we only use
           one level (p2) so there is no difference.

        Returns:
            Tensor of shape (R, C, pooler_size, pooler_size) where R is the total number of boxes
        """
        features_list = [features[k] for k in self.roi_pooler_in_features]
        features_scales = [self._feature_scales[k] for k in self.roi_pooler_in_features]

        num_boxes = sum(x.tensor.size(0) for x in boxes)
        output_size = self.roi_pooler_size
        point_coords = generate_regular_grid_point_coords(num_boxes, output_size, boxes[0].device)
        # For regular grids of points, this function is equivalent to `len(features_list)' calls
        # of `ROIAlign` (with `SAMPLING_RATIO=1`), and concat the results.
        roi_features, _ = point_sample_fine_grained_features(
            features_list, features_scales, boxes, point_coords
        )
        return roi_features.view(num_boxes, roi_features.shape[1], output_size, output_size)

    def _forward_mask_point(self, features, mask_coarse_logits, instances):
        """
        Forward logic of the mask point head.
        """
        if not self.mask_point_on:
            return {} if self.training else mask_coarse_logits

        mask_features_list = [features[k] for k in self.mask_point_in_features]
        features_scales = [self._feature_scales[k] for k in self.mask_point_in_features]

        if self.training:
            proposal_boxes = [x.proposal_boxes for x in instances]
            gt_classes = cat([x.gt_classes for x in instances])
            with torch.no_grad():
                point_coords = get_uncertain_point_coords_with_randomness(
                    mask_coarse_logits,
                    lambda logits: calculate_uncertainty(logits, gt_classes),
                    self.mask_point_train_num_points,
                    self.mask_point_oversample_ratio,
                    self.mask_point_importance_sample_ratio,
                )

            fine_grained_features, point_coords_wrt_image = point_sample_fine_grained_features(
                mask_features_list, features_scales, proposal_boxes, point_coords
            )
            coarse_features = point_sample(mask_coarse_logits, point_coords, align_corners=False)
            point_logits = self.point_head(fine_grained_features, coarse_features)
            return {
                "loss_mask_point": roi_mask_point_loss(
                    point_logits, instances, point_coords_wrt_image
                )
            }
        else:
            pred_boxes = [x.pred_boxes for x in instances]
            pred_classes = cat([x.pred_classes for x in instances])
            # The subdivision code will fail with the empty list of boxes
            if len(pred_classes) == 0:
                return mask_coarse_logits

            mask_logits = None
            # +1 here to include an initial step to generate the coarsest mask
            # prediction with init_resolution, when mask_logits is None.
            # We compute initial mask by sampling on a regular grid. coarse_mask
            # can be used as initial mask as well, but it's typically very low-res
            # so it will be completely overwritten during subdivision anyway.
            for _ in range(self.mask_point_subdivision_steps + 1):
                if mask_logits is None:
                    point_coords = generate_regular_grid_point_coords(
                        pred_classes.size(0),
                        self.mask_point_subdivision_init_resolution,
                        pred_boxes[0].device,
                    )
                else:
                    mask_logits = interpolate(
                        mask_logits, scale_factor=2, mode="bilinear", align_corners=False
                    )
                    uncertainty_map = calculate_uncertainty(mask_logits, pred_classes)
                    point_indices, point_coords = get_uncertain_point_coords_on_grid(
                        uncertainty_map, self.mask_point_subdivision_num_points
                    )

                # Run the point head for every point in point_coords
                fine_grained_features, _ = point_sample_fine_grained_features(
                    mask_features_list, features_scales, pred_boxes, point_coords
                )
                coarse_features = point_sample(
                    mask_coarse_logits, point_coords, align_corners=False
                )
                point_logits = self.point_head(fine_grained_features, coarse_features)

                if mask_logits is None:
                    # Create initial mask_logits using point_logits on this regular grid
                    R, C, _ = point_logits.shape
                    mask_logits = point_logits.reshape(
                        R,
                        C,
                        self.mask_point_subdivision_init_resolution,
                        self.mask_point_subdivision_init_resolution,
                    )
                else:
                    # Put point predictions to the right places on the upsampled grid.
                    R, C, H, W = mask_logits.shape
                    point_indices = point_indices.unsqueeze(1).expand(-1, C, -1)
                    mask_logits = (
                        mask_logits.reshape(R, C, H * W)
                        .scatter_(2, point_indices, point_logits)
                        .view(R, C, H, W)
                    )
            return mask_logits
