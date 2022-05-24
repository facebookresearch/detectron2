# Copyright (c) Facebook, Inc. and its affiliates.
import logging
import math
import numpy as np
from typing import Dict, List, Tuple
import fvcore.nn.weight_init as weight_init
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.layers import Conv2d, ShapeSpec, cat, interpolate
from detectron2.modeling import ROI_MASK_HEAD_REGISTRY
from detectron2.modeling.roi_heads.mask_head import mask_rcnn_inference, mask_rcnn_loss
from detectron2.structures import Boxes

from .point_features import (
    generate_regular_grid_point_coords,
    get_point_coords_wrt_image,
    get_uncertain_point_coords_on_grid,
    get_uncertain_point_coords_with_randomness,
    point_sample,
    point_sample_fine_grained_features,
    sample_point_labels,
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


class ConvFCHead(nn.Module):
    """
    A mask head with fully connected layers. Given pooled features it first reduces channels and
    spatial dimensions with conv layers and then uses FC layers to predict coarse masks analogously
    to the standard box head.
    """

    _version = 2

    @configurable
    def __init__(
        self, input_shape: ShapeSpec, *, conv_dim: int, fc_dims: List[int], output_shape: Tuple[int]
    ):
        """
        Args:
            conv_dim: the output dimension of the conv layers
            fc_dims: a list of N>0 integers representing the output dimensions of N FC layers
            output_shape: shape of the output mask prediction
        """
        super().__init__()

        # fmt: off
        input_channels    = input_shape.channels
        input_h           = input_shape.height
        input_w           = input_shape.width
        self.output_shape = output_shape
        # fmt: on

        self.conv_layers = []
        if input_channels > conv_dim:
            self.reduce_channel_dim_conv = Conv2d(
                input_channels,
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

        input_dim = conv_dim * input_h * input_w
        input_dim //= 4

        self.fcs = []
        for k, fc_dim in enumerate(fc_dims):
            fc = nn.Linear(input_dim, fc_dim)
            self.add_module("fc{}".format(k + 1), fc)
            self.fcs.append(fc)
            input_dim = fc_dim

        output_dim = int(np.prod(self.output_shape))

        self.prediction = nn.Linear(fc_dims[-1], output_dim)
        # use normal distribution initialization for mask prediction layer
        nn.init.normal_(self.prediction.weight, std=0.001)
        nn.init.constant_(self.prediction.bias, 0)

        for layer in self.conv_layers:
            weight_init.c2_msra_fill(layer)
        for layer in self.fcs:
            weight_init.c2_xavier_fill(layer)

    @classmethod
    def from_config(cls, cfg, input_shape):
        output_shape = (
            cfg.MODEL.ROI_HEADS.NUM_CLASSES,
            cfg.MODEL.ROI_MASK_HEAD.OUTPUT_SIDE_RESOLUTION,
            cfg.MODEL.ROI_MASK_HEAD.OUTPUT_SIDE_RESOLUTION,
        )
        fc_dim = cfg.MODEL.ROI_MASK_HEAD.FC_DIM
        num_fc = cfg.MODEL.ROI_MASK_HEAD.NUM_FC
        ret = dict(
            input_shape=input_shape,
            conv_dim=cfg.MODEL.ROI_MASK_HEAD.CONV_DIM,
            fc_dims=[fc_dim] * num_fc,
            output_shape=output_shape,
        )
        return ret

    def forward(self, x):
        N = x.shape[0]
        for layer in self.conv_layers:
            x = layer(x)
        x = torch.flatten(x, start_dim=1)
        for layer in self.fcs:
            x = F.relu(layer(x))
        output_shape = [N] + list(self.output_shape)
        return self.prediction(x).view(*output_shape)

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        version = local_metadata.get("version", None)

        if version is None or version < 2:
            logger = logging.getLogger(__name__)
            logger.warning(
                "Weight format of PointRend models have changed! "
                "Applying automatic conversion now ..."
            )
            for k in list(state_dict.keys()):
                newk = k
                if k.startswith(prefix + "coarse_mask_fc"):
                    newk = k.replace(prefix + "coarse_mask_fc", prefix + "fc")
                if newk != k:
                    state_dict[newk] = state_dict[k]
                    del state_dict[k]


@ROI_MASK_HEAD_REGISTRY.register()
class PointRendMaskHead(nn.Module):
    def __init__(self, cfg, input_shape: Dict[str, ShapeSpec]):
        super().__init__()
        self._feature_scales = {k: 1.0 / v.stride for k, v in input_shape.items()}
        # point head
        self._init_point_head(cfg, input_shape)
        # coarse mask head
        self.roi_pooler_in_features = cfg.MODEL.ROI_MASK_HEAD.IN_FEATURES
        self.roi_pooler_size = cfg.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION
        self._feature_scales = {k: 1.0 / v.stride for k, v in input_shape.items()}
        in_channels = np.sum([input_shape[f].channels for f in self.roi_pooler_in_features])
        self._init_roi_head(
            cfg,
            ShapeSpec(
                channels=in_channels,
                width=self.roi_pooler_size,
                height=self.roi_pooler_size,
            ),
        )

    def _init_roi_head(self, cfg, input_shape):
        self.coarse_head = ConvFCHead(cfg, input_shape)

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

        in_channels = int(np.sum([input_shape[f].channels for f in self.mask_point_in_features]))
        self.point_head = build_point_head(cfg, ShapeSpec(channels=in_channels, width=1, height=1))

        # An optimization to skip unused subdivision steps: if after subdivision, all pixels on
        # the mask will be selected and recomputed anyway, we should just double our init_resolution
        while (
            4 * self.mask_point_subdivision_init_resolution**2
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
            coarse_mask = self.coarse_head(self._roi_pooler(features, proposal_boxes))
            losses = {"loss_mask": mask_rcnn_loss(coarse_mask, instances)}
            if not self.mask_point_on:
                return losses

            point_coords, point_labels = self._sample_train_points(coarse_mask, instances)
            point_fine_grained_features = self._point_pooler(features, proposal_boxes, point_coords)
            point_logits = self._get_point_logits(
                point_fine_grained_features, point_coords, coarse_mask
            )
            losses["loss_mask_point"] = roi_mask_point_loss(point_logits, instances, point_labels)
            return losses
        else:
            pred_boxes = [x.pred_boxes for x in instances]
            coarse_mask = self.coarse_head(self._roi_pooler(features, pred_boxes))
            return self._subdivision_inference(features, coarse_mask, instances)

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

    def _sample_train_points(self, coarse_mask, instances):
        assert self.training
        gt_classes = cat([x.gt_classes for x in instances])
        with torch.no_grad():
            # sample point_coords
            point_coords = get_uncertain_point_coords_with_randomness(
                coarse_mask,
                lambda logits: calculate_uncertainty(logits, gt_classes),
                self.mask_point_train_num_points,
                self.mask_point_oversample_ratio,
                self.mask_point_importance_sample_ratio,
            )
            # sample point_labels
            proposal_boxes = [x.proposal_boxes for x in instances]
            cat_boxes = Boxes.cat(proposal_boxes)
            point_coords_wrt_image = get_point_coords_wrt_image(cat_boxes.tensor, point_coords)
            point_labels = sample_point_labels(instances, point_coords_wrt_image)
        return point_coords, point_labels

    def _point_pooler(self, features, proposal_boxes, point_coords):
        point_features_list = [features[k] for k in self.mask_point_in_features]
        point_features_scales = [self._feature_scales[k] for k in self.mask_point_in_features]
        # sample image-level features
        point_fine_grained_features, _ = point_sample_fine_grained_features(
            point_features_list, point_features_scales, proposal_boxes, point_coords
        )
        return point_fine_grained_features

    def _get_point_logits(self, point_fine_grained_features, point_coords, coarse_mask):
        coarse_features = point_sample(coarse_mask, point_coords, align_corners=False)
        point_logits = self.point_head(point_fine_grained_features, coarse_features)
        return point_logits

    def _subdivision_inference(self, features, mask_representations, instances):
        assert not self.training

        pred_boxes = [x.pred_boxes for x in instances]
        pred_classes = cat([x.pred_classes for x in instances])

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
            fine_grained_features = self._point_pooler(features, pred_boxes, point_coords)
            point_logits = self._get_point_logits(
                fine_grained_features, point_coords, mask_representations
            )

            if mask_logits is None:
                # Create initial mask_logits using point_logits on this regular grid
                R, C, _ = point_logits.shape
                mask_logits = point_logits.reshape(
                    R,
                    C,
                    self.mask_point_subdivision_init_resolution,
                    self.mask_point_subdivision_init_resolution,
                )
                # The subdivision code will fail with the empty list of boxes
                if len(pred_classes) == 0:
                    mask_rcnn_inference(mask_logits, instances)
                    return instances
            else:
                # Put point predictions to the right places on the upsampled grid.
                R, C, H, W = mask_logits.shape
                point_indices = point_indices.unsqueeze(1).expand(-1, C, -1)
                mask_logits = (
                    mask_logits.reshape(R, C, H * W)
                    .scatter_(2, point_indices, point_logits)
                    .view(R, C, H, W)
                )
        mask_rcnn_inference(mask_logits, instances)
        return instances


@ROI_MASK_HEAD_REGISTRY.register()
class ImplicitPointRendMaskHead(PointRendMaskHead):
    def __init__(self, cfg, input_shape: Dict[str, ShapeSpec]):
        super().__init__(cfg, input_shape)

    def _init_roi_head(self, cfg, input_shape):
        assert hasattr(self, "num_params"), "Please initialize point_head first!"
        self.parameter_head = ConvFCHead(cfg, input_shape, output_shape=(self.num_params,))
        self.regularizer = cfg.MODEL.IMPLICIT_POINTREND.PARAMS_L2_REGULARIZER

    def _init_point_head(self, cfg, input_shape):
        # fmt: off
        self.mask_point_on = True  # always on
        assert cfg.MODEL.ROI_HEADS.NUM_CLASSES == cfg.MODEL.POINT_HEAD.NUM_CLASSES
        self.mask_point_in_features             = cfg.MODEL.POINT_HEAD.IN_FEATURES
        self.mask_point_train_num_points        = cfg.MODEL.POINT_HEAD.TRAIN_NUM_POINTS
        # next two parameters are use in the adaptive subdivions inference procedure
        self.mask_point_subdivision_steps       = cfg.MODEL.POINT_HEAD.SUBDIVISION_STEPS
        self.mask_point_subdivision_num_points  = cfg.MODEL.POINT_HEAD.SUBDIVISION_NUM_POINTS
        # fmt: on

        in_channels = int(np.sum([input_shape[f].channels for f in self.mask_point_in_features]))
        self.point_head = build_point_head(cfg, ShapeSpec(channels=in_channels, width=1, height=1))
        self.num_params = self.point_head.num_params

        # inference parameters
        self.mask_point_subdivision_init_resolution = int(
            math.sqrt(self.mask_point_subdivision_num_points)
        )
        assert (
            self.mask_point_subdivision_init_resolution
            * self.mask_point_subdivision_init_resolution
            == self.mask_point_subdivision_num_points
        )

    def forward(self, features, instances):
        """
        Args:
            features (dict[str, Tensor]): a dict of image-level features
            instances (list[Instances]): proposals in training; detected
                instances in inference
        """
        if self.training:
            proposal_boxes = [x.proposal_boxes for x in instances]
            parameters = self.parameter_head(self._roi_pooler(features, proposal_boxes))
            losses = {"loss_l2": self.regularizer * (parameters**2).mean()}

            point_coords, point_labels = self._uniform_sample_train_points(instances)
            point_fine_grained_features = self._point_pooler(features, proposal_boxes, point_coords)
            point_logits = self._get_point_logits(
                point_fine_grained_features, point_coords, parameters
            )
            losses["loss_mask_point"] = roi_mask_point_loss(point_logits, instances, point_labels)
            return losses
        else:
            pred_boxes = [x.pred_boxes for x in instances]
            parameters = self.parameter_head(self._roi_pooler(features, pred_boxes))
            return self._subdivision_inference(features, parameters, instances)

    def _uniform_sample_train_points(self, instances):
        assert self.training
        proposal_boxes = [x.proposal_boxes for x in instances]
        cat_boxes = Boxes.cat(proposal_boxes)
        # uniform sample
        point_coords = torch.rand(
            len(cat_boxes), self.mask_point_train_num_points, 2, device=cat_boxes.tensor.device
        )
        # sample point_labels
        point_coords_wrt_image = get_point_coords_wrt_image(cat_boxes.tensor, point_coords)
        point_labels = sample_point_labels(instances, point_coords_wrt_image)
        return point_coords, point_labels

    def _get_point_logits(self, fine_grained_features, point_coords, parameters):
        return self.point_head(fine_grained_features, point_coords, parameters)
