# Copyright (c) Facebook, Inc. and its affiliates.
import numpy as np
import fvcore.nn.weight_init as weight_init
import torch
from torch import nn
from torch.nn import functional as F

from detectron2.layers import ShapeSpec, cat
from detectron2.utils.events import get_event_storage
from detectron2.utils.registry import Registry

POINT_HEAD_REGISTRY = Registry("POINT_HEAD")
POINT_HEAD_REGISTRY.__doc__ = """
Registry for point heads, which makes prediction for a given set of per-point features.

The registered object will be called with `obj(cfg, input_shape)`.
"""


def roi_mask_point_loss(mask_logits, instances, point_labels):
    """
    Compute the point-based loss for instance segmentation mask predictions
    given point-wise mask prediction and its corresponding point-wise labels.
    Args:
        mask_logits (Tensor): A tensor of shape (R, C, P) or (R, 1, P) for class-specific or
            class-agnostic, where R is the total number of predicted masks in all images, C is the
            number of foreground classes, and P is the number of points sampled for each mask.
            The values are logits.
        instances (list[Instances]): A list of N Instances, where N is the number of images
            in the batch. These instances are in 1:1 correspondence with the `mask_logits`. So, i_th
            elememt of the list contains R_i objects and R_1 + ... + R_N is equal to R.
            The ground-truth labels (class, box, mask, ...) associated with each instance are stored
            in fields.
        point_labels (Tensor): A tensor of shape (R, P), where R is the total number of
            predicted masks and P is the number of points for each mask.
            Labels with value of -1 will be ignored.
    Returns:
        point_loss (Tensor): A scalar tensor containing the loss.
    """
    with torch.no_grad():
        cls_agnostic_mask = mask_logits.size(1) == 1
        total_num_masks = mask_logits.size(0)

        gt_classes = []
        for instances_per_image in instances:
            if len(instances_per_image) == 0:
                continue

            if not cls_agnostic_mask:
                gt_classes_per_image = instances_per_image.gt_classes.to(dtype=torch.int64)
                gt_classes.append(gt_classes_per_image)

    gt_mask_logits = point_labels
    point_ignores = point_labels == -1
    if gt_mask_logits.shape[0] == 0:
        return mask_logits.sum() * 0

    assert gt_mask_logits.numel() > 0, gt_mask_logits.shape

    if cls_agnostic_mask:
        mask_logits = mask_logits[:, 0]
    else:
        indices = torch.arange(total_num_masks)
        gt_classes = cat(gt_classes, dim=0)
        mask_logits = mask_logits[indices, gt_classes]

    # Log the training accuracy (using gt classes and 0.0 threshold for the logits)
    mask_accurate = (mask_logits > 0.0) == gt_mask_logits.to(dtype=torch.uint8)
    mask_accurate = mask_accurate[~point_ignores]
    mask_accuracy = mask_accurate.nonzero().size(0) / max(mask_accurate.numel(), 1.0)
    get_event_storage().put_scalar("point/accuracy", mask_accuracy)

    point_loss = F.binary_cross_entropy_with_logits(
        mask_logits, gt_mask_logits.to(dtype=torch.float32), weight=~point_ignores, reduction="mean"
    )
    return point_loss


@POINT_HEAD_REGISTRY.register()
class StandardPointHead(nn.Module):
    """
    A point head multi-layer perceptron which we model with conv1d layers with kernel 1. The head
    takes both fine-grained and coarse prediction features as its input.
    """

    def __init__(self, cfg, input_shape: ShapeSpec):
        """
        The following attributes are parsed from config:
            fc_dim: the output dimension of each FC layers
            num_fc: the number of FC layers
            coarse_pred_each_layer: if True, coarse prediction features are concatenated to each
                layer's input
        """
        super(StandardPointHead, self).__init__()
        # fmt: off
        num_classes                 = cfg.MODEL.POINT_HEAD.NUM_CLASSES
        fc_dim                      = cfg.MODEL.POINT_HEAD.FC_DIM
        num_fc                      = cfg.MODEL.POINT_HEAD.NUM_FC
        cls_agnostic_mask           = cfg.MODEL.POINT_HEAD.CLS_AGNOSTIC_MASK
        self.coarse_pred_each_layer = cfg.MODEL.POINT_HEAD.COARSE_PRED_EACH_LAYER
        input_channels              = input_shape.channels
        # fmt: on

        fc_dim_in = input_channels + num_classes
        self.fc_layers = []
        for k in range(num_fc):
            fc = nn.Conv1d(fc_dim_in, fc_dim, kernel_size=1, stride=1, padding=0, bias=True)
            self.add_module("fc{}".format(k + 1), fc)
            self.fc_layers.append(fc)
            fc_dim_in = fc_dim
            fc_dim_in += num_classes if self.coarse_pred_each_layer else 0

        num_mask_classes = 1 if cls_agnostic_mask else num_classes
        self.predictor = nn.Conv1d(fc_dim_in, num_mask_classes, kernel_size=1, stride=1, padding=0)

        for layer in self.fc_layers:
            weight_init.c2_msra_fill(layer)
        # use normal distribution initialization for mask prediction layer
        nn.init.normal_(self.predictor.weight, std=0.001)
        if self.predictor.bias is not None:
            nn.init.constant_(self.predictor.bias, 0)

    def forward(self, fine_grained_features, coarse_features):
        x = torch.cat((fine_grained_features, coarse_features), dim=1)
        for layer in self.fc_layers:
            x = F.relu(layer(x))
            if self.coarse_pred_each_layer:
                x = cat((x, coarse_features), dim=1)
        return self.predictor(x)


@POINT_HEAD_REGISTRY.register()
class ImplicitPointHead(nn.Module):
    """
    A point head multi-layer perceptron which we model with conv1d layers with kernel 1. The head
    takes both fine-grained features and instance-wise MLP parameters as its input.
    """

    def __init__(self, cfg, input_shape: ShapeSpec):
        """
        The following attributes are parsed from config:
            channels: the output dimension of each FC layers
            num_layers: the number of FC layers (including the final prediction layer)
            image_feature_enabled: if True, fine-grained image-level features are used
            positional_encoding_enabled: if True, positional encoding is used
        """
        super(ImplicitPointHead, self).__init__()
        # fmt: off
        self.num_layers                         = cfg.MODEL.POINT_HEAD.NUM_FC + 1
        self.channels                           = cfg.MODEL.POINT_HEAD.FC_DIM
        self.image_feature_enabled              = cfg.MODEL.IMPLICIT_POINTREND.IMAGE_FEATURE_ENABLED
        self.positional_encoding_enabled        = cfg.MODEL.IMPLICIT_POINTREND.POS_ENC_ENABLED
        self.num_classes                        = (
            cfg.MODEL.POINT_HEAD.NUM_CLASSES if not cfg.MODEL.POINT_HEAD.CLS_AGNOSTIC_MASK else 1
        )
        self.in_channels                        = input_shape.channels
        # fmt: on

        if not self.image_feature_enabled:
            self.in_channels = 0
        if self.positional_encoding_enabled:
            self.in_channels += 256
            self.register_buffer("positional_encoding_gaussian_matrix", torch.randn((2, 128)))

        assert self.in_channels > 0

        num_weight_params, num_bias_params = [], []
        assert self.num_layers >= 2
        for l in range(self.num_layers):
            if l == 0:
                # input layer
                num_weight_params.append(self.in_channels * self.channels)
                num_bias_params.append(self.channels)
            elif l == self.num_layers - 1:
                # output layer
                num_weight_params.append(self.channels * self.num_classes)
                num_bias_params.append(self.num_classes)
            else:
                # intermediate layer
                num_weight_params.append(self.channels * self.channels)
                num_bias_params.append(self.channels)

        self.num_weight_params = num_weight_params
        self.num_bias_params = num_bias_params
        self.num_params = sum(num_weight_params) + sum(num_bias_params)

    def forward(self, fine_grained_features, point_coords, parameters):
        # features: [R, channels, K]
        # point_coords: [R, K, 2]
        num_instances = fine_grained_features.size(0)
        num_points = fine_grained_features.size(2)

        if num_instances == 0:
            return torch.zeros((0, 1, num_points), device=fine_grained_features.device)

        if self.positional_encoding_enabled:
            # locations: [R*K, 2]
            locations = 2 * point_coords.reshape(num_instances * num_points, 2) - 1
            locations = locations @ self.positional_encoding_gaussian_matrix.to(locations.device)
            locations = 2 * np.pi * locations
            locations = torch.cat([torch.sin(locations), torch.cos(locations)], dim=1)
            # locations: [R, C, K]
            locations = locations.reshape(num_instances, num_points, 256).permute(0, 2, 1)
            if not self.image_feature_enabled:
                fine_grained_features = locations
            else:
                fine_grained_features = torch.cat([locations, fine_grained_features], dim=1)

        # features [R, C, K]
        mask_feat = fine_grained_features.reshape(num_instances, self.in_channels, num_points)

        weights, biases = self._parse_params(
            parameters,
            self.in_channels,
            self.channels,
            self.num_classes,
            self.num_weight_params,
            self.num_bias_params,
        )

        point_logits = self._dynamic_mlp(mask_feat, weights, biases, num_instances)
        point_logits = point_logits.reshape(-1, self.num_classes, num_points)

        return point_logits

    @staticmethod
    def _dynamic_mlp(features, weights, biases, num_instances):
        assert features.dim() == 3, features.dim()
        n_layers = len(weights)
        x = features
        for i, (w, b) in enumerate(zip(weights, biases)):
            x = torch.einsum("nck,ndc->ndk", x, w) + b
            if i < n_layers - 1:
                x = F.relu(x)
        return x

    @staticmethod
    def _parse_params(
        pred_params,
        in_channels,
        channels,
        num_classes,
        num_weight_params,
        num_bias_params,
    ):
        assert pred_params.dim() == 2
        assert len(num_weight_params) == len(num_bias_params)
        assert pred_params.size(1) == sum(num_weight_params) + sum(num_bias_params)

        num_instances = pred_params.size(0)
        num_layers = len(num_weight_params)

        params_splits = list(
            torch.split_with_sizes(pred_params, num_weight_params + num_bias_params, dim=1)
        )

        weight_splits = params_splits[:num_layers]
        bias_splits = params_splits[num_layers:]

        for l in range(num_layers):
            if l == 0:
                # input layer
                weight_splits[l] = weight_splits[l].reshape(num_instances, channels, in_channels)
                bias_splits[l] = bias_splits[l].reshape(num_instances, channels, 1)
            elif l < num_layers - 1:
                # intermediate layer
                weight_splits[l] = weight_splits[l].reshape(num_instances, channels, channels)
                bias_splits[l] = bias_splits[l].reshape(num_instances, channels, 1)
            else:
                # output layer
                weight_splits[l] = weight_splits[l].reshape(num_instances, num_classes, channels)
                bias_splits[l] = bias_splits[l].reshape(num_instances, num_classes, 1)

        return weight_splits, bias_splits


def build_point_head(cfg, input_channels):
    """
    Build a point head defined by `cfg.MODEL.POINT_HEAD.NAME`.
    """
    head_name = cfg.MODEL.POINT_HEAD.NAME
    return POINT_HEAD_REGISTRY.get(head_name)(cfg, input_channels)
