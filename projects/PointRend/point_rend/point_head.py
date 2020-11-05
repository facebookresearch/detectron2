# Copyright (c) Facebook, Inc. and its affiliates.
import fvcore.nn.weight_init as weight_init
import torch
from torch import nn
from torch.nn import functional as F

from detectron2.layers import ShapeSpec, cat
from detectron2.structures import BitMasks
from detectron2.utils.events import get_event_storage
from detectron2.utils.registry import Registry

from .point_features import point_sample

POINT_HEAD_REGISTRY = Registry("POINT_HEAD")
POINT_HEAD_REGISTRY.__doc__ = """
Registry for point heads, which makes prediction for a given set of per-point features.

The registered object will be called with `obj(cfg, input_shape)`.
"""


def roi_mask_point_loss(mask_logits, instances, points_coord):
    """
    Compute the point-based loss for instance segmentation mask predictions.

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
        points_coords (Tensor): A tensor of shape (R, P, 2), where R is the total number of
            predicted masks and P is the number of points for each mask. The coordinates are in
            the image pixel coordinate space, i.e. [0, H] x [0, W].
    Returns:
        point_loss (Tensor): A scalar tensor containing the loss.
    """
    with torch.no_grad():
        cls_agnostic_mask = mask_logits.size(1) == 1
        total_num_masks = mask_logits.size(0)

        gt_classes = []
        gt_mask_logits = []
        idx = 0
        for instances_per_image in instances:
            if len(instances_per_image) == 0:
                continue
            assert isinstance(
                instances_per_image.gt_masks, BitMasks
            ), "Point head works with GT in 'bitmask' format. Set INPUT.MASK_FORMAT to 'bitmask'."

            if not cls_agnostic_mask:
                gt_classes_per_image = instances_per_image.gt_classes.to(dtype=torch.int64)
                gt_classes.append(gt_classes_per_image)

            gt_bit_masks = instances_per_image.gt_masks.tensor
            h, w = instances_per_image.gt_masks.image_size
            scale = torch.tensor([w, h], dtype=torch.float, device=gt_bit_masks.device)
            points_coord_grid_sample_format = (
                points_coord[idx : idx + len(instances_per_image)] / scale
            )
            idx += len(instances_per_image)
            gt_mask_logits.append(
                point_sample(
                    gt_bit_masks.to(torch.float32).unsqueeze(1),
                    points_coord_grid_sample_format,
                    align_corners=False,
                ).squeeze(1)
            )

    if len(gt_mask_logits) == 0:
        return mask_logits.sum() * 0

    gt_mask_logits = cat(gt_mask_logits)
    assert gt_mask_logits.numel() > 0, gt_mask_logits.shape

    if cls_agnostic_mask:
        mask_logits = mask_logits[:, 0]
    else:
        indices = torch.arange(total_num_masks)
        gt_classes = cat(gt_classes, dim=0)
        mask_logits = mask_logits[indices, gt_classes]

    # Log the training accuracy (using gt classes and 0.0 threshold for the logits)
    mask_accurate = (mask_logits > 0.0) == gt_mask_logits.to(dtype=torch.uint8)
    mask_accuracy = mask_accurate.nonzero().size(0) / mask_accurate.numel()
    get_event_storage().put_scalar("point_rend/accuracy", mask_accuracy)

    point_loss = F.binary_cross_entropy_with_logits(
        mask_logits, gt_mask_logits.to(dtype=torch.float32), reduction="mean"
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


def build_point_head(cfg, input_channels):
    """
    Build a point head defined by `cfg.MODEL.POINT_HEAD.NAME`.
    """
    head_name = cfg.MODEL.POINT_HEAD.NAME
    return POINT_HEAD_REGISTRY.get(head_name)(cfg, input_channels)
