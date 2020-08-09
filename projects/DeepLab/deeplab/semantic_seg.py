# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from typing import Dict
import fvcore.nn.weight_init as weight_init
import torch
from torch import nn
from torch.nn import functional as F

from detectron2.layers import ASPP, Conv2d, ShapeSpec, get_norm
from detectron2.modeling import SEM_SEG_HEADS_REGISTRY

from .loss import DeepLabCE


@SEM_SEG_HEADS_REGISTRY.register()
class DeepLabV3PlusHead(nn.Module):
    """
    A semantic segmentation head described in :paper:`DeepLabV3+`.
    """

    def __init__(self, cfg, input_shape: Dict[str, ShapeSpec]):
        super().__init__()

        # fmt: off
        self.in_features      = cfg.MODEL.SEM_SEG_HEAD.IN_FEATURES  # starting from "res2" to "res5"
        in_channels           = [input_shape[f].channels for f in self.in_features]
        project_features      = cfg.MODEL.SEM_SEG_HEAD.PROJECT_FEATURES
        project_channels      = cfg.MODEL.SEM_SEG_HEAD.PROJECT_CHANNELS
        aspp_channels         = cfg.MODEL.SEM_SEG_HEAD.ASPP_CHANNELS
        aspp_dilations        = cfg.MODEL.SEM_SEG_HEAD.ASPP_DILATIONS
        self.ignore_value     = cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE
        num_classes           = cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES
        conv_dims             = cfg.MODEL.SEM_SEG_HEAD.CONVS_DIM
        self.common_stride    = cfg.MODEL.SEM_SEG_HEAD.COMMON_STRIDE  # output stride
        norm                  = cfg.MODEL.SEM_SEG_HEAD.NORM
        self.loss_weight      = cfg.MODEL.SEM_SEG_HEAD.LOSS_WEIGHT
        self.loss_type        = cfg.MODEL.SEM_SEG_HEAD.LOSS_TYPE
        train_crop_size       = cfg.INPUT.CROP.SIZE
        aspp_dropout          = cfg.MODEL.SEM_SEG_HEAD.ASPP_DROPOUT
        res5_dilation         = cfg.MODEL.RESNETS.RES5_DILATION
        # fmt: on

        assert len(project_features) == len(self.in_features) - 1
        self.decoder = nn.ModuleDict()

        use_bias = norm == ""
        for idx, in_channels in enumerate(in_channels):
            decoder_stage = nn.ModuleDict()

            if idx == len(self.in_features) - 1:
                # ASPP module
                if cfg.INPUT.CROP.ENABLED:
                    assert cfg.INPUT.CROP.TYPE == "absolute"
                    train_crop_h, train_crop_w = train_crop_size
                    encoder_stride = 32 // res5_dilation
                    if train_crop_h % encoder_stride or train_crop_w % encoder_stride:
                        raise ValueError("Crop size need to be divisible by encoder stride.")
                    pool_h = train_crop_h // encoder_stride
                    pool_w = train_crop_w // encoder_stride
                    pool_kernel_size = (pool_h, pool_w)
                else:
                    pool_kernel_size = None
                project_conv = ASPP(
                    in_channels,
                    aspp_channels,
                    aspp_dilations,
                    norm=norm,
                    activation=F.relu,
                    pool_kernel_size=pool_kernel_size,
                    dropout=aspp_dropout,
                )
                fuse_conv = None
            else:
                project_conv = Conv2d(
                    in_channels,
                    project_channels[idx],
                    kernel_size=1,
                    bias=use_bias,
                    norm=get_norm(norm, project_channels[idx]),
                    activation=F.relu,
                )
                fuse_conv = nn.Sequential(
                    Conv2d(
                        project_channels[idx] + conv_dims,
                        conv_dims,
                        kernel_size=3,
                        padding=1,
                        bias=use_bias,
                        norm=get_norm(norm, conv_dims),
                        activation=F.relu,
                    ),
                    Conv2d(
                        conv_dims,
                        conv_dims,
                        kernel_size=3,
                        padding=1,
                        bias=use_bias,
                        norm=get_norm(norm, conv_dims),
                        activation=F.relu,
                    ),
                )
                weight_init.c2_xavier_fill(project_conv)
                weight_init.c2_xavier_fill(fuse_conv[0])
                weight_init.c2_xavier_fill(fuse_conv[1])

            decoder_stage["project_conv"] = project_conv
            decoder_stage["fuse_conv"] = fuse_conv

            self.decoder[self.in_features[idx]] = decoder_stage

        self.predictor = Conv2d(conv_dims, num_classes, kernel_size=1, stride=1, padding=0)
        nn.init.normal_(self.predictor.weight, 0, 0.001)
        nn.init.constant_(self.predictor.bias, 0)

        if self.loss_type == "cross_entropy":
            self.loss = nn.CrossEntropyLoss(reduction="mean", ignore_index=self.ignore_value)
        elif self.loss_type == "hard_pixel_mining":
            self.loss = DeepLabCE(ignore_label=self.ignore_value, top_k_percent_pixels=0.2)
        else:
            raise ValueError("Unexpected loss type: %s" % self.loss_type)

    def forward(self, features, targets=None):
        """
        Returns:
            In training, returns (None, dict of losses)
            In inference, returns (CxHxW logits, {})
        """
        # Reverse feature maps into top-down order (from low to high resolution)
        for f in self.in_features[::-1]:
            x = features[f]
            proj_x = self.decoder[f]["project_conv"](x)
            if self.decoder[f]["fuse_conv"] is None:
                # This is aspp module
                y = proj_x
            else:
                # Upsample y
                y = F.interpolate(y, size=proj_x.size()[2:], mode="bilinear", align_corners=False)
                y = torch.cat([proj_x, y], dim=1)
                y = self.decoder[f]["fuse_conv"](y)
        y = self.predictor(y)
        if self.training:
            return None, self.losses(y, targets)
        else:
            y = F.interpolate(
                y, scale_factor=self.common_stride, mode="bilinear", align_corners=False
            )
            return y, {}

    def losses(self, predictions, targets):
        predictions = F.interpolate(
            predictions, scale_factor=self.common_stride, mode="bilinear", align_corners=False
        )
        loss = self.loss(predictions, targets)
        losses = {"loss_sem_seg": loss * self.loss_weight}
        return losses


@SEM_SEG_HEADS_REGISTRY.register()
class DeepLabV3Head(nn.Module):
    """
    A semantic segmentation head described in :paper:`DeepLabV3`.
    """

    def __init__(self, cfg, input_shape: Dict[str, ShapeSpec]):
        super().__init__()

        # fmt: off
        self.in_features      = cfg.MODEL.SEM_SEG_HEAD.IN_FEATURES
        in_channels           = [input_shape[f].channels for f in self.in_features]
        aspp_channels         = cfg.MODEL.SEM_SEG_HEAD.ASPP_CHANNELS
        aspp_dilations        = cfg.MODEL.SEM_SEG_HEAD.ASPP_DILATIONS
        self.ignore_value     = cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE
        num_classes           = cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES
        conv_dims             = cfg.MODEL.SEM_SEG_HEAD.CONVS_DIM
        self.common_stride    = cfg.MODEL.SEM_SEG_HEAD.COMMON_STRIDE  # output stride
        norm                  = cfg.MODEL.SEM_SEG_HEAD.NORM
        self.loss_weight      = cfg.MODEL.SEM_SEG_HEAD.LOSS_WEIGHT
        self.loss_type        = cfg.MODEL.SEM_SEG_HEAD.LOSS_TYPE
        train_crop_size       = cfg.INPUT.CROP.SIZE
        aspp_dropout          = cfg.MODEL.SEM_SEG_HEAD.ASPP_DROPOUT
        # fmt: on

        assert len(self.in_features) == 1
        assert len(in_channels) == 1

        # ASPP module
        if cfg.INPUT.CROP.ENABLED:
            assert cfg.INPUT.CROP.TYPE == "absolute"
            train_crop_h, train_crop_w = train_crop_size
            if train_crop_h % self.common_stride or train_crop_w % self.common_stride:
                raise ValueError("Crop size need to be divisible by output stride.")
            pool_h = train_crop_h // self.common_stride
            pool_w = train_crop_w // self.common_stride
            pool_kernel_size = (pool_h, pool_w)
        else:
            pool_kernel_size = None
        self.aspp = ASPP(
            in_channels[0],
            aspp_channels,
            aspp_dilations,
            norm=norm,
            activation=F.relu,
            pool_kernel_size=pool_kernel_size,
            dropout=aspp_dropout,
        )

        self.predictor = Conv2d(conv_dims, num_classes, kernel_size=1, stride=1, padding=0)
        nn.init.normal_(self.predictor.weight, 0, 0.001)
        nn.init.constant_(self.predictor.bias, 0)

        if self.loss_type == "cross_entropy":
            self.loss = nn.CrossEntropyLoss(reduction="mean", ignore_index=self.ignore_value)
        elif self.loss_type == "hard_pixel_mining":
            self.loss = DeepLabCE(ignore_label=self.ignore_value, top_k_percent_pixels=0.2)
        else:
            raise ValueError("Unexpected loss type: %s" % self.loss_type)

    def forward(self, features, targets=None):
        """
        Returns:
            In training, returns (None, dict of losses)
            In inference, returns (CxHxW logits, {})
        """
        x = features[self.in_features[0]]
        x = self.aspp(x)
        x = self.predictor(x)
        if self.training:
            return None, self.losses(x, targets)
        else:
            x = F.interpolate(
                x, scale_factor=self.common_stride, mode="bilinear", align_corners=False
            )
            return x, {}

    def losses(self, predictions, targets):
        predictions = F.interpolate(
            predictions, scale_factor=self.common_stride, mode="bilinear", align_corners=False
        )
        loss = self.loss(predictions, targets)
        losses = {"loss_sem_seg": loss * self.loss_weight}
        return losses
