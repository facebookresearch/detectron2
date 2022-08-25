# -*- coding: utf-8 -*-

from detectron2.config import LazyCall as L
from detectron2.layers import ShapeSpec
from detectron2.modeling.meta_arch import RetinaNet
from detectron2.modeling.anchor_generator import DefaultAnchorGenerator
from detectron2.modeling.backbone.fpn import LastLevelP6P7
from detectron2.modeling.backbone import BasicStem, FPN, ResNet
from detectron2.modeling.box_regression import Box2BoxTransform
from detectron2.modeling.matcher import Matcher
from detectron2.modeling.meta_arch.retinanet import RetinaNetHead

from ..data.constants import constants

model = L(RetinaNet)(
    backbone=L(FPN)(
        bottom_up=L(ResNet)(
            stem=L(BasicStem)(in_channels=3, out_channels=64, norm="FrozenBN"),
            stages=L(ResNet.make_default_stages)(
                depth=50,
                stride_in_1x1=True,
                norm="FrozenBN",
            ),
            out_features=["res3", "res4", "res5"],
        ),
        in_features=["res3", "res4", "res5"],
        out_channels=256,
        top_block=L(LastLevelP6P7)(in_channels=2048, out_channels="${..out_channels}"),
    ),
    head=L(RetinaNetHead)(
        # Shape for each input feature map
        input_shape=[ShapeSpec(channels=256)] * 5,
        num_classes="${..num_classes}",
        conv_dims=[256, 256, 256, 256],
        prior_prob=0.01,
        num_anchors=9,
    ),
    anchor_generator=L(DefaultAnchorGenerator)(
        sizes=[[x, x * 2 ** (1.0 / 3), x * 2 ** (2.0 / 3)] for x in [32, 64, 128, 256, 512]],
        aspect_ratios=[0.5, 1.0, 2.0],
        strides=[8, 16, 32, 64, 128],
        offset=0.0,
    ),
    box2box_transform=L(Box2BoxTransform)(weights=[1.0, 1.0, 1.0, 1.0]),
    anchor_matcher=L(Matcher)(
        thresholds=[0.4, 0.5], labels=[0, -1, 1], allow_low_quality_matches=True
    ),
    num_classes=80,
    head_in_features=["p3", "p4", "p5", "p6", "p7"],
    focal_loss_alpha=0.25,
    focal_loss_gamma=2.0,
    pixel_mean=constants.imagenet_bgr256_mean,
    pixel_std=constants.imagenet_bgr256_std,
    input_format="BGR",
)
