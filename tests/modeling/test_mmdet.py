# Copyright (c) Facebook, Inc. and its affiliates.
import unittest

from detectron2.layers import ShapeSpec
from detectron2.modeling.mmdet_wrapper import MMDetBackbone, MMDetDetector

try:
    import mmdet.models  # noqa

    HAS_MMDET = True
except ImportError:
    HAS_MMDET = False


@unittest.skipIf(not HAS_MMDET, "mmdet not available")
class TestMMDetWrapper(unittest.TestCase):
    def test_backbone(self):
        MMDetBackbone(
            backbone=dict(
                type="DetectoRS_ResNet",
                conv_cfg=dict(type="ConvAWS"),
                sac=dict(type="SAC", use_deform=True),
                stage_with_sac=(False, True, True, True),
                depth=50,
                num_stages=4,
                out_indices=(0, 1, 2, 3),
                frozen_stages=1,
                norm_cfg=dict(type="BN", requires_grad=True),
                norm_eval=True,
                style="pytorch",
            ),
            neck=dict(
                type="FPN",
                in_channels=[256, 512, 1024, 2048],
                out_channels=256,
                num_outs=5,
            ),
            # skip pretrained model for tests
            # pretrained_backbone="torchvision://resnet50",
            output_shapes=[ShapeSpec(channels=256, stride=s) for s in [4, 8, 16, 32, 64]],
            output_names=["p2", "p3", "p4", "p5", "p6"],
        )

    def test_detector(self):
        # a basic R50 Mask R-CNN
        MMDetDetector(
            detector=dict(
                type="MaskRCNN",
                backbone=dict(
                    type="ResNet",
                    depth=50,
                    num_stages=4,
                    out_indices=(0, 1, 2, 3),
                    frozen_stages=1,
                    norm_cfg=dict(type="BN", requires_grad=True),
                    norm_eval=True,
                    style="pytorch",
                    # skip pretrained model for tests
                    # init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50'))
                ),
                neck=dict(
                    type="FPN", in_channels=[256, 512, 1024, 2048], out_channels=256, num_outs=5
                ),
                rpn_head=dict(
                    type="RPNHead",
                    in_channels=256,
                    feat_channels=256,
                    anchor_generator=dict(
                        type="AnchorGenerator",
                        scales=[8],
                        ratios=[0.5, 1.0, 2.0],
                        strides=[4, 8, 16, 32, 64],
                    ),
                    bbox_coder=dict(
                        type="DeltaXYWHBBoxCoder",
                        target_means=[0.0, 0.0, 0.0, 0.0],
                        target_stds=[1.0, 1.0, 1.0, 1.0],
                    ),
                    loss_cls=dict(type="CrossEntropyLoss", use_sigmoid=True, loss_weight=1.0),
                    loss_bbox=dict(type="L1Loss", loss_weight=1.0),
                ),
                roi_head=dict(
                    type="StandardRoIHead",
                    bbox_roi_extractor=dict(
                        type="SingleRoIExtractor",
                        roi_layer=dict(type="RoIAlign", output_size=7, sampling_ratio=0),
                        out_channels=256,
                        featmap_strides=[4, 8, 16, 32],
                    ),
                    bbox_head=dict(
                        type="Shared2FCBBoxHead",
                        in_channels=256,
                        fc_out_channels=1024,
                        roi_feat_size=7,
                        num_classes=80,
                        bbox_coder=dict(
                            type="DeltaXYWHBBoxCoder",
                            target_means=[0.0, 0.0, 0.0, 0.0],
                            target_stds=[0.1, 0.1, 0.2, 0.2],
                        ),
                        reg_class_agnostic=False,
                        loss_cls=dict(type="CrossEntropyLoss", use_sigmoid=False, loss_weight=1.0),
                        loss_bbox=dict(type="L1Loss", loss_weight=1.0),
                    ),
                    mask_roi_extractor=dict(
                        type="SingleRoIExtractor",
                        roi_layer=dict(type="RoIAlign", output_size=14, sampling_ratio=0),
                        out_channels=256,
                        featmap_strides=[4, 8, 16, 32],
                    ),
                    mask_head=dict(
                        type="FCNMaskHead",
                        num_convs=4,
                        in_channels=256,
                        conv_out_channels=256,
                        num_classes=80,
                        loss_mask=dict(type="CrossEntropyLoss", use_mask=True, loss_weight=1.0),
                    ),
                ),
                # model training and testing settings
                train_cfg=dict(
                    rpn=dict(
                        assigner=dict(
                            type="MaxIoUAssigner",
                            pos_iou_thr=0.7,
                            neg_iou_thr=0.3,
                            min_pos_iou=0.3,
                            match_low_quality=True,
                            ignore_iof_thr=-1,
                        ),
                        sampler=dict(
                            type="RandomSampler",
                            num=256,
                            pos_fraction=0.5,
                            neg_pos_ub=-1,
                            add_gt_as_proposals=False,
                        ),
                        allowed_border=-1,
                        pos_weight=-1,
                        debug=False,
                    ),
                    rpn_proposal=dict(
                        nms_pre=2000,
                        max_per_img=1000,
                        nms=dict(type="nms", iou_threshold=0.7),
                        min_bbox_size=0,
                    ),
                    rcnn=dict(
                        assigner=dict(
                            type="MaxIoUAssigner",
                            pos_iou_thr=0.5,
                            neg_iou_thr=0.5,
                            min_pos_iou=0.5,
                            match_low_quality=True,
                            ignore_iof_thr=-1,
                        ),
                        sampler=dict(
                            type="RandomSampler",
                            num=512,
                            pos_fraction=0.25,
                            neg_pos_ub=-1,
                            add_gt_as_proposals=True,
                        ),
                        mask_size=28,
                        pos_weight=-1,
                        debug=False,
                    ),
                ),
                test_cfg=dict(
                    rpn=dict(
                        nms_pre=1000,
                        max_per_img=1000,
                        nms=dict(type="nms", iou_threshold=0.7),
                        min_bbox_size=0,
                    ),
                    rcnn=dict(
                        score_thr=0.05,
                        nms=dict(type="nms", iou_threshold=0.5),
                        max_per_img=100,
                        mask_thr_binary=0.5,
                    ),
                ),
            ),
            pixel_mean=[1, 2, 3],
            pixel_std=[1, 2, 3],
        )
