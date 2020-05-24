# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch
from torch import nn
from torch.autograd.function import Function
import numpy as np
from typing import Dict

from detectron2.layers import ShapeSpec
from detectron2.structures import Boxes, Instances, pairwise_iou
from detectron2.utils.events import get_event_storage

from ..backbone.resnet import BottleneckBlock, make_stage
from ..box_regression import Box2BoxTransform
from ..matcher import Matcher
from ..poolers import ROIPooler
from ..proposal_generator.proposal_utils import add_ground_truth_to_proposals
from ..sampling import subsample_labels
from .box_head import build_box_head
from .fast_rcnn import FastRCNNOutputLayers, FastRCNNOutputs, fast_rcnn_inference
from .roi_heads import ROI_HEADS_REGISTRY, StandardROIHeads
from .keypoint_head import build_keypoint_head, keypoint_rcnn_inference, keypoint_rcnn_loss
from .mask_head import build_mask_head, mask_rcnn_inference, mask_rcnn_loss
from .roi_heads import ROIHeads

class _ScaleGradient(Function):
    @staticmethod
    def forward(ctx, input, scale):
        ctx.scale = scale
        return input

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output * ctx.scale, None

@ROI_HEADS_REGISTRY.register()
class AttentionROIHeads(ROIHeads):
    """
    It's "standard" in a sense that there is no ROI transform sharing
    or feature sharing between tasks.
    The cropped rois go to separate branches (boxes and masks) directly.
    This way, it is easier to make separate abstractions for different branches.

    This class is used by most models, such as FPN and C5.
    To implement more models, you can subclass it and implement a different
    :meth:`forward()` or a head.
    """

    def __init__(self, cfg, input_shape):
        super(AttentionROIHeads, self).__init__(cfg, input_shape)
        self._init_global_box_head(cfg)
        self._init_box_head(cfg)
        self._init_mask_head(cfg)
        self._init_keypoint_head(cfg)
        
    def _init_global_box_head(self, cfg):
        
        # fmt: off
        pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pooler_scales     = tuple(1.0 / self.feature_strides[k] for k in self.in_features)
        sampling_ratio    = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler_type       = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE
        self.enhance_size = cfg.MODEL.ATTENTION_ROI_HEAD.CHANNEL_OF_ENHANCED_FEATURE
        self.relu = nn.ReLU(inplace=True)

        
        # fmt: on

        # If StandardROIHeads is applied on multiple feature maps (as in FPN),
        # then we share the same predictors and therefore the channel counts must be the same
        in_channels = [self.feature_channels[f] for f in self.in_features]
        # Check all channel counts are equal
        assert len(set(in_channels)) == 1, in_channels
        in_channels = in_channels[0]

        self.global_box_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )
        # Here we split "box head" and "box predictor", which is mainly due to historical reasons.
        # They are used together so the "box predictor" layers should be part of the "box head".
        # New subclasses of ROIHeads do not need "box predictor"s.
        self.global_box_head = build_box_head(
            cfg, ShapeSpec(channels=in_channels, height=pooler_resolution, width=pooler_resolution)
        )
        assert not cfg.MODEL.ROI_BOX_HEAD.CLS_AGNOSTIC_BBOX_REG,  \
            "ParallelROIHeads only support non class-agnostic regression now!"
        self.global_box_predictor =  FastRCNNOutputLayers(
            self.global_box_head.output_size, self.num_classes, self.cls_agnostic_bbox_reg
        )
        
        self.squeeze = nn.ModuleList()
        self.squeeze.append(
            nn.Conv2d(1024, 1024 // 16,
                       3, stride=2, padding=1, bias=True))
        self.squeeze.append(
            nn.Linear(1024 // 16, self.global_box_predictor.cls_score.weight.shape[1] + 1))

        self.W_G = nn.Linear(self.global_box_predictor.cls_score.weight.shape[1] + 1, self.enhance_size)

    def _init_box_head(self, cfg):
        
        # fmt: off
        pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pooler_scales     = tuple(1.0 / self.feature_strides[k] for k in self.in_features)
        sampling_ratio    = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler_type       = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE
        # fmt: on

        # If StandardROIHeads is applied on multiple feature maps (as in FPN),
        # then we share the same predictors and therefore the channel counts must be the same
        in_channels = [self.feature_channels[f] for f in self.in_features]
        # Check all channel counts are equal
        assert len(set(in_channels)) == 1, in_channels
        in_channels = in_channels[0]

        self.box_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )
        # Here we split "box head" and "box predictor", which is mainly due to historical reasons.
        # They are used together so the "box predictor" layers should be part of the "box head".
        # New subclasses of ROIHeads do not need "box predictor"s.
        self.box_head = build_box_head(
            cfg, ShapeSpec(channels=in_channels, height=pooler_resolution, width=pooler_resolution)
        )
        
        self.box_predictor = FastRCNNOutputLayers(
            self.box_head.output_size + self.enhance_size, self.num_classes, self.cls_agnostic_bbox_reg
        )

    def _init_mask_head(self, cfg):
        # fmt: off
        self.mask_on           = cfg.MODEL.MASK_ON
        if not self.mask_on:
            return
        pooler_resolution = cfg.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION
        pooler_scales     = tuple(1.0 / self.feature_strides[k] for k in self.in_features)
        sampling_ratio    = cfg.MODEL.ROI_MASK_HEAD.POOLER_SAMPLING_RATIO
        pooler_type       = cfg.MODEL.ROI_MASK_HEAD.POOLER_TYPE
        # fmt: on

        in_channels = [self.feature_channels[f] for f in self.in_features][0]

        self.mask_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )
        self.mask_head = build_mask_head(
            cfg, ShapeSpec(channels=in_channels, width=pooler_resolution, height=pooler_resolution)
        )

    def _init_keypoint_head(self, cfg):
        # fmt: off
        self.keypoint_on                         = cfg.MODEL.KEYPOINT_ON
        if not self.keypoint_on:
            return
        pooler_resolution                        = cfg.MODEL.ROI_KEYPOINT_HEAD.POOLER_RESOLUTION
        pooler_scales                            = tuple(1.0 / self.feature_strides[k] for k in self.in_features)  # noqa
        sampling_ratio                           = cfg.MODEL.ROI_KEYPOINT_HEAD.POOLER_SAMPLING_RATIO
        pooler_type                              = cfg.MODEL.ROI_KEYPOINT_HEAD.POOLER_TYPE
        self.normalize_loss_by_visible_keypoints = cfg.MODEL.ROI_KEYPOINT_HEAD.NORMALIZE_LOSS_BY_VISIBLE_KEYPOINTS  # noqa
        self.keypoint_loss_weight                = cfg.MODEL.ROI_KEYPOINT_HEAD.LOSS_WEIGHT
        # fmt: on

        in_channels = [self.feature_channels[f] for f in self.in_features][0]

        self.keypoint_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )
        self.keypoint_head = build_keypoint_head(
            cfg, ShapeSpec(channels=in_channels, width=pooler_resolution, height=pooler_resolution)
        )

    def forward(self, images, features, proposals, targets=None, use_proposals = False):
        """
        See :class:`ROIHeads.forward`.
        """
        
        del images
        if self.training:
            proposals = self.label_and_sample_proposals(proposals, targets)
        del targets

        features_list = [features[f] for f in self.in_features]
#         print(features_list[0].shape) # [2, 1024, 1, 2]
        if self.training:
            losses, enhanced_feature = self._forward_global_box(features_list, proposals)
            losses.update(self._forward_box(features_list, proposals, enhanced_feature))
            # During training the proposals used by the box head are
            # used by the mask, keypoint (and densepose) heads.
            losses.update(self._forward_mask(features_list, proposals))
            losses.update(self._forward_keypoint(features_list, proposals))
            return proposals, losses
        else:
            _, enhanced_feature = self._forward_global_box(features_list, proposals)
            pred_instances = self._forward_box(features_list, proposals, enhanced_feature)
            # During inference cascaded prediction is used: the mask and keypoints heads are only
            # applied to the top scoring box detections.
            pred_instances = self.forward_with_given_boxes(features, pred_instances)
            return pred_instances, {}

    def forward_with_given_boxes(self, features, instances):
        """
        Use the given boxes in `instances` to produce other (non-box) per-ROI outputs.

        This is useful for downstream tasks where a box is known, but need to obtain
        other attributes (outputs of other heads).
        Test-time augmentation also uses this.

        Args:
            features: same as in `forward()`
            instances (list[Instances]): instances to predict other outputs. Expect the keys
                "pred_boxes" and "pred_classes" to exist.

        Returns:
            instances (Instances):
                the same `Instances` object, with extra
                fields such as `pred_masks` or `pred_keypoints`.
        """
        assert not self.training
        assert instances[0].has("pred_boxes") and instances[0].has("pred_classes")
        features = [features[f] for f in self.in_features]

        instances = self._forward_mask(features, instances)
        instances = self._forward_keypoint(features, instances)
        return instances


    def _forward_global_box(self, features, proposals):
        """
        Forward logic of the box prediction branch.

        Args:
            features (list[Tensor]): #level input features for box prediction
            proposals (list[Instances]): the per-image object proposals with
                their matching ground truth.
                Each has fields "proposal_boxes", and "objectness_logits",
                "gt_classes", "gt_boxes".

        Returns:
            In training, a dict of losses.
            In inference, a list of `Instances`, the predicted instances.
        """
#         print(len(features)) # [1] batch size
#         print(features[0].shape) # [2, 1024, 1, 2]
        box_features = self.global_box_pooler(features, [x.proposal_boxes for x in proposals])
#         print(len(box_features)) # [11]
#         print(box_features[0].shape) # [1024, 14, 14]
        box_features = self.global_box_head(box_features)
#         print(len(box_features)) # [11]
#         print(box_features[0].shape) # [1024]
        pred_class_logits, pred_proposal_deltas = self.global_box_predictor(box_features)
#         print(self.box_predictor.cls_score.weight.shape) # 81, 1024
#         print(self.box_predictor.cls_score.bias.shape) # 81
#         print(len(pred_class_logits)) # 11
#         print(pred_class_logits[0].shape) # 81 -> class_num + 1
#         print(len(pred_proposal_deltas)) # 11
#         print(pred_proposal_deltas[0].shape) # 320 -> 4 * num_class

#         tmp_features = torch.stack(features).squeeze(0)
#         for ops in self.cmp_attention:
#             squeeze_ext_feature = ops(tmp_features)
#             print(squeeze_ext_feature.shape)
#             if len(box_features.size()) > 2:
#                 squeeze_ext_feature = squeeze_ext_feature.mean(3).mean(2)
#             else:
#                 squeeze_ext_feature = self.relu(squeeze_ext_feature)
        
        del box_features
        
        # my defined of attention
        # all_possible_classes_logits = [num_class + 1]
        all_possible_classes_logits = pred_class_logits.max(0).values
        all_possible_classes_logits = nn.Softmax(0)(all_possible_classes_logits)

        # global_semantic_features = [num_class + 1, 1025]
        global_semantic_features = torch.cat((self.global_box_predictor.cls_score.weight,
                                                  self.global_box_predictor.cls_score.bias.unsqueeze(1)), 1).detach()
        # img_wise_semantic_pool = [81, 1025]
        img_wise_semantic_pool = global_semantic_features*all_possible_classes_logits.unsqueeze(-1)

#         self.cmp_attention()
#         attention_map = nn.Softmax(1)(torch.mm(base_feat, torch.transpose(global_semantic_pool, 0, 1)))
        tmp_feature = torch.mm(pred_class_logits, img_wise_semantic_pool)

        enhanced_feature = self.W_G(tmp_feature)
        
        outputs = FastRCNNOutputs(
            self.box2box_transform,
            pred_class_logits,
            pred_proposal_deltas,
            proposals,
            self.smooth_l1_beta,
        )
        if self.training:
            return outputs.losses(is_global=True), enhanced_feature
        else:
            pred_instances, _ = outputs.inference(
                self.test_score_thresh, self.test_nms_thresh, self.test_detections_per_img
            )
            return pred_instances, enhanced_feature
        
    def _forward_box(self, features, proposals, enhanced_feature):
        """
        Forward logic of the box prediction branch.

        Args:
            features (list[Tensor]): #level input features for box prediction
            proposals (list[Instances]): the per-image object proposals with
                their matching ground truth.
                Each has fields "proposal_boxes", and "objectness_logits",
                "gt_classes", "gt_boxes".

        Returns:
            In training, a dict of losses.
            In inference, a list of `Instances`, the predicted instances.
        """

#         print(len(features)) # [1] batch size
#         print(features[0].shape) # [2, 1024, 1, 2]


        box_features = self.box_pooler(features, [x.proposal_boxes for x in proposals])
#         print("box_features:",len(box_features)) # [11]
#         print(box_features[0].shape) # [1024, 14, 14]
        box_features = self.box_head(box_features)
        box_features = torch.cat((box_features, enhanced_feature), 1)
#         print("box_features:",len(box_features)) # [11]
#         print(box_features[0].shape) # [1024]
        pred_class_logits, pred_proposal_deltas = self.box_predictor(box_features)
#         print(self.box_predictor.cls_score.weight.shape) # 81, 1024
#         print(self.box_predictor.cls_score.bias.shape) # 81
#         print(len(pred_class_logits)) # 11
#         print(pred_class_logits[0].shape) # 81 -> class_num + 1
#         print(len(pred_proposal_deltas)) # 11
#         print(pred_proposal_deltas[0].shape) # 320 -> 4 * num_class
        del box_features
        
        outputs = FastRCNNOutputs(
            self.box2box_transform,
            pred_class_logits,
            pred_proposal_deltas,
            proposals,
            self.smooth_l1_beta,
        )
        if self.training:
            return outputs.losses()
        else:
            pred_instances, _ = outputs.inference(
                self.test_score_thresh, self.test_nms_thresh, self.test_detections_per_img
            )
            return pred_instances

    def _forward_mask(self, features, instances):
        """
        Forward logic of the mask prediction branch.

        Args:
            features (list[Tensor]): #level input features for mask prediction
            instances (list[Instances]): the per-image instances to train/predict masks.
                In training, they can be the proposals.
                In inference, they can be the predicted boxes.

        Returns:
            In training, a dict of losses.
            In inference, update `instances` with new fields "pred_masks" and return it.
        """
        if not self.mask_on:
            return {} if self.training else instances
        
        if self.training:
            # The loss is only defined on positive proposals.
            proposals, _ = select_foreground_proposals(instances, self.num_classes)
            proposal_boxes = [x.proposal_boxes for x in proposals]
            mask_features = self.mask_pooler(features, proposal_boxes)

            mask_logits = self.mask_head(mask_features)
            return {"loss_mask": mask_rcnn_loss(mask_logits, proposals)}
        else:
            pred_boxes = [x.pred_boxes for x in instances]
            mask_features = self.mask_pooler(features, pred_boxes)
            mask_logits = self.mask_head(mask_features)
            mask_rcnn_inference(mask_logits, instances)
            return instances

    def _forward_keypoint(self, features, instances):
        """
        Forward logic of the keypoint prediction branch.

        Args:
            features (list[Tensor]): #level input features for keypoint prediction
            instances (list[Instances]): the per-image instances to train/predict keypoints.
                In training, they can be the proposals.
                In inference, they can be the predicted boxes.

        Returns:
            In training, a dict of losses.
            In inference, update `instances` with new fields "pred_keypoints" and return it.
        """
        if not self.keypoint_on:
            return {} if self.training else instances

        num_images = len(instances)

        if self.training:
            # The loss is defined on positive proposals with at >=1 visible keypoints.
            proposals, _ = select_foreground_proposals(instances, self.num_classes)
            proposals = select_proposals_with_visible_keypoints(proposals)
            proposal_boxes = [x.proposal_boxes for x in proposals]

            keypoint_features = self.keypoint_pooler(features, proposal_boxes)
            keypoint_logits = self.keypoint_head(keypoint_features)

            normalizer = (
                num_images
                * self.batch_size_per_image
                * self.positive_sample_fraction
                * keypoint_logits.shape[1]
            )
            loss = keypoint_rcnn_loss(
                keypoint_logits,
                proposals,
                normalizer=None if self.normalize_loss_by_visible_keypoints else normalizer,
            )
            return {"loss_keypoint": loss * self.keypoint_loss_weight}
        else:
            pred_boxes = [x.pred_boxes for x in instances]
            keypoint_features = self.keypoint_pooler(features, pred_boxes)
            keypoint_logits = self.keypoint_head(keypoint_features)
            keypoint_rcnn_inference(keypoint_logits, instances)
            return instances

def select_foreground_proposals(proposals, bg_label):
    """
    Given a list of N Instances (for N images), each containing a `gt_classes` field,
    return a list of Instances that contain only instances with `gt_classes != -1 &&
    gt_classes != bg_label`.

    Args:
        proposals (list[Instances]): A list of N Instances, where N is the number of
            images in the batch.
        bg_label: label index of background class.

    Returns:
        list[Instances]: N Instances, each contains only the selected foreground instances.
        list[Tensor]: N boolean vector, correspond to the selection mask of
            each instance. True for selected instances.
    """
    assert isinstance(proposals, (list, tuple))
    assert isinstance(proposals[0], Instances)
    assert proposals[0].has("gt_classes")
    fg_proposals = []
    fg_selection_masks = []
    for proposals_per_image in proposals:
        gt_classes = proposals_per_image.gt_classes
        fg_selection_mask = (gt_classes != -1) & (gt_classes != bg_label)
        fg_idxs = fg_selection_mask.nonzero().squeeze(1)
        fg_proposals.append(proposals_per_image[fg_idxs])
        fg_selection_masks.append(fg_selection_mask)
    return fg_proposals, fg_selection_masks

        