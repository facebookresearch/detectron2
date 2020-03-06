# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import logging
from typing import Dict
import torch

from detectron2.layers import ShapeSpec

from ..anchor_generator import RotatedAnchorGenerator
from ..box_regression import Box2BoxTransformRotated
from .build import PROPOSAL_GENERATOR_REGISTRY
from .rpn import RPN
from .rrpn_outputs import RRPNOutputs, find_top_rrpn_proposals

logger = logging.getLogger(__name__)


@PROPOSAL_GENERATOR_REGISTRY.register()
class RRPN(RPN):
    """
    Rotated RPN subnetwork.
    Please refer to https://arxiv.org/pdf/1703.01086.pdf for the original RRPN paper:
    Ma, J., Shao, W., Ye, H., Wang, L., Wang, H., Zheng, Y., & Xue, X. (2018).
    Arbitrary-oriented scene text detection via rotation proposals.
    IEEE Transactions on Multimedia, 20(11), 3111-3122.
    """

    def __init__(self, cfg, input_shape: Dict[str, ShapeSpec]):
        """
        Rotated RPN, 

        Requirements:
        minimal config options to run RRPN: 
        MODEL:
          ANCHOR_GENERATOR:
            NAME: RotatedAnchorGenerator
            ANGLES: [[-90,-60,-30,0,30,60,90]]
          PROPOSAL_GENERATOR:
            NAME: RRPN
          RPN:
            BBOX_REG_WEIGHTS: (1,1,1,1,1)
          ROI_BOX_HEAD:
            POOLER_TYPE: ROIAlignRotated
            BBOX_REG_WEIGHTS: (10,10,5,5,1)
          ROI_HEADS:
            NAME: RROIHeads

          dataset: you'll need training dataset with bboxes in "XYWHA" format.
        """
        super().__init__(cfg, input_shape)
        # TODO custom DataLoader for rotated bboxes
        # TODO generated dataset with rotation info (+angle)
        # TODO code for visualization of rotated bboxes

        assert isinstance(
            self.anchor_generator, RotatedAnchorGenerator
        ), "RRPN: must set MODEL.ANCHOR_GENERATOR.NAME to 'RotatedAnchorGenerator' but it is {}".format(
            cfg.MODEL.ANCHOR_GENERATOR.NAME
        )
        # Note, to avoid having to specify `BBOX_REG_WEIGHTS: (1,1,1,1,1)` just to have
        # RRPN enabled we provide the default values in correct dim (5, not 4).
        # This is so people don't have to modify their configs on unrelated places.
        weights = cfg.MODEL.RPN.BBOX_REG_WEIGHTS
        if weights == (
            1.0,
            1.0,
            1.0,
            1.0,
        ):  # 4-element, default values -> we can automatically change to correct dim
            weights = (1, 1, 1, 1, 1)  # 5-elem
        assert (
            len(weights) == 5
        ), "RRPN: must provide 5-element tuple weights, but got {}.\
                 Please set cfg.MODEL.RPN.BBOX_REG_WEIGHTS correctly.".format(
            weights
        )
        assert (
            cfg.MODEL.ROI_HEADS.POOLER_TYPE == "ROIAlignRotated"
        ), "RRPN must use MODEL.ROI_HEADS.POOLER_TYPE: 'ROIAlignRotated' "
        assert(cfg.MODEL.ANCHOR_GENERATOR.ANGLES[0] >0), "RRPN: must provide list of angles in \
                cfg.MODEL.ANCHOR_GENERATOR.ANGLES"
        assert(len(cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_WEIGHTS)==5), "RRPN: provide 5-element weights in\
                MODEL.ROI_BOX_HEAD.BBOX_REG_WEIGHTS"
        assert(cfg.MODEL.ROI_HEADS.NAME == "RROIHeads")

        self.box2box_transform = Box2BoxTransformRotated(weights=weights)

    def forward(self, images, features, gt_instances=None):
        # same signature as RPN.forward
        gt_boxes = [x.gt_boxes for x in gt_instances] if gt_instances is not None else None
        del gt_instances
        features = [features[f] for f in self.in_features]
        pred_objectness_logits, pred_anchor_deltas = self.rpn_head(features)
        anchors = self.anchor_generator(features)

        outputs = RRPNOutputs(
            self.box2box_transform,
            self.anchor_matcher,
            self.batch_size_per_image,
            self.positive_fraction,
            images,
            pred_objectness_logits,
            pred_anchor_deltas,
            anchors,
            self.boundary_threshold,
            gt_boxes,
            self.smooth_l1_beta,
        )

        if self.training:
            losses = outputs.losses()
        else:
            losses = {}

        with torch.no_grad():
            # Find the top proposals by applying NMS and removing boxes that
            # are too small. The proposals are treated as fixed for approximate
            # joint training with roi heads. This approach ignores the derivative
            # w.r.t. the proposal boxes’ coordinates that are also network
            # responses, so is approximate.
            proposals = find_top_rrpn_proposals(
                outputs.predict_proposals(),
                outputs.predict_objectness_logits(),
                images,
                self.nms_thresh,
                self.pre_nms_topk[self.training],
                self.post_nms_topk[self.training],
                self.min_box_side_len,
                self.training,
            )

        return proposals, losses
