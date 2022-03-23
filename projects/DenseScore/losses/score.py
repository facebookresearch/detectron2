import torch
from torch.nn import functional as F

from detectron2.config import CfgNode
from detectron2.structures import Instances

import logging

from .registry import DENSEPOSE_LOSS_REGISTRY
from .utils import (
    LossDict,
)


@DENSEPOSE_LOSS_REGISTRY.register()
class ScoringLoss:
    """'
    UV Scoring loss for char-based training. 

    Estimated values are tensors:
     * U loss, tensor of shape [N,]
     * V loss, tensor of shape [N,]
    Where N is the number of detections

    The losses are:
    * regression (l2 loss) loss for U and V loss
    """
    
    def __init__(self, cfg: CfgNode):
        """
        Initialize UV loss regression from configuration options

        Args:
            cfg (CfgNode): configuration options
        """
        self.uv_weights = cfg.MODEL.ROI_SCORING_HEAD.UV_WEIGHTS

    def __call__(
        self, scoring_predictions, scoring_gt
    ) -> LossDict:
        """
        Produce chart-based DensePose losses
        """
        if len(scoring_gt[0]) == 0:
            return scoring_predictions * 0
        
        if len(scoring_gt) == 2:
            return {
                "scoring_u_loss": F.mse_loss(scoring_predictions[0], scoring_gt[0], reduction='sum') * self.uv_weights * 0.5, 
                "scoring_v_loss": F.mse_loss(scoring_predictions[1], scoring_gt[1], reduction='sum') * self.uv_weights * 0.5,
            }
        elif len(scoring_gt) == 1:
            if len(scoring_gt[0]) == 0:
                return {
                    "gps_loss": scoring_predictions[0].sum() * 0
                }
            else:
                return {
                    # "gps_loss":  F.mse_loss(scoring_predictions[0], scoring_gt[0], reduction='mean') * self.uv_weights
                    "gps_loss":  F.mse_loss(scoring_predictions[0], scoring_gt[0], reduction='mean') * self.uv_weights
                }


@DENSEPOSE_LOSS_REGISTRY.register()
class IoULoss:
    """'
    UV Scoring loss for char-based training. 

    Estimated values are tensors:
     * U loss, tensor of shape [N,]
     * V loss, tensor of shape [N,]
    Where N is the number of detections

    The losses are:
    * regression (l2 loss) loss for U and V loss
    """
    
    def __init__(self, cfg: CfgNode):
        """
        Initialize UV loss regression from configuration options

        Args:
            cfg (CfgNode): configuration options
        """
        self.uv_weights = cfg.MODEL.ROI_IOU_HEAD.IOU_WEIGHTS

    def __call__(
        self, iou_predictions, iou_gt
    ) -> LossDict:
        """
        Produce chart-based DensePose losses
        """
        if len(iou_gt[0]) == 0:
            return iou_predictions * 0
        
        if len(iou_gt) == 2:
            return {
                "scoring_u_loss": F.mse_loss(iou_predictions[0], iou_gt[0], reduction='sum') * self.uv_weights * 0.5, 
                "scoring_v_loss": F.mse_loss(iou_predictions[1], iou_gt[1], reduction='sum') * self.uv_weights * 0.5,
            }
        elif len(iou_gt) == 1:
            if len(iou_gt[0]) == 0:
                return {
                    "iou_loss": iou_predictions[0].sum() * 0
                }
            else:
                return {
                    "iou_loss":  F.mse_loss(iou_predictions[0], iou_gt[0], reduction='mean') * self.uv_weights
                }