# Copyright (c) Facebook, Inc. and its affiliates.

import torch
from torch import nn
from torch.nn import functional as F

from detectron2.config import CfgNode
from detectron2.layers import Conv2d
from detectron2.layers import get_norm
import fvcore.nn.weight_init as weight_init

from ..utils import initialize_module_params
from .registry import ROI_DENSEPOSE_HEAD_REGISTRY
from ..confidence import DensePoseConfidenceModelConfig, DensePoseUVConfidenceType



@ROI_DENSEPOSE_HEAD_REGISTRY.register()
class ScoringConvHeads(nn.Module):
    """
    Convolutional Scoring head.
    """

    def __init__(self, cfg: CfgNode, input_channels: int, channel=4):
        super(ScoringConvHeads, self).__init__()

        self.input_channels     = input_channels
        # default: 'DensePoseChartWithConfidenceLoss'
        self.confidence_model_cfg       = DensePoseConfidenceModelConfig.from_cfg(cfg)
        self.num_patches        = cfg.MODEL.ROI_DENSEPOSE_HEAD.NUM_PATCHES + 1
        self.n_segm_chan        = cfg.MODEL.ROI_DENSEPOSE_HEAD.NUM_COARSE_SEGM_CHANNELS
        self.channel = channel

        if self.confidence_model_cfg.uv_confidence.enabled:

            # build one branch to predict uv loss
            if self.confidence_model_cfg.uv_confidence.type == DensePoseUVConfidenceType.IID_ISO:
                self.uv_head = ScoringConvHead(cfg, input_channels + self.num_patches * 4)
            elif self.confidence_model_cfg.uv_confidence.type == DensePoseUVConfidenceType.INDEP_ANISO:
                self.uv_head = ScoringConvHead(cfg, input_channels + self.num_patches * 6)
            self.uv_output = ScoringOutput(cfg, self.uv_head.n_out_channels)
        else:
            if self.channel == 4:
                # pred + uv + segm
                # self.gps_head = ScoringConvHead(cfg, input_channels + self.num_patches * 3 + self.n_segm_chan)
                # u, v pred + coarse_segm
                self.head = ScoringConvHead(cfg, input_channels + self.num_patches * 3 + self.n_segm_chan)
                
                # head + pool
                # self.gps_head = ScoringConvHead(cfg, input_channels)
                self.output = ScoringOutput(cfg, self.head.n_out_channels)
            elif self.channel == 2:
                self.head = ScoringConvHead(cfg, input_channels + self.num_patches + self.n_segm_chan)
                self.output = ScoringOutput(cfg, self.head.n_out_channels)
    
    def forward(self, features: torch.Tensor, uv_prediction, bbox_indices=None):
        """
        Args:
            features:       [N, C, w, h]
            uv_prediction:  DensePoseCharPredictorOutput or 
                            decorate_predictor_output_class_with_confidences
            bbox_indices:   Indices of bbox whose uv loss need to be regression
        """
        # extract features which have dp annotation
        if bbox_indices is not None:
            features = features[bbox_indices]
            uv_prediction = uv_prediction[bbox_indices]
        result = []

        if self.confidence_model_cfg.uv_confidence.enabled:

            prediction = torch.cat((uv_prediction.fine_segm, uv_prediction.u, uv_prediction.v), 1)

            if self.confidence_model_cfg.uv_confidence.type == DensePoseUVConfidenceType.IID_ISO:
                prediction = torch.cat((prediction, uv_prediction.sigma_2), 1)
            elif self.confidence_model_cfg.uv_confidence.type == DensePoseUVConfidenceType.INDEP_ANISO:
                prediction = torch.cat((prediction, uv_prediction.sigma_2, uv_prediction.kappa_u,
                                        uv_prediction.kappa_v), 1)
            
            result.append(self.uv_output(self.uv_head(features, prediction)))

        else:
            if self.channel == 4:
                # u, v pred + coarse_segm
                uv_prediction = torch.cat((uv_prediction.coarse_segm, uv_prediction.fine_segm, uv_prediction.u, uv_prediction.v), 1)
                result.append(self.output(self.head(features, uv_prediction)))

                # head + pool
                # result.append(self.gps_output(self.gps_head(features, uv_prediction)))
            elif self.channel == 2:
                uv_prediction = torch.cat((uv_prediction.coarse_segm, uv_prediction.fine_segm), 1)
                result.append(self.output(self.head(features, uv_prediction)))

        return result

@ROI_DENSEPOSE_HEAD_REGISTRY.register()
class ScoringConvHead(nn.Module):
    """
    convolutional Scoring head.
    """

    def __init__(self, cfg: CfgNode, input_channels: int):
        """
        Initialize Scoring convolutional head

        Args:
            cfg (CfgNode): configuration options
            input_channels (int): number of input channels
        """
        super(ScoringConvHead, self).__init__()

        self.input_channels     = input_channels
        self.n_stacked_convs       = cfg.MODEL.ROI_SCORING_HEAD.NUM_STACKED_CONVS
        conv_dim                = cfg.MODEL.ROI_SCORING_HEAD.CONV_DIM
        mlp_dim                 = cfg.MODEL.ROI_SCORING_HEAD.MLP_DIM
        # use_bn                  = cfg.MODEL.ROI_SCORING_HEAD.USE_BN
        # use_gn                  = cfg.MODEL.ROI_SCORING_HEAD.USE_GN
        self.norm               = cfg.MODEL.ROI_SCORING_HEAD.NORM


        """
        Simpler network
        """
        # self.conv1x1 = Conv2d(
        #     self.input_channels, 
        #     self.input_channels, 
        #     kernel_size=1, 
        #     stride=1,
        #     norm=get_norm(self.norm, self.input_channels),
        #     activation=F.relu)                    

        convx = []
        """
        more complex network
        """
        for _ in range(self.n_stacked_convs):
            layer_stride = 1 if _ < self.n_stacked_convs - 1 else 2
            convx.append(Conv2d(
                self.input_channels, 
                conv_dim, 
                kernel_size=3, 
                stride=layer_stride,
                padding=1,
                norm=get_norm(self.norm, conv_dim),
                activation=F.relu
                )
            )
            self.input_channels = conv_dim

        self.conv = nn.Sequential(*convx)

        self.avgpool = nn.AdaptiveAvgPool2d(1)

        initialize_module_params(self)

        self.fc1 = nn.Linear(self.input_channels, mlp_dim)
        # self.bn1 = nn.BatchNorm1d(mlp_dim)
        self.fc2 = nn.Linear(mlp_dim, mlp_dim)
        # self.bn2 = nn.BatchNorm1d(mlp_dim)
        self.n_out_channels = mlp_dim

        weight_init.c2_xavier_fill(self.fc1)
        weight_init.c2_xavier_fill(self.fc2)

        # nn.init.constant_(self.bn1.weight, 1)
        # nn.init.constant_(self.bn1.bias, 0)
        # nn.init.constant_(self.bn2.weight, 1)
        # nn.init.constant_(self.bn2.bias, 0)

    def forward(self, features: torch.Tensor, uv_prediction: torch.Tensor):
        """
        Apply Scoring head to the input features and prediction

        Args:
            features (tensor): input features
        Result:
            A tensor of Scoring head outputs
        """

        uv_pool = F.max_pool2d(uv_prediction, kernel_size=4, stride=4)
        # u, v pred + coarse_segm
        x = torch.cat((features, uv_pool), 1)
        # x = torch.cat((features, uv_prediction), 1)

        # x = self.conv1x1(x)
        x = self.conv(x)
        x = self.avgpool(x)
        # x = F.relu(self.bn1(self.fc1(x.view(x.size(0), -1))))
        # x = F.relu(self.bn2(self.fc2(x)))
        x = F.relu(self.fc1(x.view(x.size(0), -1)))
        x = F.relu(self.fc2(x))

        return x


@ROI_DENSEPOSE_HEAD_REGISTRY.register()
class ScoringOutput(nn.Module):
    def __init__(self, cfg: CfgNode, input_channels: int):
        super(ScoringOutput, self).__init__()
        num_classes = 1

        self.gps = nn.Linear(input_channels, num_classes)

        weight_init.c2_xavier_fill(self.gps)

    def forward(self, features):
        return self.gps(features)