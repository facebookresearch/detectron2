"""
MIT License
Copyright (c) 2019 Microsoft
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from detectron2.layers import ShapeSpec
from detectron2.modeling.backbone import BACKBONE_REGISTRY
from detectron2.modeling.backbone.backbone import Backbone

from .hrnet import build_pose_hrnet_backbone


class HRFPN(Backbone):
    """ HRFPN (High Resolution Feature Pyramids)
    Transforms outputs of HRNet backbone so they are suitable for the ROI_heads
    arXiv: https://arxiv.org/abs/1904.04514
    Adapted from https://github.com/open-mmlab/mmdetection/blob/master/mmdet/models/necks/hrfpn.py
    Args:
        bottom_up: (list) output of HRNet
        in_features (list): names of the input features (output of HRNet)
        in_channels (list): number of channels for each branch
        out_channels (int): output channels of feature pyramids
        n_out_features (int): number of output stages
        pooling (str): pooling for generating feature pyramids (from {MAX, AVG})
        share_conv (bool): Have one conv per output, or share one with all the outputs
    """

    def __init__(
        self,
        bottom_up,
        in_features,
        n_out_features,
        in_channels,
        out_channels,
        pooling="AVG",
        share_conv=False,
    ):
        super(HRFPN, self).__init__()
        assert isinstance(in_channels, list)
        self.bottom_up = bottom_up
        self.in_features = in_features
        self.n_out_features = n_out_features
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.share_conv = share_conv

        if self.share_conv:
            self.fpn_conv = nn.Conv2d(
                in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1
            )
        else:
            self.fpn_conv = nn.ModuleList()
            for _ in range(self.n_out_features):
                self.fpn_conv.append(
                    nn.Conv2d(
                        in_channels=out_channels,
                        out_channels=out_channels,
                        kernel_size=3,
                        padding=1,
                    )
                )

        # Custom change: Replaces a simple bilinear interpolation
        self.interp_conv = nn.ModuleList()
        for i in range(len(self.in_features)):
            self.interp_conv.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        in_channels=in_channels[i],
                        out_channels=in_channels[i],
                        kernel_size=4,
                        stride=2 ** i,
                        padding=0,
                        output_padding=0,
                        bias=False,
                    ),
                    nn.BatchNorm2d(in_channels[i], momentum=0.1),
                    nn.ReLU(inplace=True),
                )
            )

        # Custom change: Replaces a couple (reduction conv + pooling) by one conv
        self.reduction_pooling_conv = nn.ModuleList()
        for i in range(self.n_out_features):
            self.reduction_pooling_conv.append(
                nn.Sequential(
                    nn.Conv2d(sum(in_channels), out_channels, kernel_size=2 ** i, stride=2 ** i),
                    nn.BatchNorm2d(out_channels, momentum=0.1),
                    nn.ReLU(inplace=True),
                )
            )

        if pooling == "MAX":
            self.pooling = F.max_pool2d
        else:
            self.pooling = F.avg_pool2d

        self._out_features = []
        self._out_feature_channels = {}
        self._out_feature_strides = {}

        for i in range(self.n_out_features):
            self._out_features.append("p%d" % (i + 1))
            self._out_feature_channels.update({self._out_features[-1]: self.out_channels})
            self._out_feature_strides.update({self._out_features[-1]: 2 ** (i + 2)})

    # default init_weights for conv(msra) and norm in ConvModule
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=1)
                nn.init.constant_(m.bias, 0)

    def forward(self, inputs):
        bottom_up_features = self.bottom_up(inputs)
        assert len(bottom_up_features) == len(self.in_features)
        inputs = [bottom_up_features[f] for f in self.in_features]

        outs = []
        for i in range(len(inputs)):
            outs.append(self.interp_conv[i](inputs[i]))
        shape_2 = min(o.shape[2] for o in outs)
        shape_3 = min(o.shape[3] for o in outs)
        out = torch.cat([o[:, :, :shape_2, :shape_3] for o in outs], dim=1)
        outs = []
        for i in range(self.n_out_features):
            outs.append(self.reduction_pooling_conv[i](out))
        for i in range(len(outs)):  # Make shapes consistent
            outs[-1 - i] = outs[-1 - i][
                :, :, : outs[-1].shape[2] * 2 ** i, : outs[-1].shape[3] * 2 ** i
            ]
        outputs = []
        for i in range(len(outs)):
            if self.share_conv:
                outputs.append(self.fpn_conv(outs[i]))
            else:
                outputs.append(self.fpn_conv[i](outs[i]))

        assert len(self._out_features) == len(outputs)
        return dict(zip(self._out_features, outputs))


@BACKBONE_REGISTRY.register()
def build_hrfpn_backbone(cfg, input_shape: ShapeSpec):

    in_channels = cfg.MODEL.HRNET.STAGE4.NUM_CHANNELS
    in_features = ["p%d" % (i + 1) for i in range(cfg.MODEL.HRNET.STAGE4.NUM_BRANCHES)]
    n_out_features = len(cfg.MODEL.ROI_HEADS.IN_FEATURES)
    out_channels = cfg.MODEL.HRNET.HRFPN.OUT_CHANNELS
    hrnet = build_pose_hrnet_backbone(cfg, input_shape)
    hrfpn = HRFPN(
        hrnet,
        in_features,
        n_out_features,
        in_channels,
        out_channels,
        pooling="AVG",
        share_conv=False,
    )

    return hrfpn
