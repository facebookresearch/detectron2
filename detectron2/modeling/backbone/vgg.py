import numpy as np
import fvcore.nn.weight_init as weight_init
import torch.nn.functional as F
from torch import nn

from detectron2.layers import CNNBlockBase, Conv2d, ShapeSpec, get_norm

from .backbone import Backbone
from .build import BACKBONE_REGISTRY

cfgs = {
    11: [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    13: [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    16: [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
    19: [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        512,
        "M",
    ],
}


class VGGBlock(CNNBlockBase):
    def __init__(self, in_channels, channel_cfg, norm="BN", pool=True):
        super().__init__(in_channels, channel_cfg[-1], 2)

        self.convs = []
        self.pool = pool

        for i, out_channels in enumerate(channel_cfg):
            name = "conv" + str(i + 1)
            conv = Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
                norm=get_norm(norm, out_channels),
            )
            in_channels = out_channels
            setattr(self, name, conv)
            self.convs.append(conv)

        if self.pool:
            self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        for layer in self.convs:
            if layer is not None:
                weight_init.c2_msra_fill(layer)

    def forward(self, x):
        for conv in self.convs:
            out = conv(x)
            out = F.relu_(out)
            x = out
        if self.pool:
            out = self.maxpool(out)
        return out


class VGG(Backbone):
    def __init__(self, stages, num_classes=None, out_features=None):
        """
        """
        super().__init__()
        self.num_classes = num_classes

        current_stride = 1
        self._out_feature_strides = {}
        self._out_feature_channels = {}

        self.stages_and_names = []
        for i, block in enumerate(stages):

            name = "vgg_block" + str(i + 1)
            stage = nn.Sequential(block)

            self.add_module(name, stage)
            self.stages_and_names.append((stage, name))

            self._out_feature_strides[name] = current_stride = int(
                current_stride * np.prod([block.stride])
            )
            self._out_feature_channels[name] = block.convs[-1].out_channels

        if num_classes is not None:
            self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
            self.classifier = nn.Sequential(
                nn.Linear(512 * 7 * 7, 4096),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(4096, num_classes),
            )

            for m in self.classifier.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0.01)
                    # nn.init.constant_(m.bias, 0)
                    name = "classifier"

        if out_features is None:
            out_features = [name]
        self._out_features = out_features
        assert len(self._out_features)
        children = [x[0] for x in self.named_children()]
        for out_feature in self._out_features:
            assert out_feature in children, "Available children: {}".format(", ".join(children))

    def forward(self, x):
        outputs = {}
        for stage, name in self.stages_and_names:
            x = stage(x)
            if name in self._out_features:
                outputs[name] = x
        if self.num_classes is not None:
            x = self.avgpool(x)
            x = self.classifier(x)
            if "classifer" in self._out_features:
                outputs["classifer"] = x
        return outputs

    def output_shape(self):
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name], stride=self._out_feature_strides[name]
            )
            for name in self._out_features
        }

    def freeze(self, freeze_at=0):
        for idx, (stage, _) in enumerate(self.stages_and_names, start=1):
            if freeze_at >= idx:
                for block in stage.children():
                    block.freeze()
        return self

    @staticmethod
    def make_stage(block_class, in_channels, channel_cfg, **kwargs):
        assert "stride" not in kwargs, "Stride of blocks in make_stage cannot be changed."
        blocks = block_class(in_channels=in_channels, channel_cfg=channel_cfg, **kwargs)
        return blocks


@BACKBONE_REGISTRY.register()
def build_vgg_backbone(cfg, input_shape):
    # fmt: off
    depth               = cfg.MODEL.VGG.DEPTH
    freeze_at           = cfg.MODEL.BACKBONE.FREEZE_AT
    norm                = cfg.MODEL.VGG.NORM
    out_features        = cfg.MODEL.VGG.OUT_FEATURES
    in_channels         = input_shape.channels
    # fmt: on

    stages = []
    out_stage_idx = [
        {"vgg_block1": 1, "vgg_block2": 2, "vgg_block3": 3, "vgg_block4": 4, "vgg_block5": 5}[f]
        for f in out_features
    ]
    max_stage_idx = max(out_stage_idx)
    stage_inds = [i for i, x in enumerate(cfgs[depth]) if x == "M"]
    ind = 0

    for idx, stage_idx in enumerate(range(1, max_stage_idx + 1)):

        # No maxpooling in the last block
        if stage_idx == 5:
            pool = False
        else:
            pool = True

        stage_kargs = {
            "block_class": VGGBlock,
            "in_channels": in_channels,
            "channel_cfg": cfgs[depth][ind : stage_inds[idx]],
            "norm": norm,
            "pool": pool,
        }

        blocks = VGG.make_stage(**stage_kargs)
        out_channels = cfgs[depth][ind : stage_inds[idx]][-1]
        in_channels = out_channels
        ind = stage_inds[idx] + 1
        stages.append(blocks)
    return VGG(stages, out_features=out_features).freeze(freeze_at)
