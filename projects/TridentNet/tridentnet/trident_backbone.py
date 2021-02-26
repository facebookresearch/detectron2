# Copyright (c) Facebook, Inc. and its affiliates.
import fvcore.nn.weight_init as weight_init
import torch
import torch.nn.functional as F

from detectron2.layers import Conv2d, FrozenBatchNorm2d, get_norm
from detectron2.modeling import BACKBONE_REGISTRY, ResNet, ResNetBlockBase
from detectron2.modeling.backbone.resnet import BasicStem, BottleneckBlock, DeformBottleneckBlock

from .trident_conv import TridentConv

__all__ = ["TridentBottleneckBlock", "make_trident_stage", "build_trident_resnet_backbone"]


class TridentBottleneckBlock(ResNetBlockBase):
    def __init__(
        self,
        in_channels,
        out_channels,
        *,
        bottleneck_channels,
        stride=1,
        num_groups=1,
        norm="BN",
        stride_in_1x1=False,
        num_branch=3,
        dilations=(1, 2, 3),
        concat_output=False,
        test_branch_idx=-1,
    ):
        """
        Args:
            num_branch (int): the number of branches in TridentNet.
            dilations (tuple): the dilations of multiple branches in TridentNet.
            concat_output (bool): if concatenate outputs of multiple branches in TridentNet.
                Use 'True' for the last trident block.
        """
        super().__init__(in_channels, out_channels, stride)

        assert num_branch == len(dilations)

        self.num_branch = num_branch
        self.concat_output = concat_output
        self.test_branch_idx = test_branch_idx

        if in_channels != out_channels:
            self.shortcut = Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=stride,
                bias=False,
                norm=get_norm(norm, out_channels),
            )
        else:
            self.shortcut = None

        stride_1x1, stride_3x3 = (stride, 1) if stride_in_1x1 else (1, stride)

        self.conv1 = Conv2d(
            in_channels,
            bottleneck_channels,
            kernel_size=1,
            stride=stride_1x1,
            bias=False,
            norm=get_norm(norm, bottleneck_channels),
        )

        self.conv2 = TridentConv(
            bottleneck_channels,
            bottleneck_channels,
            kernel_size=3,
            stride=stride_3x3,
            paddings=dilations,
            bias=False,
            groups=num_groups,
            dilations=dilations,
            num_branch=num_branch,
            test_branch_idx=test_branch_idx,
            norm=get_norm(norm, bottleneck_channels),
        )

        self.conv3 = Conv2d(
            bottleneck_channels,
            out_channels,
            kernel_size=1,
            bias=False,
            norm=get_norm(norm, out_channels),
        )

        for layer in [self.conv1, self.conv2, self.conv3, self.shortcut]:
            if layer is not None:  # shortcut can be None
                weight_init.c2_msra_fill(layer)

    def forward(self, x):
        num_branch = self.num_branch if self.training or self.test_branch_idx == -1 else 1
        if not isinstance(x, list):
            x = [x] * num_branch
        out = [self.conv1(b) for b in x]
        out = [F.relu_(b) for b in out]

        out = self.conv2(out)
        out = [F.relu_(b) for b in out]

        out = [self.conv3(b) for b in out]

        if self.shortcut is not None:
            shortcut = [self.shortcut(b) for b in x]
        else:
            shortcut = x

        out = [out_b + shortcut_b for out_b, shortcut_b in zip(out, shortcut)]
        out = [F.relu_(b) for b in out]
        if self.concat_output:
            out = torch.cat(out)
        return out


def make_trident_stage(block_class, num_blocks, **kwargs):
    """
    Create a resnet stage by creating many blocks for TridentNet.
    """
    concat_output = [False] * (num_blocks - 1) + [True]
    kwargs["concat_output_per_block"] = concat_output
    return ResNet.make_stage(block_class, num_blocks, **kwargs)


@BACKBONE_REGISTRY.register()
def build_trident_resnet_backbone(cfg, input_shape):
    """
    Create a ResNet instance from config for TridentNet.

    Returns:
        ResNet: a :class:`ResNet` instance.
    """
    # need registration of new blocks/stems?
    norm = cfg.MODEL.RESNETS.NORM
    stem = BasicStem(
        in_channels=input_shape.channels,
        out_channels=cfg.MODEL.RESNETS.STEM_OUT_CHANNELS,
        norm=norm,
    )
    freeze_at = cfg.MODEL.BACKBONE.FREEZE_AT

    if freeze_at >= 1:
        for p in stem.parameters():
            p.requires_grad = False
        stem = FrozenBatchNorm2d.convert_frozen_batchnorm(stem)

    # fmt: off
    out_features         = cfg.MODEL.RESNETS.OUT_FEATURES
    depth                = cfg.MODEL.RESNETS.DEPTH
    num_groups           = cfg.MODEL.RESNETS.NUM_GROUPS
    width_per_group      = cfg.MODEL.RESNETS.WIDTH_PER_GROUP
    bottleneck_channels  = num_groups * width_per_group
    in_channels          = cfg.MODEL.RESNETS.STEM_OUT_CHANNELS
    out_channels         = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS
    stride_in_1x1        = cfg.MODEL.RESNETS.STRIDE_IN_1X1
    res5_dilation        = cfg.MODEL.RESNETS.RES5_DILATION
    deform_on_per_stage  = cfg.MODEL.RESNETS.DEFORM_ON_PER_STAGE
    deform_modulated     = cfg.MODEL.RESNETS.DEFORM_MODULATED
    deform_num_groups    = cfg.MODEL.RESNETS.DEFORM_NUM_GROUPS
    num_branch           = cfg.MODEL.TRIDENT.NUM_BRANCH
    branch_dilations     = cfg.MODEL.TRIDENT.BRANCH_DILATIONS
    trident_stage        = cfg.MODEL.TRIDENT.TRIDENT_STAGE
    test_branch_idx      = cfg.MODEL.TRIDENT.TEST_BRANCH_IDX
    # fmt: on
    assert res5_dilation in {1, 2}, "res5_dilation cannot be {}.".format(res5_dilation)

    num_blocks_per_stage = {50: [3, 4, 6, 3], 101: [3, 4, 23, 3], 152: [3, 8, 36, 3]}[depth]

    stages = []

    res_stage_idx = {"res2": 2, "res3": 3, "res4": 4, "res5": 5}
    out_stage_idx = [res_stage_idx[f] for f in out_features]
    trident_stage_idx = res_stage_idx[trident_stage]
    max_stage_idx = max(out_stage_idx)
    for idx, stage_idx in enumerate(range(2, max_stage_idx + 1)):
        dilation = res5_dilation if stage_idx == 5 else 1
        first_stride = 1 if idx == 0 or (stage_idx == 5 and dilation == 2) else 2
        stage_kargs = {
            "num_blocks": num_blocks_per_stage[idx],
            "stride_per_block": [first_stride] + [1] * (num_blocks_per_stage[idx] - 1),
            "in_channels": in_channels,
            "bottleneck_channels": bottleneck_channels,
            "out_channels": out_channels,
            "num_groups": num_groups,
            "norm": norm,
            "stride_in_1x1": stride_in_1x1,
            "dilation": dilation,
        }
        if stage_idx == trident_stage_idx:
            assert not deform_on_per_stage[
                idx
            ], "Not support deformable conv in Trident blocks yet."
            stage_kargs["block_class"] = TridentBottleneckBlock
            stage_kargs["num_branch"] = num_branch
            stage_kargs["dilations"] = branch_dilations
            stage_kargs["test_branch_idx"] = test_branch_idx
            stage_kargs.pop("dilation")
        elif deform_on_per_stage[idx]:
            stage_kargs["block_class"] = DeformBottleneckBlock
            stage_kargs["deform_modulated"] = deform_modulated
            stage_kargs["deform_num_groups"] = deform_num_groups
        else:
            stage_kargs["block_class"] = BottleneckBlock
        blocks = (
            make_trident_stage(**stage_kargs)
            if stage_idx == trident_stage_idx
            else ResNet.make_stage(**stage_kargs)
        )
        in_channels = out_channels
        out_channels *= 2
        bottleneck_channels *= 2

        if freeze_at >= stage_idx:
            for block in blocks:
                block.freeze()
        stages.append(blocks)
    return ResNet(stem, stages, out_features=out_features)
