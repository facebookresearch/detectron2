# -*- coding: utf-8 -*-

import logging
from collections.abc import Mapping
from typing import Dict, List, Optional, Union
from omegaconf import DictConfig, OmegaConf
from torch import Tensor, nn

from detectron2.layers import ShapeSpec

from .backbone import Backbone

logger = logging.getLogger(__name__)


def _to_container(cfg):
    """
    mmdet will assert the type of dict/list.
    So convert omegaconf objects to dict/list.
    """
    if isinstance(cfg, DictConfig):
        return OmegaConf.to_container(cfg, resolve=True)
    return cfg


class MMDetBackbone(Backbone):
    """
    Wrapper of mmdetection backbones to use in detectron2.

    mmdet backbones produce list/tuple of tensors, while detectron2 backbones
    produce a dict of tensors.
    """

    def __init__(
        self,
        backbone: Union[nn.Module, Mapping],
        neck: Union[nn.Module, Mapping, None] = None,
        *,
        pretrained_backbone: Optional[str] = None,
        output_shapes: List[ShapeSpec],
        output_names: Optional[List[str]] = None,
    ):
        """
        Args:
            backbone: either a backbone module or a mmdet config dict that defines a
                backbone. The backbone takes a 4D image tensor and returns a
                sequence of tensors.
            neck: either a backbone module or a mmdet config dict that defines a
                neck. The neck takes outputs of backbone and returns a
                sequence of tensors. If None, no neck is used.
            pretrained_backbone: defines the backbone weights that can be loaded by
                mmdet, such as "torchvision://resnet50".
            output_shapes: shape for every output of the backbone (or neck, if given).
                stride and channels are often needed.
            output_names: names for every output of the backbone (or neck, if given).
                By default, will use "out0", "out1", ...
        """
        super().__init__()
        if isinstance(backbone, Mapping):
            from mmdet.models import build_backbone

            backbone = build_backbone(_to_container(backbone))
        self.backbone = backbone

        if isinstance(neck, Mapping):
            from mmdet.models import build_neck

            neck = build_neck(_to_container(neck))
        self.neck = neck

        # It's confusing that backbone weights are given as a separate argument,
        # but "neck" weights, if any, are part of neck itself. This is the interface
        # of mmdet so we follow it. Reference:
        # https://github.com/open-mmlab/mmdetection/blob/master/mmdet/models/detectors/two_stage.py
        logger.info(f"Initializing mmdet backbone weights: {pretrained_backbone} ...")
        self.backbone.init_weights(pretrained_backbone)
        # train() in mmdet modules is non-trivial, and has to be explicitly
        # called. Reference:
        # https://github.com/open-mmlab/mmdetection/blob/master/mmdet/models/backbones/resnet.py
        self.backbone.train()
        if self.neck is not None:
            logger.info("Initializing mmdet neck weights ...")
            if isinstance(self.neck, nn.Sequential):
                for m in self.neck:
                    m.init_weights()
            else:
                self.neck.init_weights()
            self.neck.train()

        self._output_shapes = output_shapes
        if not output_names:
            output_names = [f"out{i}" for i in range(len(output_shapes))]
        self._output_names = output_names

    def forward(self, x) -> Dict[str, Tensor]:
        outs = self.backbone(x)
        if self.neck is not None:
            outs = self.neck(outs)
        assert isinstance(
            outs, (list, tuple)
        ), "mmdet backbone should return a list/tuple of tensors!"
        if len(outs) != len(self._output_shapes):
            raise ValueError(
                "Length of output_shapes does not match outputs from the mmdet backbone: "
                f"{len(outs)} != {len(self._output_shapes)}"
            )
        return {k: v for k, v in zip(self._output_names, outs)}

    def output_shape(self) -> Dict[str, ShapeSpec]:
        return {k: v for k, v in zip(self._output_names, self._output_shapes)}
