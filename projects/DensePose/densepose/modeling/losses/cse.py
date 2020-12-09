# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from typing import Any, List
from torch import nn

from detectron2.config import CfgNode
from detectron2.structures import Instances

from .embed import EmbeddingLoss
from .embed_utils import CseAnnotationsAccumulator
from .mask_or_segm import MaskOrSegmentationLoss
from .registry import DENSEPOSE_LOSS_REGISTRY
from .soft_embed import SoftEmbeddingLoss
from .utils import BilinearInterpolationHelper, LossDict, extract_packed_annotations_from_matches


@DENSEPOSE_LOSS_REGISTRY.register()
class DensePoseCseLoss:
    """"""

    _EMBED_LOSS_REGISTRY = {
        EmbeddingLoss.__name__: EmbeddingLoss,
        SoftEmbeddingLoss.__name__: SoftEmbeddingLoss,
    }

    def __init__(self, cfg: CfgNode):
        """
        Initialize CSE loss from configuration options

        Args:
            cfg (CfgNode): configuration options
        """
        self.w_segm = cfg.MODEL.ROI_DENSEPOSE_HEAD.INDEX_WEIGHTS
        self.w_embed = cfg.MODEL.ROI_DENSEPOSE_HEAD.CSE.EMBED_LOSS_WEIGHT
        self.segm_loss = MaskOrSegmentationLoss(cfg)
        self.embed_loss = DensePoseCseLoss.create_embed_loss(cfg)

    @classmethod
    def create_embed_loss(cls, cfg: CfgNode):
        # registry not used here, since embedding losses are currently local
        # and are not used anywhere else
        return cls._EMBED_LOSS_REGISTRY[cfg.MODEL.ROI_DENSEPOSE_HEAD.CSE.EMBED_LOSS_NAME](cfg)

    def __call__(
        self,
        proposals_with_gt: List[Instances],
        densepose_predictor_outputs: Any,
        embedder: nn.Module,
    ) -> LossDict:
        if not len(proposals_with_gt):
            return self.produce_fake_losses(densepose_predictor_outputs, embedder)
        accumulator = CseAnnotationsAccumulator()
        packed_annotations = extract_packed_annotations_from_matches(proposals_with_gt, accumulator)
        if packed_annotations is None:
            return self.produce_fake_losses(densepose_predictor_outputs, embedder)
        interpolator = BilinearInterpolationHelper.from_matches(
            packed_annotations, tuple(densepose_predictor_outputs.embedding.shape[2:])
        )
        meshid_to_embed_losses = self.embed_loss(
            proposals_with_gt,
            densepose_predictor_outputs,
            packed_annotations,
            interpolator,
            embedder,
        )
        embed_loss_dict = {
            f"loss_densepose_E{meshid}": self.w_embed * meshid_to_embed_losses[meshid]
            for meshid in meshid_to_embed_losses
        }

        return {
            "loss_densepose_S": self.w_segm
            * self.segm_loss(proposals_with_gt, densepose_predictor_outputs, packed_annotations),
            **embed_loss_dict,
        }

    def produce_fake_losses(
        self, densepose_predictor_outputs: Any, embedder: nn.Module
    ) -> LossDict:
        meshname_to_embed_losses = self.embed_loss.fake_values(
            densepose_predictor_outputs, embedder=embedder
        )
        embed_loss_dict = {
            f"loss_densepose_E{mesh_name}": meshname_to_embed_losses[mesh_name]
            for mesh_name in meshname_to_embed_losses
        }
        return {
            "loss_densepose_S": self.segm_loss.fake_value(densepose_predictor_outputs),
            **embed_loss_dict,
        }
