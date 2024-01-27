# Copyright (c) Facebook, Inc. and its affiliates.

import random
from typing import Optional, Tuple
import torch
from torch.nn import functional as F

from detectron2.config import CfgNode
from detectron2.structures import Instances

from densepose.converters.base import IntTupleBox

from .densepose_cse_base import DensePoseCSEBaseSampler


class DensePoseCSEConfidenceBasedSampler(DensePoseCSEBaseSampler):
    """
    Samples DensePose data from DensePose predictions.
    Samples for each class are drawn using confidence value estimates.
    """

    def __init__(
        self,
        cfg: CfgNode,
        use_gt_categories: bool,
        embedder: torch.nn.Module,
        confidence_channel: str,
        count_per_class: int = 8,
        search_count_multiplier: Optional[float] = None,
        search_proportion: Optional[float] = None,
    ):
        """
        Constructor

        Args:
          cfg (CfgNode): the config of the model
          embedder (torch.nn.Module): necessary to compute mesh vertex embeddings
          confidence_channel (str): confidence channel to use for sampling;
            possible values:
              "coarse_segm_confidence": confidences for coarse segmentation
            (default: "coarse_segm_confidence")
          count_per_class (int): the sampler produces at most `count_per_class`
              samples for each category (default: 8)
          search_count_multiplier (float or None): if not None, the total number
              of the most confident estimates of a given class to consider is
              defined as `min(search_count_multiplier * count_per_class, N)`,
              where `N` is the total number of estimates of the class; cannot be
              specified together with `search_proportion` (default: None)
          search_proportion (float or None): if not None, the total number of the
              of the most confident estimates of a given class to consider is
              defined as `min(max(search_proportion * N, count_per_class), N)`,
              where `N` is the total number of estimates of the class; cannot be
              specified together with `search_count_multiplier` (default: None)
        """
        super().__init__(cfg, use_gt_categories, embedder, count_per_class)
        self.confidence_channel = confidence_channel
        self.search_count_multiplier = search_count_multiplier
        self.search_proportion = search_proportion
        assert (search_count_multiplier is None) or (search_proportion is None), (
            f"Cannot specify both search_count_multiplier (={search_count_multiplier})"
            f"and search_proportion (={search_proportion})"
        )

    def _produce_index_sample(self, values: torch.Tensor, count: int):
        """
        Produce a sample of indices to select data based on confidences

        Args:
            values (torch.Tensor): a tensor of length k that contains confidences
                k: number of points labeled with part_id
            count (int): number of samples to produce, should be positive and <= k

        Return:
            list(int): indices of values (along axis 1) selected as a sample
        """
        k = values.shape[1]
        if k == count:
            index_sample = list(range(k))
        else:
            # take the best count * search_count_multiplier pixels,
            # sample from them uniformly
            # (here best = smallest variance)
            _, sorted_confidence_indices = torch.sort(values[0])
            if self.search_count_multiplier is not None:
                search_count = min(int(count * self.search_count_multiplier), k)
            elif self.search_proportion is not None:
                search_count = min(max(int(k * self.search_proportion), count), k)
            else:
                search_count = min(count, k)
            sample_from_top = random.sample(range(search_count), count)
            index_sample = sorted_confidence_indices[-search_count:][sample_from_top]
        return index_sample

    def _produce_mask_and_results(
        self, instance: Instances, bbox_xywh: IntTupleBox
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Method to get labels and DensePose results from an instance

        Args:
            instance (Instances): an instance of
                `DensePoseEmbeddingPredictorOutputWithConfidences`
            bbox_xywh (IntTupleBox): the corresponding bounding box

        Return:
            mask (torch.Tensor): shape [H, W], DensePose segmentation mask
            embeddings (Tuple[torch.Tensor]): a tensor of shape [D, H, W]
                DensePose CSE Embeddings
            other_values: a tensor of shape [1, H, W], DensePose CSE confidence
        """
        _, _, w, h = bbox_xywh
        densepose_output = instance.pred_densepose
        mask, embeddings, _ = super()._produce_mask_and_results(instance, bbox_xywh)
        other_values = F.interpolate(
            getattr(densepose_output, self.confidence_channel),
            size=(h, w),
            mode="bilinear",
        )[0].cpu()
        return mask, embeddings, other_values
