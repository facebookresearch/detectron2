# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import random
from typing import List, Optional
import torch

from .densepose_base import DensePoseBaseSampler


class DensePoseConfidenceBasedSampler(DensePoseBaseSampler):
    """
    Samples DensePose data from DensePose predictions.
    Samples for each class are drawn using confidence value estimates.
    """

    def __init__(
        self,
        confidence_channel: str,
        count_per_class: int = 8,
        search_count_multiplier: Optional[float] = None,
        search_proportion: Optional[float] = None,
    ):
        """
        Constructor

        Args:
          confidence_channel (str): confidence channel to use for sampling;
            possible values:
              "sigma_2": confidences for UV values
              "fine_segm_confidence": confidences for fine segmentation
              "coarse_segm_confidence": confidences for coarse segmentation
            (default: "sigma_2")
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
        super().__init__(count_per_class)
        self.confidence_channel = confidence_channel
        self.search_count_multiplier = search_count_multiplier
        self.search_proportion = search_proportion
        assert (search_count_multiplier is None) or (search_proportion is None), (
            f"Cannot specify both search_count_multiplier (={search_count_multiplier})"
            f"and search_proportion (={search_proportion})"
        )

    def _confidence_channels(self) -> Optional[List[str]]:
        """
        Confedence channels to be used for sampling (to be overridden in children)
        """
        return [self.confidence_channel]

    def _produce_index_sample(self, values: torch.Tensor, count: int):
        """
        Produce a sample of indices to select data based on confidences

        Args:
            values (torch.Tensor): an array of size [n, k] that contains
                estimated values (U, V, confidences);
                n: number of channels (U, V, confidences)
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
            _, sorted_confidence_indices = torch.sort(values[2])
            if self.search_count_multiplier is not None:
                search_count = min(int(count * self.search_count_multiplier), k)
            elif self.search_proportion is not None:
                search_count = min(max(int(k * self.search_proportion), count), k)
            else:
                search_count = min(count, k)
            sample_from_top = random.sample(range(search_count), count)
            index_sample = sorted_confidence_indices[:search_count][sample_from_top]
        return index_sample
