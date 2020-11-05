# Copyright (c) Facebook, Inc. and its affiliates.

import random
import torch

from .densepose_base import DensePoseBaseSampler


class DensePoseUniformSampler(DensePoseBaseSampler):
    """
    Samples DensePose data from DensePose predictions.
    Samples for each class are drawn uniformly over all pixels estimated
    to belong to that class.
    """

    def __init__(self, count_per_class: int = 8):
        """
        Constructor

        Args:
          count_per_class (int): the sampler produces at most `count_per_class`
              samples for each category
        """
        super().__init__(count_per_class)

    def _produce_index_sample(self, values: torch.Tensor, count: int):
        """
        Produce a uniform sample of indices to select data

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
        return random.sample(range(k), count)
