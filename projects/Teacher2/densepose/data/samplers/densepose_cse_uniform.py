# Copyright (c) Facebook, Inc. and its affiliates.

from .densepose_cse_base import DensePoseCSEBaseSampler
from .densepose_uniform import DensePoseUniformSampler


class DensePoseCSEUniformSampler(DensePoseCSEBaseSampler, DensePoseUniformSampler):
    """
    Uniform Sampler for CSE
    """

    pass
