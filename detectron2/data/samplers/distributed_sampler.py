# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import itertools
import math
from collections import defaultdict
from typing import Optional
import torch
from torch.utils.data.sampler import Sampler
import numpy as np
from detectron2.utils import comm
import math

class TrainingSampler(Sampler):
    """
    In training, we only care about the "infinite stream" of training data.
    So this sampler produces an infinite stream of indices and
    all workers cooperate to correctly shuffle the indices and sample different indices.

    The samplers in each worker effectively produces `indices[worker_id::num_workers]`
    where `indices` is an infinite stream of indices consisting of
    `shuffle(range(size)) + shuffle(range(size)) + ...` (if shuffle is True)
    or `range(size) + range(size) + ...` (if shuffle is False)
    """

    def __init__(self, size: int, shuffle: bool = True, seed: Optional[int] = None):
        """
        Args:
            size (int): the total number of data of the underlying dataset to sample from
            shuffle (bool): whether to shuffle the indices or not
            seed (int): the initial seed of the shuffle. Must be the same
                across all workers. If None, will use a random seed shared
                among workers (require synchronization among all workers).
        """
        self._size = size
        assert size > 0
        self._shuffle = shuffle
        if seed is None:
            seed = comm.shared_random_seed()
        self._seed = int(seed)

        self._rank = comm.get_rank()
        self._world_size = comm.get_world_size()

    def __iter__(self):
        start = self._rank
        yield from itertools.islice(self._infinite_indices(), start, None, self._world_size)

    def _infinite_indices(self):
        g = torch.Generator()
        g.manual_seed(self._seed)
        while True:
            if self._shuffle:
                yield from torch.randperm(self._size, generator=g)
            else:
                yield from torch.arange(self._size)


class RepeatFactorTrainingSampler(Sampler):
    """
    Similar to TrainingSampler, but suitable for training on class imbalanced datasets
    like LVIS. In each epoch, an image may appear multiple times based on its "repeat
    factor". The repeat factor for an image is a function of the frequency the rarest
    category labeled in that image. The "frequency of category c" in [0, 1] is defined
    as the fraction of images in the training set (without repeats) in which category c
    appears.

    See https://arxiv.org/abs/1908.03195 (>= v2) Appendix B.2.
    """

    def __init__(self, dataset_dicts, repeat_thresh, shuffle=True, seed=None):
        """
        Args:
            dataset_dicts (list[dict]): annotations in Detectron2 dataset format.
            repeat_thresh (float): frequency threshold below which data is repeated.
            shuffle (bool): whether to shuffle the indices or not
            seed (int): the initial seed of the shuffle. Must be the same
                across all workers. If None, will use a random seed shared
                among workers (require synchronization among all workers).
        """
        
        self._shuffle = shuffle
        if seed is None:
            seed = comm.shared_random_seed()
        self._seed = int(seed)
        
        self._rank = comm.get_rank()
        self._world_size = comm.get_world_size()

        # Get fractional repeat factors and split into whole number (_int_part)
        # and fractional (_frac_part) parts.
        rep_factors = self._get_repeat_factors(dataset_dicts, repeat_thresh)
        self._int_part = torch.trunc(rep_factors)
        self._frac_part = rep_factors - self._int_part


    def _get_repeat_factors(self, dataset_dicts, repeat_thresh):
        """
        Compute (fractional) per-image repeat factors.

        Args:
            See __init__.

        Returns:
            torch.Tensor: the i-th element is the repeat factor for the dataset image
                at index i.
        """
        # 1. For each category c, compute the fraction of images that contain it: f(c)
        category_freq = defaultdict(int)
        for dataset_dict in dataset_dicts:  # For each image (without repeats)
            cat_ids = {ann["category_id"] for ann in dataset_dict["annotations"]}
            for cat_id in cat_ids:
                category_freq[cat_id] += 1
        num_images = len(dataset_dicts)
        for k, v in category_freq.items():
            category_freq[k] = v / num_images

        # 2. For each category c, compute the category-level repeat factor:
        #    r(c) = max(1, sqrt(t / f(c)))
        category_rep = {
            cat_id: max(1.0, math.sqrt(repeat_thresh / cat_freq))
            for cat_id, cat_freq in category_freq.items()
        }

        # 3. For each image I, compute the image-level repeat factor:
        #    r(I) = max_{c in I} r(c)
        rep_factors = []
        for dataset_dict in dataset_dicts:
            cat_ids = {ann["category_id"] for ann in dataset_dict["annotations"]}
            rep_factor = max({category_rep[cat_id] for cat_id in cat_ids})
            rep_factors.append(rep_factor)

        return torch.tensor(rep_factors, dtype=torch.float32)

    def _get_epoch_indices(self, generator):
        """
        Create a list of dataset indices (with repeats) to use for one epoch.

        Args:
            generator (torch.Generator): pseudo random number generator used for
                stochastic rounding.

        Returns:
            torch.Tensor: list of dataset indices to use in one epoch. Each index
                is repeated based on its calculated repeat factor.
        """
        # Since repeat factors are fractional, we use stochastic rounding so
        # that the target repeat factor is achieved in expectation over the
        # course of training
        rands = torch.rand(len(self._frac_part), generator=generator)
        rep_factors = self._int_part + (rands < self._frac_part).float()
        # Construct a list of indices in which we repeat images as specified
        indices = []
        for dataset_index, rep_factor in enumerate(rep_factors):
            indices.extend([dataset_index] * int(rep_factor.item()))
        return torch.tensor(indices, dtype=torch.int64)

    def __iter__(self):
        start = self._rank
        yield from itertools.islice(self._infinite_indices(), start, None, self._world_size)

    def _infinite_indices(self):
        g = torch.Generator()
        g.manual_seed(self._seed)
        while True:
            # Sample indices with repeats determined by stochastic rounding; each
            # "epoch" may have a slightly different size due to the rounding.
            indices = self._get_epoch_indices(g)
            if self._shuffle:
                randperm = torch.randperm(len(indices), generator=g)
                yield from indices[randperm]
            else:
                yield from indices

class RepeatFactorCurriLTrainingSampler(Sampler):
    """
    Similar to TrainingSampler, but suitable for training on class imbalanced datasets
    like LVIS. In each epoch, an image may appear multiple times based on its "repeat
    factor". The repeat factor for an image is a function of the frequency the rarest
    category labeled in that image. The "frequency of category c" in [0, 1] is defined
    as the fraction of images in the training set (without repeats) in which category c
    appears.

    See https://arxiv.org/abs/1908.03195 (>= v2) Appendix B.2.
    """

    def __init__(self, dataset_dicts, repeat_thresh, shuffle=True, seed=None):
        """
        Args:
            dataset_dicts (list[dict]): annotations in Detectron2 dataset format.
            repeat_thresh (float): frequency threshold below which data is repeated.
            shuffle (bool): whether to shuffle the indices or not
            seed (int): the initial seed of the shuffle. Must be the same
                across all workers. If None, will use a random seed shared
                among workers (require synchronization among all workers).
        """
        self._shuffle = shuffle
        if seed is None:
            seed = comm.shared_random_seed()
        self._seed = int(seed)

        self._rank = comm.get_rank()
        self._world_size = comm.get_world_size()

        # Get fractional repeat factors and split into whole number (_int_part)
        # and fractional (_frac_part) parts.
        rep_factors, frequency_score = self._get_repeat_factors(dataset_dicts, repeat_thresh)
        self._int_part = torch.trunc(rep_factors)
        self._frac_part = rep_factors - self._int_part
        self.frequency_score = frequency_score


    def _get_repeat_factors(self, dataset_dicts, repeat_thresh):
        """
        Compute (fractional) per-image repeat factors.

        Args:
            See __init__.

        Returns:
            torch.Tensor: the i-th element is the repeat factor for the dataset image
                at index i.
        """
        # 1. For each category c, compute the fraction of images that contain it: f(c)
        category_freq = defaultdict(int)

        for dataset_dict in dataset_dicts:  # For each image (without repeats)
            cat_ids = {ann["category_id"] for ann in dataset_dict["annotations"]}
            for cat_id in cat_ids:
                category_freq[cat_id] += 1
        num_images = len(dataset_dicts)
        for k, v in category_freq.items():
            category_freq[k] = v / num_images

        # 2. For each category c, compute the category-level repeat factor:
        #    r(c) = max(1, sqrt(t / f(c)))
        category_rep = {
            cat_id: max(1.0, math.sqrt(repeat_thresh / cat_freq))
            for cat_id, cat_freq in category_freq.items()
        }

        # 3. For each image I, compute the image-level repeat factor:
        #    r(I) = max_{c in I} r(c)
        rep_factors = []
        frequency_list = []
        for dataset_dict in dataset_dicts:
            cat_ids = {ann["category_id"] for ann in dataset_dict["annotations"]}
            frequency_ids = np.array([ann["frequency"] for ann in dataset_dict["annotations"]])
            frequency_ids[frequency_ids==2] = 34./16.
            frequency_ids[frequency_ids==1] = 34./28.
            frequency_ids[frequency_ids==0] = 34./34.
            frequency_ids = list(frequency_ids)
            rep_factor = max({category_rep[cat_id] for cat_id in cat_ids})
            rep_factors.append(rep_factor)
            frequency_list.append(sum(frequency_ids))
#         frequency_list.sort()
#         count = dict()
#         for i in range(len(frequency_list)):
#             if frequency_list[i] not in count.keys():
#                 count[frequency_list[i]] = 1
#             else:
#                 count[frequency_list[i]] += 1
#         print(count)
        return torch.tensor(rep_factors, dtype=torch.float32), torch.tensor(frequency_list, dtype=torch.int32)

    def _get_epoch_indices(self, generator):
        """
        Create a list of dataset indices (with repeats) to use for one epoch.

        Args:
            generator (torch.Generator): pseudo random number generator used for
                stochastic rounding.

        Returns:
            torch.Tensor: list of dataset indices to use in one epoch. Each index
                is repeated based on its calculated repeat factor.
        """
        # Since repeat factors are fractional, we use stochastic rounding so
        # that the target repeat factor is achieved in expectation over the
        # course of training
        rands = torch.rand(len(self._frac_part), generator=generator)
        rep_factors = self._int_part + (rands < self._frac_part).float()
        # Construct a list of indices in which we repeat images as specified
#         self.frequency_score: divide < 1, 1 <= divide <11, 11 <= divide
        indices_0 = []
        indices_1 = []
#         indices_2 = []
        Q30 = np.percentile(self.frequency_score, [30])
        for dataset_index, rep_factor in enumerate(rep_factors):
            freq_score = self.frequency_score[dataset_index]
            if freq_score <= Q30[0]:
                indices_0.extend([dataset_index] * int(rep_factor.item()))
            else:
                indices_1.extend([dataset_index] * int(rep_factor.item()))

#         total_indices = indices_0 + indices_1
#         print('indices_0:',len(indices_0)) # 42925
#         print('indices_1:',len(indices_1)) # 13523
#         print('indices_2:',len(indices_2)) # 1451
#         step_length = math.floor((len(indices_0) + len(indices_1) + len(indices_2)) / 2.)
#         step_0_indices = total_indices[:step_length]
#         step_1_indices = total_indices[step_length:]

#         step_0_indices = np.random.choice(indices_0,size=step_length,replace=False)
#         for item in step_0_indices: indices_0.remove(item)
#         indices_01 = indices_0 + indices_1
#         step_1_indices = np.random.choice(indices_01,size=step_length,replace=False)
#         for item in step_1_indices: indices_01.remove(item)
#         indices_012 = indices_01 + indices_2
#         step_2_indices = np.random.choice(indices_012,size=step_length,replace=False)

        return [torch.tensor(indices_0, dtype=torch.int64),torch.tensor(indices_1, dtype=torch.int64)]
    
    def __iter__(self):
        start = self._rank
        yield from itertools.islice(self._infinite_indices(), start, None, self._world_size)

    def _infinite_indices(self):
        g = torch.Generator()
        g.manual_seed(self._seed)
        while True:
            # Sample indices with repeats determined by stochastic rounding; each
            # "epoch" may have a slightly different size due to the rounding.
            indices_list = self._get_epoch_indices(g)
            if self._shuffle:
#                 randperm = torch.randperm(len(indices_list), generator=g)
                indices_final = []
                for i in range(len(indices_list)):
                    randperm = torch.randperm(len(indices_list[i]), generator=g)
                    indices_final.append(indices_list[i][randperm])
                indices_final = torch.cat(indices_final,0)
                yield from indices_final
            else:
                indices_final = torch.cat(indices_list,0)
                yield from indices_final

class InferenceSampler(Sampler):
    """
    Produce indices for inference.
    Inference needs to run on the __exact__ set of samples,
    therefore when the total number of samples is not divisible by the number of workers,
    this sampler produces different number of samples on different workers.
    """

    def __init__(self, size: int):
        """
        Args:
            size (int): the total number of data of the underlying dataset to sample from
        """
        self._size = size
        assert size > 0
        self._rank = comm.get_rank()
        self._world_size = comm.get_world_size()

        shard_size = (self._size - 1) // self._world_size + 1
        begin = shard_size * self._rank
        end = min(shard_size * (self._rank + 1), self._size)
        self._local_indices = range(begin, end)

    def __iter__(self):
        yield from self._local_indices

    def __len__(self):
        return len(self._local_indices)
