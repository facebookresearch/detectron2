# Copyright (c) Facebook, Inc. and its affiliates.
import itertools
import math
import operator
import unittest
import torch
from torch.utils import data
from torch.utils.data.sampler import SequentialSampler

from detectron2.data.build import worker_init_reset_seed
from detectron2.data.common import DatasetFromList, ToIterableDataset
from detectron2.data.samplers import (
    GroupedBatchSampler,
    InferenceSampler,
    RepeatFactorTrainingSampler,
    TrainingSampler,
)
from detectron2.utils.env import seed_all_rng


class TestGroupedBatchSampler(unittest.TestCase):
    def test_missing_group_id(self):
        sampler = SequentialSampler(list(range(100)))
        group_ids = [1] * 100
        samples = GroupedBatchSampler(sampler, group_ids, 2)

        for mini_batch in samples:
            self.assertEqual(len(mini_batch), 2)

    def test_groups(self):
        sampler = SequentialSampler(list(range(100)))
        group_ids = [1, 0] * 50
        samples = GroupedBatchSampler(sampler, group_ids, 2)

        for mini_batch in samples:
            self.assertEqual((mini_batch[0] + mini_batch[1]) % 2, 0)


class TestSamplerDeterministic(unittest.TestCase):
    def test_to_iterable(self):
        sampler = TrainingSampler(100, seed=10)
        gt_output = list(itertools.islice(sampler, 100))
        self.assertEqual(set(gt_output), set(range(100)))

        dataset = DatasetFromList(list(range(100)))
        dataset = ToIterableDataset(dataset, sampler)
        data_loader = data.DataLoader(dataset, num_workers=0, collate_fn=operator.itemgetter(0))

        output = list(itertools.islice(data_loader, 100))
        self.assertEqual(output, gt_output)

        data_loader = data.DataLoader(
            dataset,
            num_workers=2,
            collate_fn=operator.itemgetter(0),
            worker_init_fn=worker_init_reset_seed,
            # reset seed should not affect behavior of TrainingSampler
        )
        output = list(itertools.islice(data_loader, 100))
        # multiple workers should not lead to duplicate or different data
        self.assertEqual(output, gt_output)

    def test_training_sampler_seed(self):
        seed_all_rng(42)
        sampler = TrainingSampler(30)
        data = list(itertools.islice(sampler, 65))

        seed_all_rng(42)
        sampler = TrainingSampler(30)
        seed_all_rng(999)  # should be ineffective
        data2 = list(itertools.islice(sampler, 65))
        self.assertEqual(data, data2)


class TestRepeatFactorTrainingSampler(unittest.TestCase):
    def test_repeat_factors_from_category_frequency(self):
        repeat_thresh = 0.5

        dataset_dicts = [
            {"annotations": [{"category_id": 0}, {"category_id": 1}]},
            {"annotations": [{"category_id": 0}]},
            {"annotations": []},
        ]

        rep_factors = RepeatFactorTrainingSampler.repeat_factors_from_category_frequency(
            dataset_dicts, repeat_thresh
        )

        expected_rep_factors = torch.tensor([math.sqrt(3 / 2), 1.0, 1.0])
        self.assertTrue(torch.allclose(rep_factors, expected_rep_factors))


class TestInferenceSampler(unittest.TestCase):
    def test_local_indices(self):
        sizes = [0, 16, 2, 42]
        world_sizes = [5, 2, 3, 4]

        expected_results = [
            [range(0) for _ in range(5)],
            [range(8), range(8, 16)],
            [range(1), range(1, 2), range(0)],
            [range(11), range(11, 22), range(22, 32), range(32, 42)],
        ]

        for size, world_size, expected_result in zip(sizes, world_sizes, expected_results):
            with self.subTest(f"size={size}, world_size={world_size}"):
                local_indices = [
                    InferenceSampler._get_local_indices(size, world_size, r)
                    for r in range(world_size)
                ]
                self.assertEqual(local_indices, expected_result)
