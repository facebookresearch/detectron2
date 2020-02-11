# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import unittest
from torch.utils.data.sampler import SequentialSampler

from detectron2.data.samplers import GroupedBatchSampler


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
