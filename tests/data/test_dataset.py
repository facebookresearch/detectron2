# Copyright (c) Facebook, Inc. and its affiliates.

import os
import pickle
import sys
import unittest
from functools import partial
import torch
from iopath.common.file_io import LazyPath

from detectron2.data.build import DatasetFromList, MapDataset


def _a_slow_func(x):
    return "path/{}".format(x)


class TestDatasetFromList(unittest.TestCase):
    # Failing for py3.6, likely due to pickle
    @unittest.skipIf(sys.version_info.minor <= 6, "Not supported in Python 3.6")
    def test_using_lazy_path(self):
        dataset = []
        for i in range(10):
            dataset.append({"file_name": LazyPath(partial(_a_slow_func, i))})

        dataset = DatasetFromList(dataset)
        for i in range(10):
            path = dataset[i]["file_name"]
            self.assertTrue(isinstance(path, LazyPath))
            self.assertEqual(os.fspath(path), _a_slow_func(i))


class TestMapDataset(unittest.TestCase):
    @staticmethod
    def map_func(x):
        if x == 2:
            return None
        return x * 2

    def test_map_style(self):
        ds = DatasetFromList([1, 2, 3])
        ds = MapDataset(ds, TestMapDataset.map_func)
        self.assertEqual(ds[0], 2)
        self.assertEqual(ds[2], 6)
        self.assertIn(ds[1], [2, 6])

    def test_iter_style(self):
        class DS(torch.utils.data.IterableDataset):
            def __iter__(self):
                yield from [1, 2, 3]

        ds = DS()
        ds = MapDataset(ds, TestMapDataset.map_func)
        self.assertIsInstance(ds, torch.utils.data.IterableDataset)

        data = list(iter(ds))
        self.assertEqual(data, [2, 6])

    def test_pickleability(self):
        ds = DatasetFromList([1, 2, 3])
        ds = MapDataset(ds, lambda x: x * 2)
        ds = pickle.loads(pickle.dumps(ds))
        self.assertEqual(ds[0], 2)
