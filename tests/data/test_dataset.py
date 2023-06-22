# Copyright (c) Facebook, Inc. and its affiliates.

import os
import pickle
import sys
import unittest
from functools import partial
import torch
from iopath.common.file_io import LazyPath

from detectron2 import model_zoo
from detectron2.config import get_cfg, instantiate
from detectron2.data import (
    DatasetCatalog,
    DatasetFromList,
    MapDataset,
    ToIterableDataset,
    build_batch_data_loader,
    build_detection_test_loader,
    build_detection_train_loader,
)
from detectron2.data.common import (
    AspectRatioGroupedDataset,
    set_default_dataset_from_list_serialize_method,
)
from detectron2.data.samplers import InferenceSampler, TrainingSampler


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

    def test_alternative_serialize_method(self):
        dataset = [1, 2, 3]
        dataset = DatasetFromList(dataset, serialize=torch.tensor)
        self.assertEqual(dataset[2], torch.tensor(3))

    def test_change_default_serialize_method(self):
        dataset = [1, 2, 3]
        with set_default_dataset_from_list_serialize_method(torch.tensor):
            dataset_1 = DatasetFromList(dataset, serialize=True)
            self.assertEqual(dataset_1[2], torch.tensor(3))
        dataset_2 = DatasetFromList(dataset, serialize=True)
        self.assertEqual(dataset_2[2], 3)


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


class TestAspectRatioGrouping(unittest.TestCase):
    def test_reiter_leak(self):
        data = [(1, 0), (0, 1), (1, 0), (0, 1)]
        data = [{"width": a, "height": b} for (a, b) in data]
        batchsize = 2
        dataset = AspectRatioGroupedDataset(data, batchsize)

        for _ in range(5):
            for idx, __ in enumerate(dataset):
                if idx == 1:
                    # manually break, so the iterator does not stop by itself
                    break
            # check that bucket sizes are valid
            for bucket in dataset._buckets:
                self.assertLess(len(bucket), batchsize)


class _MyData(torch.utils.data.IterableDataset):
    def __iter__(self):
        while True:
            yield 1


class TestDataLoader(unittest.TestCase):
    def _get_kwargs(self):
        # get kwargs of build_detection_train_loader
        cfg = model_zoo.get_config("common/data/coco.py").dataloader.train
        cfg.dataset.names = "coco_2017_val_100"
        cfg.pop("_target_")
        kwargs = {k: instantiate(v) for k, v in cfg.items()}
        return kwargs

    def test_build_dataloader_train(self):
        kwargs = self._get_kwargs()
        dl = build_detection_train_loader(**kwargs)
        next(iter(dl))

    def test_build_iterable_dataloader_train(self):
        kwargs = self._get_kwargs()
        ds = DatasetFromList(kwargs.pop("dataset"))
        ds = ToIterableDataset(ds, TrainingSampler(len(ds)))
        dl = build_detection_train_loader(dataset=ds, **kwargs)
        next(iter(dl))

    def test_build_iterable_dataloader_from_cfg(self):
        cfg = get_cfg()
        cfg.DATASETS.TRAIN = ["iter_data"]
        DatasetCatalog.register("iter_data", lambda: _MyData())
        dl = build_detection_train_loader(cfg, mapper=lambda x: x, aspect_ratio_grouping=False)
        next(iter(dl))

        dl = build_detection_test_loader(cfg, "iter_data", mapper=lambda x: x)
        next(iter(dl))

    def _check_is_range(self, data_loader, N):
        # check that data_loader produces range(N)
        data = list(iter(data_loader))
        data = [x for batch in data for x in batch]  # flatten the batches
        self.assertEqual(len(data), N)
        self.assertEqual(set(data), set(range(N)))

    def test_build_batch_dataloader_inference(self):
        # Test that build_batch_data_loader can be used for inference
        N = 96
        ds = DatasetFromList(list(range(N)))
        sampler = InferenceSampler(len(ds))
        dl = build_batch_data_loader(ds, sampler, 8, num_workers=3)
        self._check_is_range(dl, N)

    def test_build_dataloader_inference(self):
        N = 50
        ds = DatasetFromList(list(range(N)))
        sampler = InferenceSampler(len(ds))
        # test that parallel loader works correctly
        dl = build_detection_test_loader(
            dataset=ds, sampler=sampler, mapper=lambda x: x, num_workers=3
        )
        self._check_is_range(dl, N)

        # test that batch_size works correctly
        dl = build_detection_test_loader(
            dataset=ds, sampler=sampler, mapper=lambda x: x, batch_size=4, num_workers=0
        )
        self._check_is_range(dl, N)

    def test_build_iterable_dataloader_inference(self):
        # Test that build_detection_test_loader supports iterable dataset
        N = 50
        ds = DatasetFromList(list(range(N)))
        ds = ToIterableDataset(ds, InferenceSampler(len(ds)))
        dl = build_detection_test_loader(dataset=ds, mapper=lambda x: x, num_workers=3)
        self._check_is_range(dl, N)
