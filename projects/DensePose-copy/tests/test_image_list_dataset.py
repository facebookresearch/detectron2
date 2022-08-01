# Copyright (c) Facebook, Inc. and its affiliates.

import contextlib
import os
import tempfile
import unittest
import torch
from torchvision.utils import save_image

from densepose.data.image_list_dataset import ImageListDataset
from densepose.data.transform import ImageResizeTransform


@contextlib.contextmanager
def temp_image(height, width):
    random_image = torch.rand(height, width)
    with tempfile.NamedTemporaryFile(suffix=".jpg") as f:
        f.close()
        save_image(random_image, f.name)
        yield f.name
    os.unlink(f.name)


class TestImageListDataset(unittest.TestCase):
    def test_image_list_dataset(self):
        height, width = 720, 1280
        with temp_image(height, width) as image_fpath:
            image_list = [image_fpath]
            category_list = [None]
            dataset = ImageListDataset(image_list, category_list)
            self.assertEqual(len(dataset), 1)
            data1, categories1 = dataset[0]["images"], dataset[0]["categories"]
            self.assertEqual(data1.shape, torch.Size((1, 3, height, width)))
            self.assertEqual(data1.dtype, torch.float32)
            self.assertIsNone(categories1[0])

    def test_image_list_dataset_with_transform(self):
        height, width = 720, 1280
        with temp_image(height, width) as image_fpath:
            image_list = [image_fpath]
            category_list = [None]
            transform = ImageResizeTransform()
            dataset = ImageListDataset(image_list, category_list, transform)
            self.assertEqual(len(dataset), 1)
            data1, categories1 = dataset[0]["images"], dataset[0]["categories"]
            self.assertEqual(data1.shape, torch.Size((1, 3, 749, 1333)))
            self.assertEqual(data1.dtype, torch.float32)
            self.assertIsNone(categories1[0])
