# Copyright (c) Facebook, Inc. and its affiliates.

import unittest
import torch

from densepose.data.transform import ImageResizeTransform


class TestImageResizeTransform(unittest.TestCase):
    def test_image_resize_1(self):
        images_batch = torch.ones((3, 3, 100, 100), dtype=torch.uint8) * 100
        transform = ImageResizeTransform()
        images_transformed = transform(images_batch)
        IMAGES_GT = torch.ones((3, 3, 800, 800), dtype=torch.float) * 100
        self.assertEqual(images_transformed.size(), IMAGES_GT.size())
        self.assertAlmostEqual(torch.abs(IMAGES_GT - images_transformed).max().item(), 0.0)
