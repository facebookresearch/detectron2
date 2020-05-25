# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import unittest
from typing import Sequence
import torch

from detectron2.structures import ImageList


class TestImageList(unittest.TestCase):
    def test_imagelist_padding_shape(self):
        class TensorToImageList(torch.nn.Module):
            def forward(self, tensors: Sequence[torch.Tensor]):
                return ImageList.from_tensors(tensors, 4).tensor

        func = torch.jit.trace(
            TensorToImageList(), ([torch.ones((3, 10, 10), dtype=torch.float32)],)
        )
        ret = func([torch.ones((3, 15, 20), dtype=torch.float32)])
        self.assertEqual(list(ret.shape), [1, 3, 16, 20], str(ret.shape))

        func = torch.jit.trace(
            TensorToImageList(),
            (
                [
                    torch.ones((3, 16, 10), dtype=torch.float32),
                    torch.ones((3, 13, 11), dtype=torch.float32),
                ],
            ),
        )
        ret = func(
            [
                torch.ones((3, 25, 20), dtype=torch.float32),
                torch.ones((3, 10, 10), dtype=torch.float32),
            ]
        )
        # does not support calling with different #images
        self.assertEqual(list(ret.shape), [2, 3, 28, 20], str(ret.shape))
