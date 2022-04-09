# Copyright (c) Facebook, Inc. and its affiliates.
import logging
import unittest
import torch

from detectron2.modeling.box_regression import (
    Box2BoxTransform,
    Box2BoxTransformLinear,
    Box2BoxTransformRotated,
)
from detectron2.utils.testing import random_boxes

logger = logging.getLogger(__name__)


class TestBox2BoxTransform(unittest.TestCase):
    def test_reconstruction(self):
        weights = (5, 5, 10, 10)
        b2b_tfm = Box2BoxTransform(weights=weights)
        src_boxes = random_boxes(10)
        dst_boxes = random_boxes(10)

        devices = [torch.device("cpu")]
        if torch.cuda.is_available():
            devices.append(torch.device("cuda"))
        for device in devices:
            src_boxes = src_boxes.to(device=device)
            dst_boxes = dst_boxes.to(device=device)
            deltas = b2b_tfm.get_deltas(src_boxes, dst_boxes)
            dst_boxes_reconstructed = b2b_tfm.apply_deltas(deltas, src_boxes)
            self.assertTrue(torch.allclose(dst_boxes, dst_boxes_reconstructed))

    def test_apply_deltas_tracing(self):
        weights = (5, 5, 10, 10)
        b2b_tfm = Box2BoxTransform(weights=weights)

        with torch.no_grad():
            func = torch.jit.trace(b2b_tfm.apply_deltas, (torch.randn(10, 20), torch.randn(10, 4)))

            o = func(torch.randn(10, 20), torch.randn(10, 4))
            self.assertEqual(o.shape, (10, 20))
            o = func(torch.randn(5, 20), torch.randn(5, 4))
            self.assertEqual(o.shape, (5, 20))


def random_rotated_boxes(mean_box, std_length, std_angle, N):
    return torch.cat(
        [torch.rand(N, 4) * std_length, torch.rand(N, 1) * std_angle], dim=1
    ) + torch.tensor(mean_box, dtype=torch.float)


class TestBox2BoxTransformRotated(unittest.TestCase):
    def test_reconstruction(self):
        weights = (5, 5, 10, 10, 1)
        b2b_transform = Box2BoxTransformRotated(weights=weights)
        src_boxes = random_rotated_boxes([10, 10, 20, 20, -30], 5, 60.0, 10)
        dst_boxes = random_rotated_boxes([10, 10, 20, 20, -30], 5, 60.0, 10)

        devices = [torch.device("cpu")]
        if torch.cuda.is_available():
            devices.append(torch.device("cuda"))
        for device in devices:
            src_boxes = src_boxes.to(device=device)
            dst_boxes = dst_boxes.to(device=device)
            deltas = b2b_transform.get_deltas(src_boxes, dst_boxes)
            dst_boxes_reconstructed = b2b_transform.apply_deltas(deltas, src_boxes)
            assert torch.allclose(dst_boxes[:, :4], dst_boxes_reconstructed[:, :4], atol=1e-5)
            # angle difference has to be normalized
            assert torch.allclose(
                (dst_boxes[:, 4] - dst_boxes_reconstructed[:, 4] + 180.0) % 360.0 - 180.0,
                torch.zeros_like(dst_boxes[:, 4]),
                atol=1e-4,
            )


class TestBox2BoxTransformLinear(unittest.TestCase):
    def test_reconstruction(self):
        b2b_tfm = Box2BoxTransformLinear()
        src_boxes = random_boxes(10)
        dst_boxes = torch.tensor([0, 0, 101, 101] * 10).reshape(10, 4).float()

        devices = [torch.device("cpu")]
        if torch.cuda.is_available():
            devices.append(torch.device("cuda"))
        for device in devices:
            src_boxes = src_boxes.to(device=device)
            dst_boxes = dst_boxes.to(device=device)
            deltas = b2b_tfm.get_deltas(src_boxes, dst_boxes)
            dst_boxes_reconstructed = b2b_tfm.apply_deltas(deltas, src_boxes)
            self.assertTrue(torch.allclose(dst_boxes, dst_boxes_reconstructed, atol=1e-3))


if __name__ == "__main__":
    unittest.main()
