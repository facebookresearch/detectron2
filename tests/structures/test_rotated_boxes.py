# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from __future__ import absolute_import, division, print_function, unicode_literals
import logging
import math
import random
import unittest
import torch
from fvcore.common.benchmark import benchmark

from detectron2.layers.rotated_boxes import pairwise_iou_rotated
from detectron2.structures.boxes import Boxes
from detectron2.structures.rotated_boxes import RotatedBoxes, pairwise_iou

logger = logging.getLogger(__name__)


class TestRotatedBoxesLayer(unittest.TestCase):
    def test_iou_0_dim_cpu(self):
        boxes1 = torch.rand(0, 5, dtype=torch.float32)
        boxes2 = torch.rand(10, 5, dtype=torch.float32)
        expected_ious = torch.zeros(0, 10, dtype=torch.float32)
        ious = pairwise_iou_rotated(boxes1, boxes2)
        self.assertTrue(torch.allclose(ious, expected_ious))

        boxes1 = torch.rand(10, 5, dtype=torch.float32)
        boxes2 = torch.rand(0, 5, dtype=torch.float32)
        expected_ious = torch.zeros(10, 0, dtype=torch.float32)
        ious = pairwise_iou_rotated(boxes1, boxes2)
        self.assertTrue(torch.allclose(ious, expected_ious))

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_iou_0_dim_cuda(self):
        boxes1 = torch.rand(0, 5, dtype=torch.float32)
        boxes2 = torch.rand(10, 5, dtype=torch.float32)
        expected_ious = torch.zeros(0, 10, dtype=torch.float32)
        ious_cuda = pairwise_iou_rotated(boxes1.cuda(), boxes2.cuda())
        self.assertTrue(torch.allclose(ious_cuda.cpu(), expected_ious))

        boxes1 = torch.rand(10, 5, dtype=torch.float32)
        boxes2 = torch.rand(0, 5, dtype=torch.float32)
        expected_ious = torch.zeros(10, 0, dtype=torch.float32)
        ious_cuda = pairwise_iou_rotated(boxes1.cuda(), boxes2.cuda())
        self.assertTrue(torch.allclose(ious_cuda.cpu(), expected_ious))

    def test_iou_half_overlap_cpu(self):
        boxes1 = torch.tensor([[0.5, 0.5, 1.0, 1.0, 0.0]], dtype=torch.float32)
        boxes2 = torch.tensor([[0.25, 0.5, 0.5, 1.0, 0.0]], dtype=torch.float32)
        expected_ious = torch.tensor([[0.5]], dtype=torch.float32)
        ious = pairwise_iou_rotated(boxes1, boxes2)
        self.assertTrue(torch.allclose(ious, expected_ious))

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_iou_half_overlap_cuda(self):
        boxes1 = torch.tensor([[0.5, 0.5, 1.0, 1.0, 0.0]], dtype=torch.float32)
        boxes2 = torch.tensor([[0.25, 0.5, 0.5, 1.0, 0.0]], dtype=torch.float32)
        expected_ious = torch.tensor([[0.5]], dtype=torch.float32)
        ious_cuda = pairwise_iou_rotated(boxes1.cuda(), boxes2.cuda())
        self.assertTrue(torch.allclose(ious_cuda.cpu(), expected_ious))

    def test_iou_precision(self):
        for device in ["cpu"] + ["cuda"] if torch.cuda.is_available() else []:
            boxes1 = torch.tensor([[565, 565, 10, 10.0, 0]], dtype=torch.float32, device=device)
            boxes2 = torch.tensor([[565, 565, 10, 8.3, 0]], dtype=torch.float32, device=device)
            iou = 8.3 / 10.0
            expected_ious = torch.tensor([[iou]], dtype=torch.float32)
            ious = pairwise_iou_rotated(boxes1, boxes2)
            self.assertTrue(torch.allclose(ious.cpu(), expected_ious))

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_iou_too_many_boxes_cuda(self):
        s1, s2 = 5, 1289035
        boxes1 = torch.zeros(s1, 5)
        boxes2 = torch.zeros(s2, 5)
        ious_cuda = pairwise_iou_rotated(boxes1.cuda(), boxes2.cuda())
        self.assertTupleEqual(tuple(ious_cuda.shape), (s1, s2))

    def test_iou_extreme(self):
        # Cause floating point issues in cuda kernels (#1266)
        for device in ["cpu"] + ["cuda"] if torch.cuda.is_available() else []:
            boxes1 = torch.tensor([[160.0, 153.0, 230.0, 23.0, -37.0]], device=device)
            boxes2 = torch.tensor(
                [
                    [
                        -1.117407639806935e17,
                        1.3858420478349148e18,
                        1000.0000610351562,
                        1000.0000610351562,
                        1612.0,
                    ]
                ],
                device=device,
            )
            ious = pairwise_iou_rotated(boxes1, boxes2)
            self.assertTrue(ious.min() >= 0, ious)


class TestRotatedBoxesStructure(unittest.TestCase):
    def test_clip_area_0_degree(self):
        for _ in range(50):
            num_boxes = 100
            boxes_5d = torch.zeros(num_boxes, 5)
            boxes_5d[:, 0] = torch.FloatTensor(num_boxes).uniform_(-100, 500)
            boxes_5d[:, 1] = torch.FloatTensor(num_boxes).uniform_(-100, 500)
            boxes_5d[:, 2] = torch.FloatTensor(num_boxes).uniform_(0, 500)
            boxes_5d[:, 3] = torch.FloatTensor(num_boxes).uniform_(0, 500)
            # Convert from (x_ctr, y_ctr, w, h, 0) to  (x1, y1, x2, y2)
            boxes_4d = torch.zeros(num_boxes, 4)
            boxes_4d[:, 0] = boxes_5d[:, 0] - boxes_5d[:, 2] / 2.0
            boxes_4d[:, 1] = boxes_5d[:, 1] - boxes_5d[:, 3] / 2.0
            boxes_4d[:, 2] = boxes_5d[:, 0] + boxes_5d[:, 2] / 2.0
            boxes_4d[:, 3] = boxes_5d[:, 1] + boxes_5d[:, 3] / 2.0

            image_size = (500, 600)
            test_boxes_4d = Boxes(boxes_4d)
            test_boxes_5d = RotatedBoxes(boxes_5d)
            # Before clip
            areas_4d = test_boxes_4d.area()
            areas_5d = test_boxes_5d.area()
            self.assertTrue(torch.allclose(areas_4d, areas_5d, atol=1e-1, rtol=1e-5))
            # After clip
            test_boxes_4d.clip(image_size)
            test_boxes_5d.clip(image_size)
            areas_4d = test_boxes_4d.area()
            areas_5d = test_boxes_5d.area()
            self.assertTrue(torch.allclose(areas_4d, areas_5d, atol=1e-1, rtol=1e-5))

    def test_clip_area_arbitrary_angle(self):
        num_boxes = 100
        boxes_5d = torch.zeros(num_boxes, 5)
        boxes_5d[:, 0] = torch.FloatTensor(num_boxes).uniform_(-100, 500)
        boxes_5d[:, 1] = torch.FloatTensor(num_boxes).uniform_(-100, 500)
        boxes_5d[:, 2] = torch.FloatTensor(num_boxes).uniform_(0, 500)
        boxes_5d[:, 3] = torch.FloatTensor(num_boxes).uniform_(0, 500)
        boxes_5d[:, 4] = torch.FloatTensor(num_boxes).uniform_(-1800, 1800)
        clip_angle_threshold = random.uniform(0, 180)

        image_size = (500, 600)
        test_boxes_5d = RotatedBoxes(boxes_5d)
        # Before clip
        areas_before = test_boxes_5d.area()
        # After clip
        test_boxes_5d.clip(image_size, clip_angle_threshold)
        areas_diff = test_boxes_5d.area() - areas_before

        # the areas should only decrease after clipping
        self.assertTrue(torch.all(areas_diff <= 0))
        # whenever the box is clipped (thus the area shrinks),
        # the angle for the box must be within the clip_angle_threshold
        # Note that the clip function will normalize the angle range
        # to be within (-180, 180]
        self.assertTrue(
            torch.all(torch.abs(boxes_5d[:, 4][torch.where(areas_diff < 0)]) < clip_angle_threshold)
        )

    def test_normalize_angles(self):
        # torch.manual_seed(0)
        for _ in range(50):
            num_boxes = 100
            boxes_5d = torch.zeros(num_boxes, 5)
            boxes_5d[:, 0] = torch.FloatTensor(num_boxes).uniform_(-100, 500)
            boxes_5d[:, 1] = torch.FloatTensor(num_boxes).uniform_(-100, 500)
            boxes_5d[:, 2] = torch.FloatTensor(num_boxes).uniform_(0, 500)
            boxes_5d[:, 3] = torch.FloatTensor(num_boxes).uniform_(0, 500)
            boxes_5d[:, 4] = torch.FloatTensor(num_boxes).uniform_(-1800, 1800)
            rotated_boxes = RotatedBoxes(boxes_5d)
            normalized_boxes = rotated_boxes.clone()
            normalized_boxes.normalize_angles()
            self.assertTrue(torch.all(normalized_boxes.tensor[:, 4] >= -180))
            self.assertTrue(torch.all(normalized_boxes.tensor[:, 4] < 180))
            # x, y, w, h should not change
            self.assertTrue(torch.allclose(boxes_5d[:, :4], normalized_boxes.tensor[:, :4]))
            # the cos/sin values of the angles should stay the same

            self.assertTrue(
                torch.allclose(
                    torch.cos(boxes_5d[:, 4] * math.pi / 180),
                    torch.cos(normalized_boxes.tensor[:, 4] * math.pi / 180),
                    atol=1e-5,
                )
            )

            self.assertTrue(
                torch.allclose(
                    torch.sin(boxes_5d[:, 4] * math.pi / 180),
                    torch.sin(normalized_boxes.tensor[:, 4] * math.pi / 180),
                    atol=1e-5,
                )
            )

    def test_pairwise_iou_0_degree(self):
        for device in ["cpu"] + ["cuda"] if torch.cuda.is_available() else []:
            boxes1 = torch.tensor(
                [[0.5, 0.5, 1.0, 1.0, 0.0], [0.5, 0.5, 1.0, 1.0, 0.0]],
                dtype=torch.float32,
                device=device,
            )
            boxes2 = torch.tensor(
                [
                    [0.5, 0.5, 1.0, 1.0, 0.0],
                    [0.25, 0.5, 0.5, 1.0, 0.0],
                    [0.5, 0.25, 1.0, 0.5, 0.0],
                    [0.25, 0.25, 0.5, 0.5, 0.0],
                    [0.75, 0.75, 0.5, 0.5, 0.0],
                    [1.0, 1.0, 1.0, 1.0, 0.0],
                ],
                dtype=torch.float32,
                device=device,
            )
            expected_ious = torch.tensor(
                [
                    [1.0, 0.5, 0.5, 0.25, 0.25, 0.25 / (2 - 0.25)],
                    [1.0, 0.5, 0.5, 0.25, 0.25, 0.25 / (2 - 0.25)],
                ],
                dtype=torch.float32,
                device=device,
            )
            ious = pairwise_iou(RotatedBoxes(boxes1), RotatedBoxes(boxes2))
            self.assertTrue(torch.allclose(ious, expected_ious))

    def test_pairwise_iou_45_degrees(self):
        for device in ["cpu"] + ["cuda"] if torch.cuda.is_available() else []:
            boxes1 = torch.tensor(
                [
                    [1, 1, math.sqrt(2), math.sqrt(2), 45],
                    [1, 1, 2 * math.sqrt(2), 2 * math.sqrt(2), -45],
                ],
                dtype=torch.float32,
                device=device,
            )
            boxes2 = torch.tensor([[1, 1, 2, 2, 0]], dtype=torch.float32, device=device)
            expected_ious = torch.tensor([[0.5], [0.5]], dtype=torch.float32, device=device)
            ious = pairwise_iou(RotatedBoxes(boxes1), RotatedBoxes(boxes2))
            self.assertTrue(torch.allclose(ious, expected_ious))

    def test_pairwise_iou_orthogonal(self):
        for device in ["cpu"] + ["cuda"] if torch.cuda.is_available() else []:
            boxes1 = torch.tensor([[5, 5, 10, 6, 55]], dtype=torch.float32, device=device)
            boxes2 = torch.tensor([[5, 5, 10, 6, -35]], dtype=torch.float32, device=device)
            iou = (6.0 * 6.0) / (6.0 * 6.0 + 4.0 * 6.0 + 4.0 * 6.0)
            expected_ious = torch.tensor([[iou]], dtype=torch.float32, device=device)
            ious = pairwise_iou(RotatedBoxes(boxes1), RotatedBoxes(boxes2))
            self.assertTrue(torch.allclose(ious, expected_ious))

    def test_pairwise_iou_large_close_boxes(self):
        for device in ["cpu"] + ["cuda"] if torch.cuda.is_available() else []:
            boxes1 = torch.tensor(
                [[299.500000, 417.370422, 600.000000, 364.259186, 27.1828]],
                dtype=torch.float32,
                device=device,
            )
            boxes2 = torch.tensor(
                [[299.500000, 417.370422, 600.000000, 364.259155, 27.1828]],
                dtype=torch.float32,
                device=device,
            )
            iou = 364.259155 / 364.259186
            expected_ious = torch.tensor([[iou]], dtype=torch.float32, device=device)
            ious = pairwise_iou(RotatedBoxes(boxes1), RotatedBoxes(boxes2))
            self.assertTrue(torch.allclose(ious, expected_ious))

    def test_pairwise_iou_many_boxes(self):
        for device in ["cpu"] + ["cuda"] if torch.cuda.is_available() else []:
            num_boxes1 = 100
            num_boxes2 = 200
            boxes1 = torch.stack(
                [
                    torch.tensor(
                        [5 + 20 * i, 5 + 20 * i, 10, 10, 0], dtype=torch.float32, device=device
                    )
                    for i in range(num_boxes1)
                ]
            )
            boxes2 = torch.stack(
                [
                    torch.tensor(
                        [5 + 20 * i, 5 + 20 * i, 10, 1 + 9 * i / num_boxes2, 0],
                        dtype=torch.float32,
                        device=device,
                    )
                    for i in range(num_boxes2)
                ]
            )
            expected_ious = torch.zeros(num_boxes1, num_boxes2, dtype=torch.float32, device=device)
            for i in range(min(num_boxes1, num_boxes2)):
                expected_ious[i][i] = (1 + 9 * i / num_boxes2) / 10.0
            ious = pairwise_iou(RotatedBoxes(boxes1), RotatedBoxes(boxes2))
            self.assertTrue(torch.allclose(ious, expected_ious))

    def test_pairwise_iou_issue1207_simplified(self):
        for device in ["cpu"] + ["cuda"] if torch.cuda.is_available() else []:
            # Simplified test case of D2-issue-1207
            boxes1 = torch.tensor([[3, 3, 8, 2, -45.0]], device=device)
            boxes2 = torch.tensor([[6, 0, 8, 2, -45.0]], device=device)
            iou = 0.0
            expected_ious = torch.tensor([[iou]], dtype=torch.float32, device=device)

            ious = pairwise_iou(RotatedBoxes(boxes1), RotatedBoxes(boxes2))
            self.assertTrue(torch.allclose(ious, expected_ious))

    def test_pairwise_iou_issue1207(self):
        for device in ["cpu"] + ["cuda"] if torch.cuda.is_available() else []:
            # The original test case in D2-issue-1207
            boxes1 = torch.tensor([[160.0, 153.0, 230.0, 23.0, -37.0]], device=device)
            boxes2 = torch.tensor([[190.0, 127.0, 80.0, 21.0, -46.0]], device=device)

            iou = 0.0
            expected_ious = torch.tensor([[iou]], dtype=torch.float32, device=device)

            ious = pairwise_iou(RotatedBoxes(boxes1), RotatedBoxes(boxes2))
            self.assertTrue(torch.allclose(ious, expected_ious))

    def test_empty_cat(self):
        x = RotatedBoxes.cat([])
        self.assertTrue(x.tensor.shape, (0, 5))


def benchmark_rotated_iou():
    num_boxes1 = 200
    num_boxes2 = 500
    boxes1 = torch.stack(
        [
            torch.tensor([5 + 20 * i, 5 + 20 * i, 10, 10, 0], dtype=torch.float32)
            for i in range(num_boxes1)
        ]
    )
    boxes2 = torch.stack(
        [
            torch.tensor(
                [5 + 20 * i, 5 + 20 * i, 10, 1 + 9 * i / num_boxes2, 0], dtype=torch.float32
            )
            for i in range(num_boxes2)
        ]
    )

    def func(dev, n=1):
        b1 = boxes1.to(device=dev)
        b2 = boxes2.to(device=dev)

        def bench():
            for _ in range(n):
                pairwise_iou_rotated(b1, b2)
            if dev.type == "cuda":
                torch.cuda.synchronize()

        return bench

    # only run it once per timed loop, since it's slow
    args = [{"dev": torch.device("cpu"), "n": 1}]
    if torch.cuda.is_available():
        args.append({"dev": torch.device("cuda"), "n": 10})

    benchmark(func, "rotated_iou", args, warmup_iters=3)


if __name__ == "__main__":
    unittest.main()
    benchmark_rotated_iou()
