# Copyright (c) Facebook, Inc. and its affiliates.
import numpy as np
import unittest
import torch

from detectron2.layers import ciou_loss, diou_loss


class TestLosses(unittest.TestCase):
    def test_diou_loss(self):
        """
        loss = 1 - iou + d/c
        where,
        d = (distance between centers of the 2 boxes)^2
        c = (diagonal length of the smallest enclosing box covering the 2 boxes)^2
        """
        # Identical boxes should have loss of 0
        box = torch.tensor([-1, -1, 1, 1], dtype=torch.float32)
        loss = diou_loss(box, box)
        self.assertTrue(np.allclose(loss, [0.0]))

        # Half size box inside other box
        # iou = 0.5, d = 0.25, c = 8
        box2 = torch.tensor([0, -1, 1, 1], dtype=torch.float32)
        loss = diou_loss(box, box2)
        self.assertTrue(np.allclose(loss, [0.53125]))

        # Two diagonally adjacent boxes
        # iou = 0, d = 2, c = 8
        box3 = torch.tensor([0, 0, 1, 1], dtype=torch.float32)
        box4 = torch.tensor([1, 1, 2, 2], dtype=torch.float32)
        loss = diou_loss(box3, box4)
        self.assertTrue(np.allclose(loss, [1.25]))

        # Test batched loss and reductions
        box1s = torch.stack([box, box3], dim=0)
        box2s = torch.stack([box2, box4], dim=0)

        loss = diou_loss(box1s, box2s, reduction="sum")
        self.assertTrue(np.allclose(loss, [1.78125]))

        loss = diou_loss(box1s, box2s, reduction="mean")
        self.assertTrue(np.allclose(loss, [0.890625]))

    def test_ciou_loss(self):
        """
        loss = 1 - iou + d/c + alpha*v
        where,
        d = (distance between centers of the 2 boxes)^2
        c = (diagonal length of the smallest enclosing box covering the 2 boxes)^2
        v = (4/pi^2) * (arctan(box1_w/box1_h) - arctan(box2_w/box2_h))^2
        alpha = v/(1 - iou + v)
        """
        # Identical boxes should have loss of 0
        box = torch.tensor([-1, -1, 1, 1], dtype=torch.float32)
        loss = ciou_loss(box, box)
        self.assertTrue(np.allclose(loss, [0.0]))

        # Half size box inside other box
        # iou = 0.5, d = 0.25, c = 8
        # v = (4/pi^2) * (arctan(1) - arctan(0.5))^2 = 0.042
        # alpha = 0.0775
        box2 = torch.tensor([0, -1, 1, 1], dtype=torch.float32)
        loss = ciou_loss(box, box2)
        self.assertTrue(np.allclose(loss, [0.5345]))

        # Two diagonally adjacent boxes
        # iou = 0, d = 2, c = 8, v = 0, alpha = 0
        box3 = torch.tensor([0, 0, 1, 1], dtype=torch.float32)
        box4 = torch.tensor([1, 1, 2, 2], dtype=torch.float32)
        loss = ciou_loss(box3, box4)
        self.assertTrue(np.allclose(loss, [1.25]))

        # Test batched loss and reductions
        box1s = torch.stack([box, box3], dim=0)
        box2s = torch.stack([box2, box4], dim=0)

        loss = ciou_loss(box1s, box2s, reduction="sum")
        self.assertTrue(np.allclose(loss, [1.7845]))

        loss = ciou_loss(box1s, box2s, reduction="mean")
        self.assertTrue(np.allclose(loss, [0.89225]))
