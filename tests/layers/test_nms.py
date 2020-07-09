# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import unittest
import torch

from detectron2.layers import batched_nms
from detectron2.utils.env import TORCH_VERSION


def nms_edit_distance(keep1, keep2):
    """
    Compare the "keep" result of two nms call.
    They are allowed to be different in terms of edit distance
    due to floating point precision issues, e.g.,
    if a box happen to have an IoU of 0.5 with another box,
    one implentation may choose to keep it while another may discard it.
    """
    if torch.equal(keep1, keep2):
        # they should be equal most of the time
        return 0
    keep1, keep2 = tuple(keep1.cpu()), tuple(keep2.cpu())
    m, n = len(keep1), len(keep2)

    # edit distance with DP
    f = [np.arange(n + 1), np.arange(n + 1)]
    for i in range(m):
        cur_row = i % 2
        other_row = (i + 1) % 2
        f[other_row][0] = i + 1
        for j in range(n):
            f[other_row][j + 1] = (
                f[cur_row][j]
                if keep1[i] == keep2[j]
                else min(min(f[cur_row][j], f[cur_row][j + 1]), f[other_row][j]) + 1
            )
    return f[m % 2][n]


class TestNMS(unittest.TestCase):
    def _create_tensors(self, N):
        boxes = torch.rand(N, 4) * 100
        # Note: the implementation of this function in torchvision is:
        # boxes[:, 2:] += torch.rand(N, 2) * 100
        # but it does not guarantee non-negative widths/heights constraints:
        # boxes[:, 2] >= boxes[:, 0] and boxes[:, 3] >= boxes[:, 1]:
        boxes[:, 2:] += boxes[:, :2]
        scores = torch.rand(N)
        return boxes, scores

    @unittest.skipIf(TORCH_VERSION < (1, 6), "Insufficient pytorch version")
    def test_nms_scriptability(self):
        N = 2000
        num_classes = 50
        boxes, scores = self._create_tensors(N)
        idxs = torch.randint(0, num_classes, (N,))
        scripted_batched_nms = torch.jit.script(batched_nms)
        err_msg = "NMS is incompatible with jit-scripted NMS for IoU={}"

        for iou in [0.2, 0.5, 0.8]:
            backup = boxes.clone()
            keep_ref = batched_nms(boxes, scores, idxs, iou)
            scripted_keep = scripted_batched_nms(boxes, scores, idxs, iou)
            assert torch.allclose(boxes, backup), "boxes modified by jit-scripted batched_nms"
            self.assertEqual(nms_edit_distance(keep_ref, scripted_keep), 0, err_msg.format(iou))


if __name__ == "__main__":
    unittest.main()
