# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from __future__ import absolute_import, division, print_function, unicode_literals
import unittest
import torch

from detectron2.layers import batched_nms
from detectron2.utils.env import TORCH_VERSION


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
            keep_ref = batched_nms(boxes, scores, idxs, iou)
            backup = boxes.clone()
            scripted_keep = scripted_batched_nms(boxes, scores, idxs, iou)
            assert torch.allclose(boxes, backup), "boxes modified by jit-scripted batched_nms"
            self.assertTrue(torch.equal(keep_ref, scripted_keep), err_msg.format(iou))


if __name__ == "__main__":
    unittest.main()
