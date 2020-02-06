# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import numpy as np
import unittest
import cv2
import torch

from detectron2.layers.roi_align import ROIAlign


class ROIAlignTest(unittest.TestCase):
    def test_forward_output(self):
        input = np.arange(25).reshape(5, 5).astype("float32")
        """
        0  1  2   3 4
        5  6  7   8 9
        10 11 12 13 14
        15 16 17 18 19
        20 21 22 23 24
        """

        output = self._simple_roialign(input, [1, 1, 3, 3], (4, 4), aligned=False)
        output_correct = self._simple_roialign(input, [1, 1, 3, 3], (4, 4), aligned=True)

        # without correction:
        old_results = [
            [7.5, 8, 8.5, 9],
            [10, 10.5, 11, 11.5],
            [12.5, 13, 13.5, 14],
            [15, 15.5, 16, 16.5],
        ]

        # with 0.5 correction:
        correct_results = [
            [4.5, 5.0, 5.5, 6.0],
            [7.0, 7.5, 8.0, 8.5],
            [9.5, 10.0, 10.5, 11.0],
            [12.0, 12.5, 13.0, 13.5],
        ]
        # This is an upsampled version of [[6, 7], [11, 12]]

        self.assertTrue(np.allclose(output.flatten(), np.asarray(old_results).flatten()))
        self.assertTrue(
            np.allclose(output_correct.flatten(), np.asarray(correct_results).flatten())
        )

        # Also see similar issues in tensorflow at
        # https://github.com/tensorflow/tensorflow/issues/26278

    def test_resize(self):
        H, W = 30, 30
        input = np.random.rand(H, W).astype("float32") * 100
        box = [10, 10, 20, 20]
        output = self._simple_roialign(input, box, (5, 5), aligned=True)

        input2x = cv2.resize(input, (W // 2, H // 2), interpolation=cv2.INTER_LINEAR)
        box2x = [x / 2 for x in box]
        output2x = self._simple_roialign(input2x, box2x, (5, 5), aligned=True)
        diff = np.abs(output2x - output)
        self.assertTrue(diff.max() < 1e-4)

    def _simple_roialign(self, img, box, resolution, aligned=True):
        """
        RoiAlign with scale 1.0 and 0 sample ratio.
        """
        if isinstance(resolution, int):
            resolution = (resolution, resolution)
        op = ROIAlign(resolution, 1.0, 0, aligned=aligned)
        input = torch.from_numpy(img[None, None, :, :].astype("float32"))

        rois = [0] + list(box)
        rois = torch.from_numpy(np.asarray(rois)[None, :].astype("float32"))
        output = op.forward(input, rois)
        if torch.cuda.is_available():
            output_cuda = op.forward(input.cuda(), rois.cuda()).cpu()
            self.assertTrue(torch.allclose(output, output_cuda))
        return output[0, 0]

    def _simple_roialign_with_grad(self, img, box, resolution, device):
        if isinstance(resolution, int):
            resolution = (resolution, resolution)

        op = ROIAlign(resolution, 1.0, 0, aligned=True)
        input = torch.from_numpy(img[None, None, :, :].astype("float32"))

        rois = [0] + list(box)
        rois = torch.from_numpy(np.asarray(rois)[None, :].astype("float32"))
        input = input.to(device=device)
        rois = rois.to(device=device)
        input.requires_grad = True
        output = op.forward(input, rois)
        return input, output

    def test_empty_box(self):
        img = np.random.rand(5, 5)
        box = [3, 4, 5, 4]
        o = self._simple_roialign(img, box, 7)
        self.assertTrue(o.shape == (7, 7))
        self.assertTrue((o == 0).all())

        for dev in ["cpu"] + ["cuda"] if torch.cuda.is_available() else []:
            input, output = self._simple_roialign_with_grad(img, box, 7, torch.device(dev))
            output.sum().backward()
            self.assertTrue(torch.allclose(input.grad, torch.zeros_like(input)))

    def test_empty_batch(self):
        input = torch.zeros(0, 3, 10, 10, dtype=torch.float32)
        rois = torch.zeros(0, 5, dtype=torch.float32)
        op = ROIAlign((7, 7), 1.0, 0, aligned=True)
        output = op.forward(input, rois)
        self.assertTrue(output.shape == (0, 3, 7, 7))


if __name__ == "__main__":
    unittest.main()
