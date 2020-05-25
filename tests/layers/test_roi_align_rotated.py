# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import logging
import unittest
import cv2
import torch
from torch.autograd import Variable, gradcheck

from detectron2.layers.roi_align import ROIAlign
from detectron2.layers.roi_align_rotated import ROIAlignRotated

logger = logging.getLogger(__name__)


class ROIAlignRotatedTest(unittest.TestCase):
    def _box_to_rotated_box(self, box, angle):
        return [
            (box[0] + box[2]) / 2.0,
            (box[1] + box[3]) / 2.0,
            box[2] - box[0],
            box[3] - box[1],
            angle,
        ]

    def _rot90(self, img, num):
        num = num % 4  # note: -1 % 4 == 3
        for _ in range(num):
            img = img.transpose(0, 1).flip(0)
        return img

    def test_forward_output_0_90_180_270(self):
        for i in range(4):
            # i = 0, 1, 2, 3 corresponding to 0, 90, 180, 270 degrees
            img = torch.arange(25, dtype=torch.float32).reshape(5, 5)
            """
            0  1  2   3 4
            5  6  7   8 9
            10 11 12 13 14
            15 16 17 18 19
            20 21 22 23 24
            """
            box = [1, 1, 3, 3]
            rotated_box = self._box_to_rotated_box(box=box, angle=90 * i)

            result = self._simple_roi_align_rotated(img=img, box=rotated_box, resolution=(4, 4))

            # Here's an explanation for 0 degree case:
            # point 0 in the original input lies at [0.5, 0.5]
            # (the center of bin [0, 1] x [0, 1])
            # point 1 in the original input lies at [1.5, 0.5], etc.
            # since the resolution is (4, 4) that divides [1, 3] x [1, 3]
            # into 4 x 4 equal bins,
            # the top-left bin is [1, 1.5] x [1, 1.5], and its center
            # (1.25, 1.25) lies at the 3/4 position
            # between point 0 and point 1, point 5 and point 6,
            # point 0 and point 5, point 1 and point 6, so it can be calculated as
            # 0.25*(0*0.25+1*0.75)+(5*0.25+6*0.75)*0.75 = 4.5
            result_expected = torch.tensor(
                [
                    [4.5, 5.0, 5.5, 6.0],
                    [7.0, 7.5, 8.0, 8.5],
                    [9.5, 10.0, 10.5, 11.0],
                    [12.0, 12.5, 13.0, 13.5],
                ]
            )
            # This is also an upsampled version of [[6, 7], [11, 12]]

            # When the box is rotated by 90 degrees CCW,
            # the result would be rotated by 90 degrees CW, thus it's -i here
            result_expected = self._rot90(result_expected, -i)

            assert torch.allclose(result, result_expected)

    def test_resize(self):
        H, W = 30, 30
        input = torch.rand(H, W) * 100
        box = [10, 10, 20, 20]
        rotated_box = self._box_to_rotated_box(box, angle=0)
        output = self._simple_roi_align_rotated(img=input, box=rotated_box, resolution=(5, 5))

        input2x = cv2.resize(input.numpy(), (W // 2, H // 2), interpolation=cv2.INTER_LINEAR)
        input2x = torch.from_numpy(input2x)
        box2x = [x / 2 for x in box]
        rotated_box2x = self._box_to_rotated_box(box2x, angle=0)
        output2x = self._simple_roi_align_rotated(img=input2x, box=rotated_box2x, resolution=(5, 5))
        assert torch.allclose(output2x, output)

    def _simple_roi_align_rotated(self, img, box, resolution):
        """
        RoiAlignRotated with scale 1.0 and 0 sample ratio.
        """
        op = ROIAlignRotated(output_size=resolution, spatial_scale=1.0, sampling_ratio=0)
        input = img[None, None, :, :]

        rois = [0] + list(box)
        rois = torch.tensor(rois, dtype=torch.float32)[None, :]
        result_cpu = op.forward(input, rois)
        if torch.cuda.is_available():
            result_cuda = op.forward(input.cuda(), rois.cuda())
            assert torch.allclose(result_cpu, result_cuda.cpu())
        return result_cpu[0, 0]

    def test_empty_box(self):
        img = torch.rand(5, 5)
        out = self._simple_roi_align_rotated(img, [2, 3, 0, 0, 0], (7, 7))
        self.assertTrue((out == 0).all())

    def test_roi_align_rotated_gradcheck_cpu(self):
        dtype = torch.float64
        device = torch.device("cpu")
        roi_align_rotated_op = ROIAlignRotated(
            output_size=(5, 5), spatial_scale=0.5, sampling_ratio=1
        ).to(dtype=dtype, device=device)
        x = torch.rand(1, 1, 10, 10, dtype=dtype, device=device, requires_grad=True)
        # roi format is (batch index, x_center, y_center, width, height, angle)
        rois = torch.tensor(
            [[0, 4.5, 4.5, 9, 9, 0], [0, 2, 7, 4, 4, 0], [0, 7, 7, 4, 4, 0]],
            dtype=dtype,
            device=device,
        )

        def func(input):
            return roi_align_rotated_op(input, rois)

        assert gradcheck(func, (x,)), "gradcheck failed for RoIAlignRotated CPU"
        assert gradcheck(func, (x.transpose(2, 3),)), "gradcheck failed for RoIAlignRotated CPU"

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_roi_align_rotated_gradient_cuda(self):
        """
        Compute gradients for ROIAlignRotated with multiple bounding boxes on the GPU,
        and compare the result with ROIAlign
        """
        # torch.manual_seed(123)
        dtype = torch.float64
        device = torch.device("cuda")
        pool_h, pool_w = (5, 5)

        roi_align = ROIAlign(output_size=(pool_h, pool_w), spatial_scale=1, sampling_ratio=2).to(
            device=device
        )

        roi_align_rotated = ROIAlignRotated(
            output_size=(pool_h, pool_w), spatial_scale=1, sampling_ratio=2
        ).to(device=device)

        x = torch.rand(1, 1, 10, 10, dtype=dtype, device=device, requires_grad=True)
        # x_rotated = x.clone() won't work (will lead to grad_fun=CloneBackward)!
        x_rotated = Variable(x.data.clone(), requires_grad=True)

        # roi_rotated format is (batch index, x_center, y_center, width, height, angle)
        rois_rotated = torch.tensor(
            [[0, 4.5, 4.5, 9, 9, 0], [0, 2, 7, 4, 4, 0], [0, 7, 7, 4, 4, 0]],
            dtype=dtype,
            device=device,
        )

        y_rotated = roi_align_rotated(x_rotated, rois_rotated)
        s_rotated = y_rotated.sum()
        s_rotated.backward()

        # roi format is (batch index, x1, y1, x2, y2)
        rois = torch.tensor(
            [[0, 0, 0, 9, 9], [0, 0, 5, 4, 9], [0, 5, 5, 9, 9]], dtype=dtype, device=device
        )

        y = roi_align(x, rois)
        s = y.sum()
        s.backward()

        assert torch.allclose(
            x.grad, x_rotated.grad
        ), "gradients for ROIAlign and ROIAlignRotated mismatch on CUDA"


if __name__ == "__main__":
    unittest.main()
