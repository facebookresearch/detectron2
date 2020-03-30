# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import json
import math
import numpy as np
import unittest
import torch

from detectron2.structures import Boxes, BoxMode, pairwise_iou


class TestBoxMode(unittest.TestCase):
    def _convert_xy_to_wh(self, x):
        return BoxMode.convert(x, BoxMode.XYXY_ABS, BoxMode.XYWH_ABS)

    def _convert_xywha_to_xyxy(self, x):
        return BoxMode.convert(x, BoxMode.XYWHA_ABS, BoxMode.XYXY_ABS)

    def _convert_xywh_to_xywha(self, x):
        return BoxMode.convert(x, BoxMode.XYWH_ABS, BoxMode.XYWHA_ABS)

    def test_box_convert_list(self):
        for tp in [list, tuple]:
            box = tp([5.0, 5.0, 10.0, 10.0])
            output = self._convert_xy_to_wh(box)
            self.assertIsInstance(output, tp)
            self.assertIsInstance(output[0], float)
            self.assertEqual(output, tp([5.0, 5.0, 5.0, 5.0]))

            with self.assertRaises(Exception):
                self._convert_xy_to_wh([box])

    def test_box_convert_array(self):
        box = np.asarray([[5, 5, 10, 10], [1, 1, 2, 3]])
        output = self._convert_xy_to_wh(box)
        self.assertEqual(output.dtype, box.dtype)
        self.assertEqual(output.shape, box.shape)
        self.assertTrue((output[0] == [5, 5, 5, 5]).all())
        self.assertTrue((output[1] == [1, 1, 1, 2]).all())

    def test_box_convert_cpu_tensor(self):
        box = torch.tensor([[5, 5, 10, 10], [1, 1, 2, 3]])
        output = self._convert_xy_to_wh(box)
        self.assertEqual(output.dtype, box.dtype)
        self.assertEqual(output.shape, box.shape)
        output = output.numpy()
        self.assertTrue((output[0] == [5, 5, 5, 5]).all())
        self.assertTrue((output[1] == [1, 1, 1, 2]).all())

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_box_convert_cuda_tensor(self):
        box = torch.tensor([[5, 5, 10, 10], [1, 1, 2, 3]]).cuda()
        output = self._convert_xy_to_wh(box)
        self.assertEqual(output.dtype, box.dtype)
        self.assertEqual(output.shape, box.shape)
        self.assertEqual(output.device, box.device)
        output = output.cpu().numpy()
        self.assertTrue((output[0] == [5, 5, 5, 5]).all())
        self.assertTrue((output[1] == [1, 1, 1, 2]).all())

    def test_box_convert_xywha_to_xyxy_list(self):
        for tp in [list, tuple]:
            box = tp([50, 50, 30, 20, 0])
            output = self._convert_xywha_to_xyxy(box)
            self.assertIsInstance(output, tp)
            self.assertEqual(output, tp([35, 40, 65, 60]))

            with self.assertRaises(Exception):
                self._convert_xywha_to_xyxy([box])

    def test_box_convert_xywha_to_xyxy_array(self):
        for dtype in [np.float64, np.float32]:
            box = np.asarray(
                [
                    [50, 50, 30, 20, 0],
                    [50, 50, 30, 20, 90],
                    [1, 1, math.sqrt(2), math.sqrt(2), -45],
                ],
                dtype=dtype,
            )
            output = self._convert_xywha_to_xyxy(box)
            self.assertEqual(output.dtype, box.dtype)
            expected = np.asarray([[35, 40, 65, 60], [40, 35, 60, 65], [0, 0, 2, 2]], dtype=dtype)
            self.assertTrue(np.allclose(output, expected, atol=1e-6), "output={}".format(output))

    def test_box_convert_xywha_to_xyxy_tensor(self):
        for dtype in [torch.float32, torch.float64]:
            box = torch.tensor(
                [
                    [50, 50, 30, 20, 0],
                    [50, 50, 30, 20, 90],
                    [1, 1, math.sqrt(2), math.sqrt(2), -45],
                ],
                dtype=dtype,
            )
            output = self._convert_xywha_to_xyxy(box)
            self.assertEqual(output.dtype, box.dtype)
            expected = torch.tensor([[35, 40, 65, 60], [40, 35, 60, 65], [0, 0, 2, 2]], dtype=dtype)

            self.assertTrue(torch.allclose(output, expected, atol=1e-6), "output={}".format(output))

    def test_box_convert_xywh_to_xywha_list(self):
        for tp in [list, tuple]:
            box = tp([50, 50, 30, 20])
            output = self._convert_xywh_to_xywha(box)
            self.assertIsInstance(output, tp)
            self.assertEqual(output, tp([65, 60, 30, 20, 0]))

            with self.assertRaises(Exception):
                self._convert_xywh_to_xywha([box])

    def test_box_convert_xywh_to_xywha_array(self):
        for dtype in [np.float64, np.float32]:
            box = np.asarray([[30, 40, 70, 60], [30, 40, 60, 70], [-1, -1, 2, 2]], dtype=dtype)
            output = self._convert_xywh_to_xywha(box)
            self.assertEqual(output.dtype, box.dtype)
            expected = np.asarray(
                [[65, 70, 70, 60, 0], [60, 75, 60, 70, 0], [0, 0, 2, 2, 0]], dtype=dtype
            )
            self.assertTrue(np.allclose(output, expected, atol=1e-6), "output={}".format(output))

    def test_box_convert_xywh_to_xywha_tensor(self):
        for dtype in [torch.float32, torch.float64]:
            box = torch.tensor([[30, 40, 70, 60], [30, 40, 60, 70], [-1, -1, 2, 2]], dtype=dtype)
            output = self._convert_xywh_to_xywha(box)
            self.assertEqual(output.dtype, box.dtype)
            expected = torch.tensor(
                [[65, 70, 70, 60, 0], [60, 75, 60, 70, 0], [0, 0, 2, 2, 0]], dtype=dtype
            )

            self.assertTrue(torch.allclose(output, expected, atol=1e-6), "output={}".format(output))

    def test_json_serializable(self):
        payload = {"box_mode": BoxMode.XYWH_REL}
        try:
            json.dumps(payload)
        except Exception:
            self.fail("JSON serialization failed")

    def test_json_deserializable(self):
        payload = '{"box_mode": 2}'
        obj = json.loads(payload)
        try:
            obj["box_mode"] = BoxMode(obj["box_mode"])
        except Exception:
            self.fail("JSON deserialization failed")


class TestBoxIOU(unittest.TestCase):
    def test_pairwise_iou(self):
        boxes1 = torch.tensor([[0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 1.0, 1.0]])

        boxes2 = torch.tensor(
            [
                [0.0, 0.0, 1.0, 1.0],
                [0.0, 0.0, 0.5, 1.0],
                [0.0, 0.0, 1.0, 0.5],
                [0.0, 0.0, 0.5, 0.5],
                [0.5, 0.5, 1.0, 1.0],
                [0.5, 0.5, 1.5, 1.5],
            ]
        )

        expected_ious = torch.tensor(
            [
                [1.0, 0.5, 0.5, 0.25, 0.25, 0.25 / (2 - 0.25)],
                [1.0, 0.5, 0.5, 0.25, 0.25, 0.25 / (2 - 0.25)],
            ]
        )

        ious = pairwise_iou(Boxes(boxes1), Boxes(boxes2))

        self.assertTrue(torch.allclose(ious, expected_ious))


if __name__ == "__main__":
    unittest.main()
