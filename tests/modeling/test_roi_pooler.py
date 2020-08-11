# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import logging
import unittest
import torch

from detectron2.modeling.poolers import ROIPooler
from detectron2.structures import Boxes, RotatedBoxes
from detectron2.utils.env import TORCH_VERSION

logger = logging.getLogger(__name__)


class TestROIPooler(unittest.TestCase):
    def _rand_boxes(self, num_boxes, x_max, y_max):
        coords = torch.rand(num_boxes, 4)
        coords[:, 0] *= x_max
        coords[:, 1] *= y_max
        coords[:, 2] *= x_max
        coords[:, 3] *= y_max
        boxes = torch.zeros(num_boxes, 4)
        boxes[:, 0] = torch.min(coords[:, 0], coords[:, 2])
        boxes[:, 1] = torch.min(coords[:, 1], coords[:, 3])
        boxes[:, 2] = torch.max(coords[:, 0], coords[:, 2])
        boxes[:, 3] = torch.max(coords[:, 1], coords[:, 3])
        return boxes

    def _test_roialignv2_roialignrotated_match(self, device):
        pooler_resolution = 14
        canonical_level = 4
        canonical_scale_factor = 2 ** canonical_level
        pooler_scales = (1.0 / canonical_scale_factor,)
        sampling_ratio = 0

        N, C, H, W = 2, 4, 10, 8
        N_rois = 10
        std = 11
        mean = 0
        feature = (torch.rand(N, C, H, W) - 0.5) * 2 * std + mean

        features = [feature.to(device)]

        rois = []
        rois_rotated = []
        for _ in range(N):
            boxes = self._rand_boxes(
                num_boxes=N_rois, x_max=W * canonical_scale_factor, y_max=H * canonical_scale_factor
            )

            rotated_boxes = torch.zeros(N_rois, 5)
            rotated_boxes[:, 0] = (boxes[:, 0] + boxes[:, 2]) / 2.0
            rotated_boxes[:, 1] = (boxes[:, 1] + boxes[:, 3]) / 2.0
            rotated_boxes[:, 2] = boxes[:, 2] - boxes[:, 0]
            rotated_boxes[:, 3] = boxes[:, 3] - boxes[:, 1]
            rois.append(Boxes(boxes).to(device))
            rois_rotated.append(RotatedBoxes(rotated_boxes).to(device))

        roialignv2_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type="ROIAlignV2",
        )

        roialignv2_out = roialignv2_pooler(features, rois)

        roialignrotated_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type="ROIAlignRotated",
        )

        roialignrotated_out = roialignrotated_pooler(features, rois_rotated)

        self.assertTrue(torch.allclose(roialignv2_out, roialignrotated_out, atol=1e-4))

    def test_roialignv2_roialignrotated_match_cpu(self):
        self._test_roialignv2_roialignrotated_match(device="cpu")

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_roialignv2_roialignrotated_match_cuda(self):
        self._test_roialignv2_roialignrotated_match(device="cuda")

    def _test_scriptability(self, device):
        pooler_resolution = 14
        canonical_level = 4
        canonical_scale_factor = 2 ** canonical_level
        pooler_scales = (1.0 / canonical_scale_factor,)
        sampling_ratio = 0

        N, C, H, W = 2, 4, 10, 8
        N_rois = 10
        std = 11
        mean = 0
        feature = (torch.rand(N, C, H, W) - 0.5) * 2 * std + mean

        features = [feature.to(device)]

        rois = []
        for _ in range(N):
            boxes = self._rand_boxes(
                num_boxes=N_rois, x_max=W * canonical_scale_factor, y_max=H * canonical_scale_factor
            )

            rois.append(Boxes(boxes).to(device))

        roialignv2_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type="ROIAlignV2",
        )

        roialignv2_out = roialignv2_pooler(features, rois)
        scripted_roialignv2_out = torch.jit.script(roialignv2_pooler)(features, rois)
        self.assertTrue(torch.equal(roialignv2_out, scripted_roialignv2_out))

    @unittest.skipIf(TORCH_VERSION < (1, 7), "Insufficient pytorch version")
    def test_scriptability_cpu(self):
        self._test_scriptability(device="cpu")

    @unittest.skipIf(TORCH_VERSION < (1, 7), "Insufficient pytorch version")
    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_scriptability_gpu(self):
        self._test_scriptability(device="cuda")

    def test_no_images(self):
        N, C, H, W = 0, 32, 32, 32
        feature = torch.rand(N, C, H, W) - 0.5
        features = [feature]
        pooler = ROIPooler(
            output_size=14, scales=(1.0,), sampling_ratio=0.0, pooler_type="ROIAlignV2"
        )
        output = pooler.forward(features, [])
        self.assertEqual(output.shape, (0, C, 14, 14))


if __name__ == "__main__":
    unittest.main()
