# Copyright (c) Facebook, Inc. and its affiliates.
import numpy as np
import unittest
import torch

from detectron2.layers import DeformConv, ModulatedDeformConv
from detectron2.utils.env import TORCH_VERSION


@unittest.skipIf(
    TORCH_VERSION == (1, 8) and torch.cuda.is_available(),
    "This test fails under cuda11 + torch1.8.",
)
class DeformableTest(unittest.TestCase):
    @unittest.skipIf(not torch.cuda.is_available(), "Deformable not supported for cpu")
    def test_forward_output(self):
        device = torch.device("cuda")
        N, C, H, W = shape = 1, 1, 5, 5
        kernel_size = 3
        padding = 1

        inputs = torch.arange(np.prod(shape), dtype=torch.float32).reshape(*shape).to(device)
        """
        0  1  2   3 4
        5  6  7   8 9
        10 11 12 13 14
        15 16 17 18 19
        20 21 22 23 24
        """
        offset_channels = kernel_size * kernel_size * 2
        offset = torch.full((N, offset_channels, H, W), 0.5, dtype=torch.float32).to(device)

        # Test DCN v1
        deform = DeformConv(C, C, kernel_size=kernel_size, padding=padding).to(device)
        deform.weight = torch.nn.Parameter(torch.ones_like(deform.weight))
        output = deform(inputs, offset)
        output = output.detach().cpu().numpy()
        deform_results = np.array(
            [
                [30, 41.25, 48.75, 45, 28.75],
                [62.25, 81, 90, 80.25, 50.25],
                [99.75, 126, 135, 117.75, 72.75],
                [105, 131.25, 138.75, 120, 73.75],
                [71.75, 89.25, 93.75, 80.75, 49.5],
            ]
        )
        self.assertTrue(np.allclose(output.flatten(), deform_results.flatten()))

        # Test DCN v2
        mask_channels = kernel_size * kernel_size
        mask = torch.full((N, mask_channels, H, W), 0.5, dtype=torch.float32).to(device)
        modulate_deform = ModulatedDeformConv(C, C, kernel_size, padding=padding, bias=False).to(
            device
        )
        modulate_deform.weight = deform.weight
        output = modulate_deform(inputs, offset, mask)
        output = output.detach().cpu().numpy()
        self.assertTrue(np.allclose(output.flatten(), deform_results.flatten() * 0.5))

    def test_forward_output_on_cpu(self):
        device = torch.device("cpu")
        N, C, H, W = shape = 1, 1, 5, 5
        kernel_size = 3
        padding = 1

        inputs = torch.arange(np.prod(shape), dtype=torch.float32).reshape(*shape).to(device)

        offset_channels = kernel_size * kernel_size * 2
        offset = torch.full((N, offset_channels, H, W), 0.5, dtype=torch.float32).to(device)

        # Test DCN v1 on cpu
        deform = DeformConv(C, C, kernel_size=kernel_size, padding=padding).to(device)
        deform.weight = torch.nn.Parameter(torch.ones_like(deform.weight))
        output = deform(inputs, offset)
        output = output.detach().cpu().numpy()
        deform_results = np.array(
            [
                [30, 41.25, 48.75, 45, 28.75],
                [62.25, 81, 90, 80.25, 50.25],
                [99.75, 126, 135, 117.75, 72.75],
                [105, 131.25, 138.75, 120, 73.75],
                [71.75, 89.25, 93.75, 80.75, 49.5],
            ]
        )
        self.assertTrue(np.allclose(output.flatten(), deform_results.flatten()))

    @unittest.skipIf(not torch.cuda.is_available(), "This test requires gpu access")
    def test_forward_output_on_cpu_equals_output_on_gpu(self):
        N, C, H, W = shape = 2, 4, 10, 10
        kernel_size = 3
        padding = 1

        for groups in [1, 2]:
            inputs = torch.arange(np.prod(shape), dtype=torch.float32).reshape(*shape)
            offset_channels = kernel_size * kernel_size * 2
            offset = torch.full((N, offset_channels, H, W), 0.5, dtype=torch.float32)

            deform_gpu = DeformConv(
                C, C, kernel_size=kernel_size, padding=padding, groups=groups
            ).to("cuda")
            deform_gpu.weight = torch.nn.Parameter(torch.ones_like(deform_gpu.weight))
            output_gpu = deform_gpu(inputs.to("cuda"), offset.to("cuda")).detach().cpu().numpy()

            deform_cpu = DeformConv(
                C, C, kernel_size=kernel_size, padding=padding, groups=groups
            ).to("cpu")
            deform_cpu.weight = torch.nn.Parameter(torch.ones_like(deform_cpu.weight))
            output_cpu = deform_cpu(inputs.to("cpu"), offset.to("cpu")).detach().numpy()

        self.assertTrue(np.allclose(output_gpu.flatten(), output_cpu.flatten()))

    @unittest.skipIf(not torch.cuda.is_available(), "Deformable not supported for cpu")
    def test_small_input(self):
        device = torch.device("cuda")
        for kernel_size in [3, 5]:
            padding = kernel_size // 2
            N, C, H, W = shape = (1, 1, kernel_size - 1, kernel_size - 1)

            inputs = torch.rand(shape).to(device)  # input size is smaller than kernel size

            offset_channels = kernel_size * kernel_size * 2
            offset = torch.randn((N, offset_channels, H, W), dtype=torch.float32).to(device)
            deform = DeformConv(C, C, kernel_size=kernel_size, padding=padding).to(device)
            output = deform(inputs, offset)
            self.assertTrue(output.shape == inputs.shape)

            mask_channels = kernel_size * kernel_size
            mask = torch.ones((N, mask_channels, H, W), dtype=torch.float32).to(device)
            modulate_deform = ModulatedDeformConv(
                C, C, kernel_size, padding=padding, bias=False
            ).to(device)
            output = modulate_deform(inputs, offset, mask)
            self.assertTrue(output.shape == inputs.shape)

    @unittest.skipIf(not torch.cuda.is_available(), "Deformable not supported for cpu")
    def test_raise_exception(self):
        device = torch.device("cuda")
        N, C, H, W = shape = 1, 1, 3, 3
        kernel_size = 3
        padding = 1

        inputs = torch.rand(shape, dtype=torch.float32).to(device)
        offset_channels = kernel_size * kernel_size  # This is wrong channels for offset
        offset = torch.randn((N, offset_channels, H, W), dtype=torch.float32).to(device)
        deform = DeformConv(C, C, kernel_size=kernel_size, padding=padding).to(device)
        self.assertRaises(RuntimeError, deform, inputs, offset)

        offset_channels = kernel_size * kernel_size * 2
        offset = torch.randn((N, offset_channels, H, W), dtype=torch.float32).to(device)
        mask_channels = kernel_size * kernel_size * 2  # This is wrong channels for mask
        mask = torch.ones((N, mask_channels, H, W), dtype=torch.float32).to(device)
        modulate_deform = ModulatedDeformConv(C, C, kernel_size, padding=padding, bias=False).to(
            device
        )
        self.assertRaises(RuntimeError, modulate_deform, inputs, offset, mask)

    def test_repr(self):
        module = DeformConv(3, 10, kernel_size=3, padding=1, deformable_groups=2)
        correct_string = (
            "DeformConv(in_channels=3, out_channels=10, kernel_size=(3, 3), "
            "stride=(1, 1), padding=(1, 1), dilation=(1, 1), "
            "groups=1, deformable_groups=2, bias=False)"
        )
        self.assertEqual(repr(module), correct_string)

        module = ModulatedDeformConv(3, 10, kernel_size=3, padding=1, deformable_groups=2)
        correct_string = (
            "ModulatedDeformConv(in_channels=3, out_channels=10, kernel_size=(3, 3), "
            "stride=1, padding=1, dilation=1, groups=1, deformable_groups=2, bias=True)"
        )
        self.assertEqual(repr(module), correct_string)


if __name__ == "__main__":
    unittest.main()
