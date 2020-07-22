# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import unittest
import torch

from detectron2.export.patch_instance import patch_instance
from detectron2.structures import Instances
from detectron2.utils.env import TORCH_VERSION


class TestInstancesIndexing(unittest.TestCase):
    def test_int_indexing(self):
        attr1 = torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 0.5], [0.0, 0.0, 1.0], [0.0, 0.5, 0.5]])
        attr2 = torch.tensor([0.1, 0.2, 0.3, 0.4])
        instances = Instances((100, 100))
        instances.attr1 = attr1
        instances.attr2 = attr2
        for i in range(-len(instances), len(instances)):
            inst = instances[i]
            self.assertEqual((inst.attr1 == attr1[i]).all(), True)
            self.assertEqual((inst.attr2 == attr2[i]).all(), True)

        self.assertRaises(IndexError, lambda: instances[len(instances)])
        self.assertRaises(IndexError, lambda: instances[-len(instances) - 1])


class TestInstancesScriptability(unittest.TestCase):
    @unittest.skipIf(TORCH_VERSION < (1, 6), "Insufficient pytorch version")
    def test_new_fields(self):
        fields = {"proposal_boxes": "Boxes", "objectness_logits": "Tensor"}

        class f(torch.nn.Module):
            def forward(self, x: Instances):
                proposal_boxes = x.proposal_boxes  # noqa F841
                objectness_logits = x.objectness_logits  # noqa F841
                return x

        class g(torch.nn.Module):
            def forward(self, x: Instances):
                mask = x.mask  # noqa F841
                return x

        class g2(torch.nn.Module):
            def forward(self, x: Instances):
                proposal_boxes = x.proposal_boxes  # noqa F841
                return x

        with patch_instance(fields):
            torch.jit.script(f())

        new_fields = {"mask": "Tensor"}
        with patch_instance(new_fields):
            torch.jit.script(g())
            with self.assertRaises(Exception):
                torch.jit.script(g2())


if __name__ == "__main__":
    unittest.main()
