# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import unittest
import torch

from detectron2.export.torchscript import patch_instances
from detectron2.structures import Boxes, Instances
from detectron2.utils.env import TORCH_VERSION


class TestInstances(unittest.TestCase):
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

    @unittest.skipIf(TORCH_VERSION < (1, 7), "Insufficient pytorch version")
    def test_script_new_fields(self):
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

        fields = {"proposal_boxes": "Boxes", "objectness_logits": "Tensor"}
        with patch_instances(fields):
            torch.jit.script(f())

        # can't script anymore after exiting the context
        with self.assertRaises(Exception):
            torch.jit.script(g2())

        new_fields = {"mask": "Tensor"}
        with patch_instances(new_fields):
            torch.jit.script(g())
            with self.assertRaises(Exception):
                torch.jit.script(g2())

    @unittest.skipIf(TORCH_VERSION < (1, 7), "Insufficient pytorch version")
    def test_script_access_fields(self):
        class f(torch.nn.Module):
            def forward(self, x: Instances):
                proposal_boxes = x.proposal_boxes
                objectness_logits = x.objectness_logits
                return proposal_boxes.tensor + objectness_logits

        fields = {"proposal_boxes": "Boxes", "objectness_logits": "Tensor"}
        with patch_instances(fields):
            torch.jit.script(f())

    @unittest.skipIf(TORCH_VERSION < (1, 7), "Insufficient pytorch version")
    def test_script_len(self):
        class f(torch.nn.Module):
            def forward(self, x: Instances):
                return len(x)

        class g(torch.nn.Module):
            def forward(self, x: Instances):
                return len(x)

        image_shape = (15, 15)

        fields = {"proposal_boxes": "Boxes"}
        with patch_instances(fields) as new_instance:
            script_module = torch.jit.script(f())
            x = new_instance(image_shape)
            with self.assertRaises(Exception):
                script_module(x)
            box_tensors = torch.tensor([[5, 5, 10, 10], [1, 1, 2, 3]])
            x.proposal_boxes = Boxes(box_tensors)
            length = script_module(x)
            self.assertEqual(length, 2)

        fields = {"objectness_logits": "Tensor"}
        with patch_instances(fields) as new_instance:
            script_module = torch.jit.script(g())
            x = new_instance(image_shape)
            objectness_logits = torch.tensor([1.0]).reshape(1, 1)
            x.objectness_logits = objectness_logits
            length = script_module(x)
            self.assertEqual(length, 1)

    @unittest.skipIf(TORCH_VERSION < (1, 7), "Insufficient pytorch version")
    def test_script_has(self):
        class f(torch.nn.Module):
            def forward(self, x: Instances):
                return x.has("proposal_boxes")

        image_shape = (15, 15)
        fields = {"proposal_boxes": "Boxes"}
        with patch_instances(fields) as new_instance:
            script_module = torch.jit.script(f())
            x = new_instance(image_shape)
            self.assertFalse(script_module(x))

            box_tensors = torch.tensor([[5, 5, 10, 10], [1, 1, 2, 3]])
            x.proposal_boxes = Boxes(box_tensors)
            self.assertTrue(script_module(x))


if __name__ == "__main__":
    unittest.main()
