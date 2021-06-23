# Copyright (c) Facebook, Inc. and its affiliates.
import unittest
import torch
from torch import Tensor

from detectron2.export.torchscript import patch_instances
from detectron2.structures import Boxes, Instances
from detectron2.utils.env import TORCH_VERSION
from detectron2.utils.testing import convert_scripted_instances


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

    def test_script_new_fields(self):
        def get_mask(x: Instances) -> torch.Tensor:
            return x.mask

        class f(torch.nn.Module):
            def forward(self, x: Instances):
                proposal_boxes = x.proposal_boxes  # noqa F841
                objectness_logits = x.objectness_logits  # noqa F841
                return x

        class g(torch.nn.Module):
            def forward(self, x: Instances):
                return get_mask(x)

        class g2(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.g = g()

            def forward(self, x: Instances):
                proposal_boxes = x.proposal_boxes  # noqa F841
                return x, self.g(x)

        fields = {"proposal_boxes": Boxes, "objectness_logits": Tensor}
        with patch_instances(fields):
            torch.jit.script(f())

        # can't script anymore after exiting the context
        with self.assertRaises(Exception):
            # will create a ConcreteType for g
            torch.jit.script(g2())

        new_fields = {"mask": Tensor}
        with patch_instances(new_fields):
            # will compile g with a different Instances; this should pass
            torch.jit.script(g())
            with self.assertRaises(Exception):
                torch.jit.script(g2())

        new_fields = {"mask": Tensor, "proposal_boxes": Boxes}
        with patch_instances(new_fields) as NewInstances:
            # get_mask will be compiled with a different Instances; this should pass
            scripted_g2 = torch.jit.script(g2())
            x = NewInstances((3, 4))
            x.mask = torch.rand(3)
            x.proposal_boxes = Boxes(torch.rand(3, 4))
            scripted_g2(x)  # it should accept the new Instances object and run successfully

    def test_script_access_fields(self):
        class f(torch.nn.Module):
            def forward(self, x: Instances):
                proposal_boxes = x.proposal_boxes
                objectness_logits = x.objectness_logits
                return proposal_boxes.tensor + objectness_logits

        fields = {"proposal_boxes": Boxes, "objectness_logits": Tensor}
        with patch_instances(fields):
            torch.jit.script(f())

    def test_script_len(self):
        class f(torch.nn.Module):
            def forward(self, x: Instances):
                return len(x)

        class g(torch.nn.Module):
            def forward(self, x: Instances):
                return len(x)

        image_shape = (15, 15)

        fields = {"proposal_boxes": Boxes}
        with patch_instances(fields) as new_instance:
            script_module = torch.jit.script(f())
            x = new_instance(image_shape)
            with self.assertRaises(Exception):
                script_module(x)
            box_tensors = torch.tensor([[5, 5, 10, 10], [1, 1, 2, 3]])
            x.proposal_boxes = Boxes(box_tensors)
            length = script_module(x)
            self.assertEqual(length, 2)

        fields = {"objectness_logits": Tensor}
        with patch_instances(fields) as new_instance:
            script_module = torch.jit.script(g())
            x = new_instance(image_shape)
            objectness_logits = torch.tensor([1.0]).reshape(1, 1)
            x.objectness_logits = objectness_logits
            length = script_module(x)
            self.assertEqual(length, 1)

    def test_script_has(self):
        class f(torch.nn.Module):
            def forward(self, x: Instances):
                return x.has("proposal_boxes")

        image_shape = (15, 15)
        fields = {"proposal_boxes": Boxes}
        with patch_instances(fields) as new_instance:
            script_module = torch.jit.script(f())
            x = new_instance(image_shape)
            self.assertFalse(script_module(x))

            box_tensors = torch.tensor([[5, 5, 10, 10], [1, 1, 2, 3]])
            x.proposal_boxes = Boxes(box_tensors)
            self.assertTrue(script_module(x))

    @unittest.skipIf(TORCH_VERSION < (1, 8), "Insufficient pytorch version")
    def test_script_to(self):
        class f(torch.nn.Module):
            def forward(self, x: Instances):
                return x.to(torch.device("cpu"))

        image_shape = (15, 15)
        fields = {"proposal_boxes": Boxes, "a": Tensor}
        with patch_instances(fields) as new_instance:
            script_module = torch.jit.script(f())
            x = new_instance(image_shape)
            script_module(x)

            box_tensors = torch.tensor([[5, 5, 10, 10], [1, 1, 2, 3]])
            x.proposal_boxes = Boxes(box_tensors)
            x.a = box_tensors
            script_module(x)

    def test_script_getitem(self):
        class f(torch.nn.Module):
            def forward(self, x: Instances, idx):
                return x[idx]

        image_shape = (15, 15)
        fields = {"proposal_boxes": Boxes, "a": Tensor}
        inst = Instances(image_shape)
        inst.proposal_boxes = Boxes(torch.rand(4, 4))
        inst.a = torch.rand(4, 10)
        idx = torch.tensor([True, False, True, False])
        with patch_instances(fields) as new_instance:
            script_module = torch.jit.script(f())

            out = f()(inst, idx)
            out_scripted = script_module(new_instance.from_instances(inst), idx)
            self.assertTrue(
                torch.equal(out.proposal_boxes.tensor, out_scripted.proposal_boxes.tensor)
            )
            self.assertTrue(torch.equal(out.a, out_scripted.a))

    def test_from_to_instances(self):
        orig = Instances((30, 30))
        orig.proposal_boxes = Boxes(torch.rand(3, 4))

        fields = {"proposal_boxes": Boxes, "a": Tensor}
        with patch_instances(fields) as NewInstances:
            # convert to NewInstances and back
            new1 = NewInstances.from_instances(orig)
            new2 = convert_scripted_instances(new1)
        self.assertTrue(torch.equal(orig.proposal_boxes.tensor, new1.proposal_boxes.tensor))
        self.assertTrue(torch.equal(orig.proposal_boxes.tensor, new2.proposal_boxes.tensor))


if __name__ == "__main__":
    unittest.main()
