# Copyright (c) Facebook, Inc. and its affiliates.
import unittest
from collections import OrderedDict
import torch
from torch import nn

from detectron2.checkpoint.c2_model_loading import align_and_update_state_dicts
from detectron2.utils.logger import setup_logger


class TestCheckpointer(unittest.TestCase):
    def setUp(self):
        setup_logger()

    def create_complex_model(self):
        m = nn.Module()
        m.block1 = nn.Module()
        m.block1.layer1 = nn.Linear(2, 3)
        m.layer2 = nn.Linear(3, 2)
        m.res = nn.Module()
        m.res.layer2 = nn.Linear(3, 2)

        state_dict = OrderedDict()
        state_dict["layer1.weight"] = torch.rand(3, 2)
        state_dict["layer1.bias"] = torch.rand(3)
        state_dict["layer2.weight"] = torch.rand(2, 3)
        state_dict["layer2.bias"] = torch.rand(2)
        state_dict["res.layer2.weight"] = torch.rand(2, 3)
        state_dict["res.layer2.bias"] = torch.rand(2)
        return m, state_dict

    def test_complex_model_loaded(self):
        for add_data_parallel in [False, True]:
            model, state_dict = self.create_complex_model()
            if add_data_parallel:
                model = nn.DataParallel(model)
            model_sd = model.state_dict()

            align_and_update_state_dicts(model_sd, state_dict)
            for loaded, stored in zip(model_sd.values(), state_dict.values()):
                # different tensor references
                self.assertFalse(id(loaded) == id(stored))
                # same content
                self.assertTrue(loaded.equal(stored))


if __name__ == "__main__":
    unittest.main()
