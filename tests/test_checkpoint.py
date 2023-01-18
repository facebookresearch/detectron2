# Copyright (c) Facebook, Inc. and its affiliates.
import os
import tempfile
import unittest
from collections import OrderedDict
import torch
from iopath.common.file_io import PathHandler, PathManager
from torch import nn

from detectron2.checkpoint import DetectionCheckpointer
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

            sd_to_load = align_and_update_state_dicts(model_sd, state_dict)
            model.load_state_dict(sd_to_load)
            for loaded, stored in zip(model_sd.values(), state_dict.values()):
                # different tensor references
                self.assertFalse(id(loaded) == id(stored))
                # same content
                self.assertTrue(loaded.to(stored).equal(stored))

    def test_load_with_matching_heuristics(self):
        with tempfile.TemporaryDirectory(prefix="detectron2_test") as d:
            model, state_dict = self.create_complex_model()
            torch.save({"model": state_dict}, os.path.join(d, "checkpoint.pth"))
            checkpointer = DetectionCheckpointer(model, save_dir=d)

            with torch.no_grad():
                # use a different weight from the `state_dict`, since torch.rand is less than 1
                model.block1.layer1.weight.fill_(1)

            # load checkpoint without matching_heuristics
            checkpointer.load(os.path.join(d, "checkpoint.pth"))
            self.assertTrue(model.block1.layer1.weight.equal(torch.ones(3, 2)))

            # load checkpoint with matching_heuristics
            checkpointer.load(os.path.join(d, "checkpoint.pth?matching_heuristics=True"))
            self.assertFalse(model.block1.layer1.weight.equal(torch.ones(3, 2)))

    def test_custom_path_manager_handler(self):
        with tempfile.TemporaryDirectory(prefix="detectron2_test") as d:

            class CustomPathManagerHandler(PathHandler):
                PREFIX = "detectron2_test://"

                def _get_supported_prefixes(self):
                    return [self.PREFIX]

                def _get_local_path(self, path, **kwargs):
                    name = path[len(self.PREFIX) :]
                    return os.path.join(d, name)

                def _open(self, path, mode="r", **kwargs):
                    return open(self._get_local_path(path), mode, **kwargs)

            pathmgr = PathManager()
            pathmgr.register_handler(CustomPathManagerHandler())

            model, state_dict = self.create_complex_model()
            torch.save({"model": state_dict}, os.path.join(d, "checkpoint.pth"))
            checkpointer = DetectionCheckpointer(model, save_dir=d)
            checkpointer.path_manager = pathmgr
            checkpointer.load("detectron2_test://checkpoint.pth")
            checkpointer.load("detectron2_test://checkpoint.pth?matching_heuristics=True")


if __name__ == "__main__":
    unittest.main()
