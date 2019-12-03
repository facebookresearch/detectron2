import os
from unittest import TestCase

from utils.inference import get_data_loader, MsgpackFileDataset

dir_path = "/home/wuyuanyi/nndata/nncrystal/validation/small extrapolation/extrapolation_20_30_1/"
file_path = os.path.join(dir_path, "images.bin")

class TestDataLoader(TestCase):
    def test_get_data_loader(self):
        dataset = MsgpackFileDataset(file_path)
        loader = get_data_loader(dataset)

        for name, data in loader:
            print(name, data.shape)
            return
