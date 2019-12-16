from unittest import TestCase

from utils import montage_previewer


class TestPreviewer(TestCase):
    def test_montage_previewer(self):
        file_path = "/home/wuyuanyi/nndata/nncrystal/validation/small extrapolation/extrapolation_20_30_1/images.bin"
        fig = montage_previewer(file_path, 5, 5)
        fig.show(True)

