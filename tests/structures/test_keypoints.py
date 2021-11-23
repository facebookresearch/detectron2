# Copyright (c) Facebook, Inc. and its affiliates.
import unittest
import torch

from detectron2.structures.keypoints import Keypoints


class TestKeypoints(unittest.TestCase):
    def test_cat_keypoints(self):
        keypoints1 = Keypoints(torch.rand(2, 21, 3))
        keypoints2 = Keypoints(torch.rand(4, 21, 3))

        cat_keypoints = keypoints1.cat([keypoints1, keypoints2])
        self.assertTrue(torch.all(cat_keypoints.tensor[:2] == keypoints1.tensor).item())
        self.assertTrue(torch.all(cat_keypoints.tensor[2:] == keypoints2.tensor).item())


if __name__ == "__main__":
    unittest.main()
