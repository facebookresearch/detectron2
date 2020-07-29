import unittest
import torch

from detectron2.structures.masks import BitMasks


class TestBitMask(unittest.TestCase):
    def test_get_bounding_box(self):
        mask = torch.tensor(
            [
                [
                    [False, False, False, True],
                    [False, False, True, True],
                    [False, True, True, False],
                    [False, True, True, False],
                ],
                torch.zeros(4, 4),
            ]
        )
        bitmask = BitMasks(mask)
        box_true = torch.tensor([[1, 0, 4, 4], [0, 0, 0, 0]], dtype=torch.float32)
        box = bitmask.get_bounding_boxes()
        self.assertTrue(torch.all(box.tensor == box_true))


if __name__ == "__main__":
    unittest.main()
