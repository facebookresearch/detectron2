# Copyright (c) Facebook, Inc. and its affiliates.
import unittest
import torch

from detectron2.structures.masks import BitMasks, PolygonMasks, polygons_to_bitmask


class TestBitMask(unittest.TestCase):
    def test_get_bounding_box(self):
        masks = torch.tensor(
            [
                [
                    [False, False, False, True],
                    [False, False, True, True],
                    [False, True, True, False],
                    [False, True, True, False],
                ],
                [
                    [False, False, False, False],
                    [False, False, True, False],
                    [False, True, True, False],
                    [False, True, True, False],
                ],
                torch.zeros(4, 4),
            ]
        )
        bitmask = BitMasks(masks)
        box_true = torch.tensor([[1, 0, 4, 4], [1, 1, 3, 4], [0, 0, 0, 0]], dtype=torch.float32)
        box = bitmask.get_bounding_boxes()
        self.assertTrue(torch.all(box.tensor == box_true).item())

        for box in box_true:
            poly = box[[0, 1, 2, 1, 2, 3, 0, 3]].numpy()
            mask = polygons_to_bitmask([poly], 4, 4)
            reconstruct_box = BitMasks(mask[None, :, :]).get_bounding_boxes()[0].tensor
            self.assertTrue(torch.all(box == reconstruct_box).item())

            reconstruct_box = PolygonMasks([[poly]]).get_bounding_boxes()[0].tensor
            self.assertTrue(torch.all(box == reconstruct_box).item())

    def test_from_empty_polygons(self):
        masks = BitMasks.from_polygon_masks([], 100, 100)
        self.assertEqual(masks.tensor.shape, (0, 100, 100))

    def test_getitem(self):
        masks = BitMasks(torch.ones(3, 10, 10))
        self.assertEqual(masks[1].tensor.shape, (1, 10, 10))
        self.assertEqual(masks[1:3].tensor.shape, (2, 10, 10))
        self.assertEqual(masks[torch.tensor([True, False, False])].tensor.shape, (1, 10, 10))


if __name__ == "__main__":
    unittest.main()
