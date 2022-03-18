# Copyright (c) Facebook, Inc. and its affiliates.

import unittest

from densepose.structures import normalized_coords_transform


class TestStructures(unittest.TestCase):
    def test_normalized_coords_transform(self):
        bbox = (32, 24, 288, 216)
        x0, y0, w, h = bbox
        xmin, ymin, xmax, ymax = x0, y0, x0 + w, y0 + h
        f = normalized_coords_transform(*bbox)
        # Top-left
        expected_p, actual_p = (-1, -1), f((xmin, ymin))
        self.assertEqual(expected_p, actual_p)
        # Top-right
        expected_p, actual_p = (1, -1), f((xmax, ymin))
        self.assertEqual(expected_p, actual_p)
        # Bottom-left
        expected_p, actual_p = (-1, 1), f((xmin, ymax))
        self.assertEqual(expected_p, actual_p)
        # Bottom-right
        expected_p, actual_p = (1, 1), f((xmax, ymax))
        self.assertEqual(expected_p, actual_p)
