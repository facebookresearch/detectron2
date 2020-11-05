# Copyright (c) Facebook, Inc. and its affiliates.

import random
import unittest
from typing import Any, Iterable, Iterator, Tuple

from densepose.data import CombinedDataLoader


def _grouper(iterable: Iterable[Any], n: int, fillvalue=None) -> Iterator[Tuple[Any]]:
    """
    Group elements of an iterable by chunks of size `n`, e.g.
    grouper(range(9), 4) ->
        (0, 1, 2, 3), (4, 5, 6, 7), (8, None, None, None)
    """
    it = iter(iterable)
    while True:
        values = []
        for _ in range(n):
            try:
                value = next(it)
            except StopIteration:
                values.extend([fillvalue] * (n - len(values)))
                yield tuple(values)
                return
            values.append(value)
        yield tuple(values)


class TestCombinedDataLoader(unittest.TestCase):
    def test_combine_loaders_1(self):
        loader1 = _grouper([f"1_{i}" for i in range(10)], 2)
        loader2 = _grouper([f"2_{i}" for i in range(11)], 3)
        batch_size = 4
        ratios = (0.1, 0.9)
        random.seed(43)
        combined = CombinedDataLoader((loader1, loader2), batch_size, ratios)
        BATCHES_GT = [
            ["1_0", "1_1", "2_0", "2_1"],
            ["2_2", "2_3", "2_4", "2_5"],
            ["1_2", "1_3", "2_6", "2_7"],
            ["2_8", "2_9", "2_10", None],
        ]
        for i, batch in enumerate(combined):
            self.assertEqual(len(batch), batch_size)
            self.assertEqual(batch, BATCHES_GT[i])
