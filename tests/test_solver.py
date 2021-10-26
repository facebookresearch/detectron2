import unittest

from detectron2.solver.build import _expand_param_groups, reduce_param_groups


class TestOptimizer(unittest.TestCase):
    def testExpandParamsGroups(self):
        params = [
            {
                "params": ["p1", "p2", "p3", "p4"],
                "lr": 1.0,
                "weight_decay": 3.0,
            },
            {
                "params": ["p2", "p3", "p5"],
                "lr": 2.0,
                "momentum": 2.0,
            },
            {
                "params": ["p1"],
                "weight_decay": 4.0,
            },
        ]
        out = _expand_param_groups(params)
        gt = [
            dict(params=["p1"], lr=1.0, weight_decay=4.0),  # noqa
            dict(params=["p2"], lr=2.0, weight_decay=3.0, momentum=2.0),  # noqa
            dict(params=["p3"], lr=2.0, weight_decay=3.0, momentum=2.0),  # noqa
            dict(params=["p4"], lr=1.0, weight_decay=3.0),  # noqa
            dict(params=["p5"], lr=2.0, momentum=2.0),  # noqa
        ]
        self.assertEqual(out, gt)

    def testReduceParamGroups(self):
        params = [
            dict(params=["p1"], lr=1.0, weight_decay=4.0),  # noqa
            dict(params=["p2", "p6"], lr=2.0, weight_decay=3.0, momentum=2.0),  # noqa
            dict(params=["p3"], lr=2.0, weight_decay=3.0, momentum=2.0),  # noqa
            dict(params=["p4"], lr=1.0, weight_decay=3.0),  # noqa
            dict(params=["p5"], lr=2.0, momentum=2.0),  # noqa
        ]
        gt_groups = [
            {
                "lr": 1.0,
                "weight_decay": 4.0,
                "params": ["p1"],
            },
            {
                "lr": 2.0,
                "weight_decay": 3.0,
                "momentum": 2.0,
                "params": ["p2", "p6", "p3"],
            },
            {
                "lr": 1.0,
                "weight_decay": 3.0,
                "params": ["p4"],
            },
            {
                "lr": 2.0,
                "momentum": 2.0,
                "params": ["p5"],
            },
        ]
        out = reduce_param_groups(params)
        self.assertEqual(out, gt_groups)
