# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved


def add_point_sup_config(cfg):
    """
    Add config for point supervision.
    """
    # Use point annotation
    cfg.INPUT.POINT_SUP = False
    # Sample only part of points in each iteration.
    # Default: 0, use all available points.
    cfg.INPUT.SAMPLE_POINTS = 0
