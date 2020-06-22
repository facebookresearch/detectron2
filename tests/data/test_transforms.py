# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import logging
import numpy as np
import unittest
from unittest import mock

from detectron2.config import get_cfg
from detectron2.data import detection_utils
from detectron2.data import transforms as T
from detectron2.utils.logger import setup_logger

logger = logging.getLogger(__name__)


class TestTransforms(unittest.TestCase):
    def setUp(self):
        setup_logger()

    def test_apply_rotated_boxes(self):
        np.random.seed(125)
        cfg = get_cfg()
        is_train = True
        transform_gen = detection_utils.build_transform_gen(cfg, is_train)
        image = np.random.rand(200, 300)
        image, transforms = T.apply_transform_gens(transform_gen, image)
        image_shape = image.shape[:2]  # h, w
        assert image_shape == (800, 1200)
        annotation = {"bbox": [179, 97, 62, 40, -56]}

        boxes = np.array([annotation["bbox"]], dtype=np.float64)  # boxes.shape = (1, 5)
        transformed_bbox = transforms.apply_rotated_box(boxes)[0]

        expected_bbox = np.array([484, 388, 248, 160, 56], dtype=np.float64)
        err_msg = "transformed_bbox = {}, expected {}".format(transformed_bbox, expected_bbox)
        assert np.allclose(transformed_bbox, expected_bbox), err_msg

    def test_apply_rotated_boxes_unequal_scaling_factor(self):
        np.random.seed(125)
        h, w = 400, 200
        newh, neww = 800, 800
        image = np.random.rand(h, w)
        transform_gen = []
        transform_gen.append(T.Resize(shape=(newh, neww)))
        image, transforms = T.apply_transform_gens(transform_gen, image)
        image_shape = image.shape[:2]  # h, w
        assert image_shape == (newh, neww)

        boxes = np.array(
            [
                [150, 100, 40, 20, 0],
                [150, 100, 40, 20, 30],
                [150, 100, 40, 20, 90],
                [150, 100, 40, 20, -90],
            ],
            dtype=np.float64,
        )
        transformed_boxes = transforms.apply_rotated_box(boxes)

        expected_bboxes = np.array(
            [
                [600, 200, 160, 40, 0],
                [600, 200, 144.22205102, 52.91502622, 49.10660535],
                [600, 200, 80, 80, 90],
                [600, 200, 80, 80, -90],
            ],
            dtype=np.float64,
        )
        err_msg = "transformed_boxes = {}, expected {}".format(transformed_boxes, expected_bboxes)
        assert np.allclose(transformed_boxes, expected_bboxes), err_msg

    def test_print_transform_gen(self):
        t = T.RandomCrop("relative", (100, 100))
        self.assertTrue(str(t) == "RandomCrop(crop_type='relative', crop_size=(100, 100))")

        t = T.RandomFlip(prob=0.5)
        self.assertTrue(str(t) == "RandomFlip(prob=0.5)")

        t = T.RandomFlip()
        self.assertTrue(str(t) == "RandomFlip()")

    def test_random_apply_prob_out_of_range_check(self):
        # GIVEN
        test_probabilities = {0.0: True, 0.5: True, 1.0: True, -0.01: False, 1.01: False}

        # WHEN
        for given_probability, is_valid in test_probabilities.items():
            # THEN
            if not is_valid:
                self.assertRaises(AssertionError, T.RandomApply, None, prob=given_probability)
            else:
                T.RandomApply(T.NoOpTransform(), prob=given_probability)

    def test_random_apply_wrapping_transform_gen_probability_occured_evaluation(self):
        # GIVEN
        transform_mock = mock.MagicMock(name="MockTransform", spec=T.TransformGen)
        image_mock = mock.MagicMock(name="MockImage")
        random_apply = T.RandomApply(transform_mock, prob=0.001)

        # WHEN
        with mock.patch.object(random_apply, "_rand_range", return_value=0.0001):
            transform = random_apply.get_transform(image_mock)

        # THEN
        transform_mock.get_transform.assert_called_once_with(image_mock)
        self.assertIsNot(transform, transform_mock)

    def test_random_apply_wrapping_std_transform_probability_occured_evaluation(self):
        # GIVEN
        transform_mock = mock.MagicMock(name="MockTransform", spec=T.Transform)
        image_mock = mock.MagicMock(name="MockImage")
        random_apply = T.RandomApply(transform_mock, prob=0.001)

        # WHEN
        with mock.patch.object(random_apply, "_rand_range", return_value=0.0001):
            transform = random_apply.get_transform(image_mock)

        # THEN
        self.assertIs(transform, transform_mock)

    def test_random_apply_probability_not_occured_evaluation(self):
        # GIVEN
        transform_mock = mock.MagicMock(name="MockTransform", spec=T.TransformGen)
        image_mock = mock.MagicMock(name="MockImage")
        random_apply = T.RandomApply(transform_mock, prob=0.001)

        # WHEN
        with mock.patch.object(random_apply, "_rand_range", return_value=0.9):
            transform = random_apply.get_transform(image_mock)

        # THEN
        transform_mock.get_transform.assert_not_called()
        self.assertIsInstance(transform, T.NoOpTransform)
