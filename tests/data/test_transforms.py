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
        augs = detection_utils.build_augmentation(cfg, is_train)
        image = np.random.rand(200, 300)
        image, transforms = T.apply_augmentations(augs, image)
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
        augs = []
        augs.append(T.Resize(shape=(newh, neww)))
        image, transforms = T.apply_augmentations(augs, image)
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

    def test_print_augmentation(self):
        t = T.RandomCrop("relative", (100, 100))
        self.assertTrue(str(t) == "RandomCrop(crop_type='relative', crop_size=(100, 100))")

        t = T.RandomFlip(prob=0.5)
        self.assertTrue(str(t) == "RandomFlip(prob=0.5)")

        t = T.RandomFlip()
        self.assertTrue(str(t) == "RandomFlip()")

    def test_random_apply_prob_out_of_range_check(self):
        test_probabilities = {0.0: True, 0.5: True, 1.0: True, -0.01: False, 1.01: False}

        for given_probability, is_valid in test_probabilities.items():
            if not is_valid:
                self.assertRaises(AssertionError, T.RandomApply, None, prob=given_probability)
            else:
                T.RandomApply(T.NoOpTransform(), prob=given_probability)

    def test_random_apply_wrapping_aug_probability_occured_evaluation(self):
        transform_mock = mock.MagicMock(name="MockTransform", spec=T.Augmentation)
        image_mock = mock.MagicMock(name="MockImage")
        random_apply = T.RandomApply(transform_mock, prob=0.001)

        with mock.patch.object(random_apply, "_rand_range", return_value=0.0001):
            transform = random_apply.get_transform(image_mock)
        transform_mock.get_transform.assert_called_once_with(image_mock)
        self.assertIsNot(transform, transform_mock)

    def test_random_apply_wrapping_std_transform_probability_occured_evaluation(self):
        transform_mock = mock.MagicMock(name="MockTransform", spec=T.Transform)
        image_mock = mock.MagicMock(name="MockImage")
        random_apply = T.RandomApply(transform_mock, prob=0.001)

        with mock.patch.object(random_apply, "_rand_range", return_value=0.0001):
            transform = random_apply.get_transform(image_mock)
        self.assertIs(transform, transform_mock)

    def test_random_apply_probability_not_occured_evaluation(self):
        transform_mock = mock.MagicMock(name="MockTransform", spec=T.Augmentation)
        image_mock = mock.MagicMock(name="MockImage")
        random_apply = T.RandomApply(transform_mock, prob=0.001)

        with mock.patch.object(random_apply, "_rand_range", return_value=0.9):
            transform = random_apply.get_transform(image_mock)
        transform_mock.get_transform.assert_not_called()
        self.assertIsInstance(transform, T.NoOpTransform)

    def test_augmentation_input_args(self):
        input_shape = (100, 100)
        output_shape = (50, 50)

        # define two augmentations with different args
        class TG1(T.Augmentation):
            input_args = ("image", "sem_seg")

            def get_transform(self, image, sem_seg):
                return T.ResizeTransform(
                    input_shape[0], input_shape[1], output_shape[0], output_shape[1]
                )

        class TG2(T.Augmentation):
            def get_transform(self, image):
                assert image.shape[:2] == output_shape  # check that TG1 is applied
                return T.HFlipTransform(output_shape[1])

        image = np.random.rand(*input_shape).astype("float32")
        sem_seg = (np.random.rand(*input_shape) < 0.5).astype("uint8")
        inputs = T.StandardAugInput(image, sem_seg=sem_seg)  # provide two args
        tfms = inputs.apply_augmentations([TG1(), TG2()])
        self.assertIsInstance(tfms[0], T.ResizeTransform)
        self.assertIsInstance(tfms[1], T.HFlipTransform)
        self.assertTrue(inputs.image.shape[:2] == output_shape)
        self.assertTrue(inputs.sem_seg.shape[:2] == output_shape)

        class TG3(T.Augmentation):
            input_args = ("image", "nonexist")

            def get_transform(self, image, nonexist):
                pass

        with self.assertRaises(AttributeError):
            inputs.apply_augmentations([TG3()])
