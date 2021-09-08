# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.

import logging
import numpy as np
import unittest
from unittest import mock
import torch
from PIL import Image, ImageOps
from torch.nn import functional as F

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

    def test_resize_and_crop(self):
        np.random.seed(125)
        min_scale = 0.2
        max_scale = 2.0
        target_height = 1100
        target_width = 1000
        resize_aug = T.ResizeScale(min_scale, max_scale, target_height, target_width)
        fixed_size_crop_aug = T.FixedSizeCrop((target_height, target_width))
        hflip_aug = T.RandomFlip()
        augs = [resize_aug, fixed_size_crop_aug, hflip_aug]
        original_image = np.random.rand(900, 800)
        image, transforms = T.apply_augmentations(augs, original_image)
        image_shape = image.shape[:2]  # h, w
        self.assertEqual((1100, 1000), image_shape)

        boxes = np.array(
            [[91, 46, 144, 111], [523, 251, 614, 295]],
            dtype=np.float64,
        )
        transformed_bboxs = transforms.apply_box(boxes)
        expected_bboxs = np.array(
            [
                [895.42, 33.42666667, 933.91125, 80.66],
                [554.0825, 182.39333333, 620.17125, 214.36666667],
            ],
            dtype=np.float64,
        )
        err_msg = "transformed_bbox = {}, expected {}".format(transformed_bboxs, expected_bboxs)
        self.assertTrue(np.allclose(transformed_bboxs, expected_bboxs), err_msg)

        polygon = np.array([[91, 46], [144, 46], [144, 111], [91, 111]])
        transformed_polygons = transforms.apply_polygons([polygon])
        expected_polygon = np.array([[934.0, 33.0], [934.0, 80.0], [896.0, 80.0], [896.0, 33.0]])
        self.assertEqual(1, len(transformed_polygons))
        err_msg = "transformed_polygon = {}, expected {}".format(
            transformed_polygons[0], expected_polygon
        )
        self.assertTrue(np.allclose(transformed_polygons[0], expected_polygon), err_msg)

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
        self.assertEqual(str(t), "RandomCrop(crop_type='relative', crop_size=(100, 100))")

        t0 = T.RandomFlip(prob=0.5)
        self.assertEqual(str(t0), "RandomFlip(prob=0.5)")

        t1 = T.RandomFlip()
        self.assertEqual(str(t1), "RandomFlip()")

        t = T.AugmentationList([t0, t1])
        self.assertEqual(str(t), f"AugmentationList[{t0}, {t1}]")

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
        inputs = T.AugInput(image, sem_seg=sem_seg)  # provide two args
        tfms = inputs.apply_augmentations([TG1(), TG2()])
        self.assertIsInstance(tfms[0], T.ResizeTransform)
        self.assertIsInstance(tfms[1], T.HFlipTransform)
        self.assertTrue(inputs.image.shape[:2] == output_shape)
        self.assertTrue(inputs.sem_seg.shape[:2] == output_shape)

        class TG3(T.Augmentation):
            def get_transform(self, image, nonexist):
                pass

        with self.assertRaises(AttributeError):
            inputs.apply_augmentations([TG3()])

    def test_augmentation_list(self):
        input_shape = (100, 100)
        image = np.random.rand(*input_shape).astype("float32")
        sem_seg = (np.random.rand(*input_shape) < 0.5).astype("uint8")
        inputs = T.AugInput(image, sem_seg=sem_seg)  # provide two args

        augs = T.AugmentationList([T.RandomFlip(), T.Resize(20)])
        _ = T.AugmentationList([augs, T.Resize(30)])(inputs)
        # 3 in latest fvcore (flattened transformlist), 2 in older
        # self.assertEqual(len(tfms), 3)

    def test_color_transforms(self):
        rand_img = np.random.random((100, 100, 3)) * 255
        rand_img = rand_img.astype("uint8")

        # Test no-op
        noop_transform = T.ColorTransform(lambda img: img)
        self.assertTrue(np.array_equal(rand_img, noop_transform.apply_image(rand_img)))

        # Test a ImageOps operation
        magnitude = np.random.randint(0, 256)
        solarize_transform = T.PILColorTransform(lambda img: ImageOps.solarize(img, magnitude))
        expected_img = ImageOps.solarize(Image.fromarray(rand_img), magnitude)
        self.assertTrue(np.array_equal(expected_img, solarize_transform.apply_image(rand_img)))

    def test_resize_transform(self):
        input_shapes = [(100, 100), (100, 100, 1), (100, 100, 3)]
        output_shapes = [(200, 200), (200, 200, 1), (200, 200, 3)]
        for in_shape, out_shape in zip(input_shapes, output_shapes):
            in_img = np.random.randint(0, 255, size=in_shape, dtype=np.uint8)
            tfm = T.ResizeTransform(in_shape[0], in_shape[1], out_shape[0], out_shape[1])
            out_img = tfm.apply_image(in_img)
            self.assertEqual(out_img.shape, out_shape)

    def test_resize_shorted_edge_scriptable(self):
        def f(image):
            newh, neww = T.ResizeShortestEdge.get_output_shape(
                image.shape[-2], image.shape[-1], 80, 133
            )
            return F.interpolate(image.unsqueeze(0), size=(newh, neww))

        input = torch.randn(3, 10, 10)
        script_f = torch.jit.script(f)
        self.assertTrue(torch.allclose(f(input), script_f(input)))

        # generalize to new shapes
        input = torch.randn(3, 8, 100)
        self.assertTrue(torch.allclose(f(input), script_f(input)))

    def test_extent_transform(self):
        input_shapes = [(100, 100), (100, 100, 1), (100, 100, 3)]
        src_rect = (20, 20, 80, 80)
        output_shapes = [(200, 200), (200, 200, 1), (200, 200, 3)]
        for in_shape, out_shape in zip(input_shapes, output_shapes):
            in_img = np.random.randint(0, 255, size=in_shape, dtype=np.uint8)
            tfm = T.ExtentTransform(src_rect, out_shape[:2])
            out_img = tfm.apply_image(in_img)
            self.assertTrue(out_img.shape == out_shape)
