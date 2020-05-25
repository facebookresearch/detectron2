# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import copy
import numpy as np
import unittest
import pycocotools.mask as mask_util

from detectron2.data import detection_utils
from detectron2.data import transforms as T
from detectron2.structures import BitMasks, BoxMode


class TestTransformAnnotations(unittest.TestCase):
    def test_transform_simple_annotation(self):
        transforms = T.TransformList([T.HFlipTransform(400)])
        anno = {
            "bbox": np.asarray([10, 10, 200, 300]),
            "bbox_mode": BoxMode.XYXY_ABS,
            "category_id": 3,
            "segmentation": [[10, 10, 100, 100, 100, 10], [150, 150, 200, 150, 200, 200]],
        }

        output = detection_utils.transform_instance_annotations(anno, transforms, (400, 400))
        self.assertTrue(np.allclose(output["bbox"], [200, 10, 390, 300]))
        self.assertEqual(len(output["segmentation"]), len(anno["segmentation"]))
        self.assertTrue(np.allclose(output["segmentation"][0], [390, 10, 300, 100, 300, 10]))

        detection_utils.annotations_to_instances([output, output], (400, 400))

    def test_flip_keypoints(self):
        transforms = T.TransformList([T.HFlipTransform(400)])
        anno = {
            "bbox": np.asarray([10, 10, 200, 300]),
            "bbox_mode": BoxMode.XYXY_ABS,
            "keypoints": np.random.rand(17, 3) * 50 + 15,
        }

        output = detection_utils.transform_instance_annotations(
            copy.deepcopy(anno),
            transforms,
            (400, 400),
            keypoint_hflip_indices=detection_utils.create_keypoint_hflip_indices(
                ["keypoints_coco_2017_train"]
            ),
        )
        # The first keypoint is nose
        self.assertTrue(np.allclose(output["keypoints"][0, 0], 400 - anno["keypoints"][0, 0]))
        # The last 16 keypoints are 8 left-right pairs
        self.assertTrue(
            np.allclose(
                output["keypoints"][1:, 0].reshape(-1, 2)[:, ::-1],
                400 - anno["keypoints"][1:, 0].reshape(-1, 2),
            )
        )
        self.assertTrue(
            np.allclose(
                output["keypoints"][1:, 1:].reshape(-1, 2, 2)[:, ::-1, :],
                anno["keypoints"][1:, 1:].reshape(-1, 2, 2),
            )
        )

    def test_transform_RLE(self):
        transforms = T.TransformList([T.HFlipTransform(400)])
        mask = np.zeros((300, 400), order="F").astype("uint8")
        mask[:, :200] = 1

        anno = {
            "bbox": np.asarray([10, 10, 200, 300]),
            "bbox_mode": BoxMode.XYXY_ABS,
            "segmentation": mask_util.encode(mask[:, :, None])[0],
            "category_id": 3,
        }
        output = detection_utils.transform_instance_annotations(
            copy.deepcopy(anno), transforms, (300, 400)
        )
        mask = output["segmentation"]
        self.assertTrue((mask[:, 200:] == 1).all())
        self.assertTrue((mask[:, :200] == 0).all())

        inst = detection_utils.annotations_to_instances(
            [output, output], (400, 400), mask_format="bitmask"
        )
        self.assertTrue(isinstance(inst.gt_masks, BitMasks))

    def test_transform_RLE_resize(self):
        transforms = T.TransformList(
            [T.HFlipTransform(400), T.ScaleTransform(300, 400, 400, 400, "bilinear")]
        )
        mask = np.zeros((300, 400), order="F").astype("uint8")
        mask[:, :200] = 1

        anno = {
            "bbox": np.asarray([10, 10, 200, 300]),
            "bbox_mode": BoxMode.XYXY_ABS,
            "segmentation": mask_util.encode(mask[:, :, None])[0],
            "category_id": 3,
        }
        output = detection_utils.transform_instance_annotations(
            copy.deepcopy(anno), transforms, (400, 400)
        )

        inst = detection_utils.annotations_to_instances(
            [output, output], (400, 400), mask_format="bitmask"
        )
        self.assertTrue(isinstance(inst.gt_masks, BitMasks))

    def test_gen_crop(self):
        instance = {"bbox": [10, 10, 100, 100], "bbox_mode": BoxMode.XYXY_ABS}
        t = detection_utils.gen_crop_transform_with_instance((10, 10), (150, 150), instance)
        # the box center must fall into the cropped region
        self.assertTrue(t.x0 <= 55 <= t.x0 + t.w)

    def test_gen_crop_outside_boxes(self):
        instance = {"bbox": [10, 10, 100, 100], "bbox_mode": BoxMode.XYXY_ABS}
        with self.assertRaises(AssertionError):
            detection_utils.gen_crop_transform_with_instance((10, 10), (15, 15), instance)
