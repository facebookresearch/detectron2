# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import unittest
import torch

from detectron2.structures import Boxes, BoxMode, Instances

from densepose.modeling.losses.embed_utils import CseAnnotationsAccumulator
from densepose.structures import DensePoseDataRelative, DensePoseList


class TestCseAnnotationsAccumulator(unittest.TestCase):
    def test_cse_annotations_accumulator_nodp(self):
        instances_lst = [
            self._create_instances_nodp(),
        ]
        self._test_template(instances_lst)

    def test_cse_annotations_accumulator_sparsedp(self):
        instances_lst = [
            self._create_instances_sparsedp(),
        ]
        self._test_template(instances_lst)

    def test_cse_annotations_accumulator_fulldp(self):
        instances_lst = [
            self._create_instances_fulldp(),
        ]
        self._test_template(instances_lst)

    def test_cse_annotations_accumulator_combined(self):
        instances_lst = [
            self._create_instances_nodp(),
            self._create_instances_sparsedp(),
            self._create_instances_fulldp(),
        ]
        self._test_template(instances_lst)

    def _test_template(self, instances_lst):
        acc = CseAnnotationsAccumulator()
        for instances in instances_lst:
            acc.accumulate(instances)
        packed_anns = acc.pack()
        self._check_correspondence(packed_anns, instances_lst)

    def _create_instances_nodp(self):
        image_shape = (480, 640)
        instances = Instances(image_shape)
        instances.gt_boxes = Boxes(
            torch.as_tensor(
                [
                    [40.0, 40.0, 140.0, 140.0],
                    [160.0, 160.0, 270.0, 270.0],
                    [40.0, 160.0, 160.0, 280.0],
                ]
            )
        )
        instances.proposal_boxes = Boxes(
            torch.as_tensor(
                [
                    [41.0, 39.0, 142.0, 138.0],
                    [161.0, 159.0, 272.0, 268.0],
                    [41.0, 159.0, 162.0, 278.0],
                ]
            )
        )
        # do not add gt_densepose
        return instances

    def _create_instances_sparsedp(self):
        image_shape = (540, 720)
        instances = Instances(image_shape)
        instances.gt_boxes = Boxes(
            torch.as_tensor(
                [
                    [50.0, 50.0, 130.0, 130.0],
                    [150.0, 150.0, 240.0, 240.0],
                    [50.0, 150.0, 230.0, 330.0],
                ]
            )
        )
        instances.proposal_boxes = Boxes(
            torch.as_tensor(
                [
                    [49.0, 51.0, 131.0, 129.0],
                    [151.0, 149.0, 241.0, 239.0],
                    [51.0, 149.0, 232.0, 329.0],
                ]
            )
        )
        instances.gt_densepose = DensePoseList(
            [
                None,
                self._create_dp_data(
                    {
                        "dp_x": [81.69, 153.47, 151.00],
                        "dp_y": [162.24, 128.71, 113.81],
                        "dp_vertex": [0, 1, 2],
                        "ref_model": "zebra_5002",
                        "dp_masks": [],
                    },
                    {"c": (166, 133), "r": 64},
                ),
                None,
            ],
            instances.gt_boxes,
            image_shape,
        )
        return instances

    def _create_instances_fulldp(self):
        image_shape = (680, 840)
        instances = Instances(image_shape)
        instances.gt_boxes = Boxes(
            torch.as_tensor(
                [
                    [65.0, 55.0, 165.0, 155.0],
                    [170.0, 175.0, 275.0, 280.0],
                    [55.0, 165.0, 165.0, 275.0],
                ]
            )
        )
        instances.proposal_boxes = Boxes(
            torch.as_tensor(
                [
                    [66.0, 54.0, 166.0, 154.0],
                    [171.0, 174.0, 276.0, 279.0],
                    [56.0, 164.0, 166.0, 274.0],
                ]
            )
        )
        instances.gt_densepose = DensePoseList(
            [
                self._create_dp_data(
                    {
                        "dp_x": [149.99, 198.62, 157.59],
                        "dp_y": [170.74, 197.73, 123.12],
                        "dp_vertex": [3, 4, 5],
                        "ref_model": "cat_5001",
                        "dp_masks": [],
                    },
                    {"c": (100, 100), "r": 50},
                ),
                self._create_dp_data(
                    {
                        "dp_x": [234.53, 116.72, 71.66],
                        "dp_y": [107.53, 11.31, 142.32],
                        "dp_vertex": [6, 7, 8],
                        "ref_model": "dog_5002",
                        "dp_masks": [],
                    },
                    {"c": (200, 150), "r": 40},
                ),
                self._create_dp_data(
                    {
                        "dp_x": [225.54, 202.61, 135.90],
                        "dp_y": [167.46, 181.00, 211.47],
                        "dp_vertex": [9, 10, 11],
                        "ref_model": "elephant_5002",
                        "dp_masks": [],
                    },
                    {"c": (100, 200), "r": 45},
                ),
            ],
            instances.gt_boxes,
            image_shape,
        )
        return instances

    def _create_dp_data(self, anns, blob_def=None):
        dp_data = DensePoseDataRelative(anns)
        if blob_def is not None:
            dp_data.segm[
                blob_def["c"][0] - blob_def["r"] : blob_def["c"][0] + blob_def["r"],
                blob_def["c"][1] - blob_def["r"] : blob_def["c"][1] + blob_def["r"],
            ] = 1
        return dp_data

    def _check_correspondence(self, packed_anns, instances_lst):
        instance_idx = 0
        data_idx = 0
        pt_offset = 0
        if packed_anns is not None:
            bbox_xyxy_gt = BoxMode.convert(
                packed_anns.bbox_xywh_gt.clone(), BoxMode.XYWH_ABS, BoxMode.XYXY_ABS
            )
            bbox_xyxy_est = BoxMode.convert(
                packed_anns.bbox_xywh_est.clone(), BoxMode.XYWH_ABS, BoxMode.XYXY_ABS
            )
        for instances in instances_lst:
            if not hasattr(instances, "gt_densepose"):
                instance_idx += len(instances)
                continue
            for i, dp_data in enumerate(instances.gt_densepose):
                if dp_data is None:
                    instance_idx += 1
                    continue
                n_pts = len(dp_data.x)
                self.assertTrue(
                    torch.allclose(dp_data.x, packed_anns.x_gt[pt_offset : pt_offset + n_pts])
                )
                self.assertTrue(
                    torch.allclose(dp_data.y, packed_anns.y_gt[pt_offset : pt_offset + n_pts])
                )
                self.assertTrue(torch.allclose(dp_data.segm, packed_anns.coarse_segm_gt[data_idx]))
                self.assertTrue(
                    torch.allclose(
                        torch.ones(n_pts, dtype=torch.long) * dp_data.mesh_id,
                        packed_anns.vertex_mesh_ids_gt[pt_offset : pt_offset + n_pts],
                    )
                )
                self.assertTrue(
                    torch.allclose(
                        dp_data.vertex_ids, packed_anns.vertex_ids_gt[pt_offset : pt_offset + n_pts]
                    )
                )
                self.assertTrue(
                    torch.allclose(instances.gt_boxes.tensor[i], bbox_xyxy_gt[data_idx])
                )
                self.assertTrue(
                    torch.allclose(instances.proposal_boxes.tensor[i], bbox_xyxy_est[data_idx])
                )
                self.assertTrue(
                    torch.allclose(
                        torch.ones(n_pts, dtype=torch.long) * data_idx,
                        packed_anns.point_bbox_with_dp_indices[pt_offset : pt_offset + n_pts],
                    )
                )
                self.assertTrue(
                    torch.allclose(
                        torch.ones(n_pts, dtype=torch.long) * instance_idx,
                        packed_anns.point_bbox_indices[pt_offset : pt_offset + n_pts],
                    )
                )
                self.assertEqual(instance_idx, packed_anns.bbox_indices[data_idx])
                pt_offset += n_pts
                instance_idx += 1
                data_idx += 1
        if data_idx == 0:
            self.assertIsNone(packed_anns)
