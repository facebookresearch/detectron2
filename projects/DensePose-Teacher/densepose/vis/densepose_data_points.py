# Copyright (c) Facebook, Inc. and its affiliates.
import numpy as np
from typing import Iterable, Optional, Tuple
import cv2

from densepose.structures import DensePoseDataRelative

from .base import Boxes, Image, MatrixVisualizer, PointsVisualizer


class DensePoseDataCoarseSegmentationVisualizer(object):
    """
    Visualizer for ground truth segmentation
    """

    def __init__(self, inplace=True, cmap=cv2.COLORMAP_PARULA, alpha=0.7, **kwargs):
        self.mask_visualizer = MatrixVisualizer(
            inplace=inplace,
            cmap=cmap,
            val_scale=255.0 / DensePoseDataRelative.N_BODY_PARTS,
            alpha=alpha,
        )

    def visualize(
        self,
        image_bgr: Image,
        bbox_densepose_datas: Optional[Tuple[Iterable[Boxes], Iterable[DensePoseDataRelative]]],
    ) -> Image:
        if bbox_densepose_datas is None:
            return image_bgr
        for bbox_xywh, densepose_data in zip(*bbox_densepose_datas):
            matrix = densepose_data.segm.numpy()
            mask = np.zeros(matrix.shape, dtype=np.uint8)
            mask[matrix > 0] = 1
            image_bgr = self.mask_visualizer.visualize(image_bgr, mask, matrix, bbox_xywh.numpy())
        return image_bgr


class DensePoseDataPointsVisualizer(object):
    def __init__(self, densepose_data_to_value_fn=None, cmap=cv2.COLORMAP_PARULA, **kwargs):
        self.points_visualizer = PointsVisualizer()
        self.densepose_data_to_value_fn = densepose_data_to_value_fn
        self.cmap = cmap

    def visualize(
        self,
        image_bgr: Image,
        bbox_densepose_datas: Optional[Tuple[Iterable[Boxes], Iterable[DensePoseDataRelative]]],
    ) -> Image:
        if bbox_densepose_datas is None:
            return image_bgr
        for bbox_xywh, densepose_data in zip(*bbox_densepose_datas):
            x0, y0, w, h = bbox_xywh.numpy()
            x = densepose_data.x.numpy() * w / 255.0 + x0
            y = densepose_data.y.numpy() * h / 255.0 + y0
            pts_xy = zip(x, y)
            if self.densepose_data_to_value_fn is None:
                image_bgr = self.points_visualizer.visualize(image_bgr, pts_xy)
            else:
                v = self.densepose_data_to_value_fn(densepose_data)
                img_colors_bgr = cv2.applyColorMap(v, self.cmap)
                colors_bgr = [
                    [int(v) for v in img_color_bgr.ravel()] for img_color_bgr in img_colors_bgr
                ]
                image_bgr = self.points_visualizer.visualize(image_bgr, pts_xy, colors_bgr)
        return image_bgr


def _densepose_data_u_for_cmap(densepose_data):
    u = np.clip(densepose_data.u.numpy(), 0, 1) * 255.0
    return u.astype(np.uint8)


def _densepose_data_v_for_cmap(densepose_data):
    v = np.clip(densepose_data.v.numpy(), 0, 1) * 255.0
    return v.astype(np.uint8)


def _densepose_data_i_for_cmap(densepose_data):
    i = (
        np.clip(densepose_data.i.numpy(), 0.0, DensePoseDataRelative.N_PART_LABELS)
        * 255.0
        / DensePoseDataRelative.N_PART_LABELS
    )
    return i.astype(np.uint8)


class DensePoseDataPointsUVisualizer(DensePoseDataPointsVisualizer):
    def __init__(self, **kwargs):
        super(DensePoseDataPointsUVisualizer, self).__init__(
            densepose_data_to_value_fn=_densepose_data_u_for_cmap, **kwargs
        )


class DensePoseDataPointsVVisualizer(DensePoseDataPointsVisualizer):
    def __init__(self, **kwargs):
        super(DensePoseDataPointsVVisualizer, self).__init__(
            densepose_data_to_value_fn=_densepose_data_v_for_cmap, **kwargs
        )


class DensePoseDataPointsIVisualizer(DensePoseDataPointsVisualizer):
    def __init__(self, **kwargs):
        super(DensePoseDataPointsIVisualizer, self).__init__(
            densepose_data_to_value_fn=_densepose_data_i_for_cmap, **kwargs
        )
