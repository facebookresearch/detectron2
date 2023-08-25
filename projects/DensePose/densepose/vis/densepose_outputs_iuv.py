# Copyright (c) Facebook, Inc. and its affiliates.
import numpy as np
from typing import Optional, Tuple
import cv2

from densepose.structures import DensePoseDataRelative

from ..structures import DensePoseChartPredictorOutput
from .base import Boxes, Image, MatrixVisualizer


class DensePoseOutputsVisualizer:
    def __init__(
        self, inplace=True, cmap=cv2.COLORMAP_PARULA, alpha=0.7, to_visualize=None, **kwargs
    ):
        assert to_visualize in "IUV", "can only visualize IUV"
        self.to_visualize = to_visualize

        if self.to_visualize == "I":
            val_scale = 255.0 / DensePoseDataRelative.N_PART_LABELS
        else:
            val_scale = 1.0
        self.mask_visualizer = MatrixVisualizer(
            inplace=inplace, cmap=cmap, val_scale=val_scale, alpha=alpha
        )

    def visualize(
        self,
        image_bgr: Image,
        dp_output_with_bboxes: Tuple[Optional[DensePoseChartPredictorOutput], Optional[Boxes]],
    ) -> Image:
        densepose_output, bboxes_xywh = dp_output_with_bboxes
        if densepose_output is None or bboxes_xywh is None:
            return image_bgr

        assert isinstance(
            densepose_output, DensePoseChartPredictorOutput
        ), "DensePoseChartPredictorOutput expected, {} encountered".format(type(densepose_output))

        S = densepose_output.coarse_segm
        I = densepose_output.fine_segm  # noqa
        U = densepose_output.u
        V = densepose_output.v
        N = S.size(0)
        assert N == I.size(
            0
        ), "densepose outputs S {} and I {}" " should have equal first dim size".format(
            S.size(), I.size()
        )
        assert N == U.size(
            0
        ), "densepose outputs S {} and U {}" " should have equal first dim size".format(
            S.size(), U.size()
        )
        assert N == V.size(
            0
        ), "densepose outputs S {} and V {}" " should have equal first dim size".format(
            S.size(), V.size()
        )
        assert N == len(
            bboxes_xywh
        ), "number of bounding boxes {}" " should be equal to first dim size of outputs {}".format(
            len(bboxes_xywh), N
        )
        for n in range(N):
            Sn = S[n].argmax(dim=0)
            In = I[n].argmax(dim=0) * (Sn > 0).long()
            segmentation = In.cpu().numpy().astype(np.uint8)
            mask = np.zeros(segmentation.shape, dtype=np.uint8)
            mask[segmentation > 0] = 1
            bbox_xywh = bboxes_xywh[n]

            if self.to_visualize == "I":
                vis = segmentation
            elif self.to_visualize in "UV":
                U_or_Vn = {"U": U, "V": V}[self.to_visualize][n].cpu().numpy().astype(np.float32)
                vis = np.zeros(segmentation.shape, dtype=np.float32)
                for partId in range(U_or_Vn.shape[0]):
                    vis[segmentation == partId] = (
                        U_or_Vn[partId][segmentation == partId].clip(0, 1) * 255
                    )

            # pyre-fixme[61]: `vis` may not be initialized here.
            image_bgr = self.mask_visualizer.visualize(image_bgr, mask, vis, bbox_xywh)

        return image_bgr


class DensePoseOutputsUVisualizer(DensePoseOutputsVisualizer):
    def __init__(self, inplace=True, cmap=cv2.COLORMAP_PARULA, alpha=0.7, **kwargs):
        super().__init__(inplace=inplace, cmap=cmap, alpha=alpha, to_visualize="U", **kwargs)


class DensePoseOutputsVVisualizer(DensePoseOutputsVisualizer):
    def __init__(self, inplace=True, cmap=cv2.COLORMAP_PARULA, alpha=0.7, **kwargs):
        super().__init__(inplace=inplace, cmap=cmap, alpha=alpha, to_visualize="V", **kwargs)


class DensePoseOutputsFineSegmentationVisualizer(DensePoseOutputsVisualizer):
    def __init__(self, inplace=True, cmap=cv2.COLORMAP_PARULA, alpha=0.7, **kwargs):
        super().__init__(inplace=inplace, cmap=cmap, alpha=alpha, to_visualize="I", **kwargs)
