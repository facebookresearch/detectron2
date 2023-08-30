# Copyright (c) Facebook, Inc. and its affiliates.
import logging
import numpy as np
from typing import List, Optional, Tuple
import cv2
import torch

from densepose.structures import DensePoseDataRelative

from ..structures import DensePoseChartResult
from .base import Boxes, Image, MatrixVisualizer


class DensePoseResultsVisualizer:
    def visualize(
        self,
        image_bgr: Image,
        results_and_boxes_xywh: Tuple[Optional[List[DensePoseChartResult]], Optional[Boxes]],
    ) -> Image:
        densepose_result, boxes_xywh = results_and_boxes_xywh
        if densepose_result is None or boxes_xywh is None:
            return image_bgr

        boxes_xywh = boxes_xywh.cpu().numpy()
        context = self.create_visualization_context(image_bgr)
        for i, result in enumerate(densepose_result):
            iuv_array = torch.cat(
                (result.labels[None].type(torch.float32), result.uv * 255.0)
            ).type(torch.uint8)
            self.visualize_iuv_arr(context, iuv_array.cpu().numpy(), boxes_xywh[i])
        image_bgr = self.context_to_image_bgr(context)
        return image_bgr

    def create_visualization_context(self, image_bgr: Image):
        return image_bgr

    def visualize_iuv_arr(self, context, iuv_arr: np.ndarray, bbox_xywh) -> None:
        pass

    def context_to_image_bgr(self, context):
        return context

    def get_image_bgr_from_context(self, context):
        return context


class DensePoseMaskedColormapResultsVisualizer(DensePoseResultsVisualizer):
    def __init__(
        self,
        data_extractor,
        segm_extractor,
        inplace=True,
        cmap=cv2.COLORMAP_PARULA,
        alpha=0.7,
        val_scale=1.0,
        **kwargs,
    ):
        self.mask_visualizer = MatrixVisualizer(
            inplace=inplace, cmap=cmap, val_scale=val_scale, alpha=alpha
        )
        self.data_extractor = data_extractor
        self.segm_extractor = segm_extractor

    def context_to_image_bgr(self, context):
        return context

    def visualize_iuv_arr(self, context, iuv_arr: np.ndarray, bbox_xywh) -> None:
        image_bgr = self.get_image_bgr_from_context(context)
        matrix = self.data_extractor(iuv_arr)
        segm = self.segm_extractor(iuv_arr)
        mask = np.zeros(matrix.shape, dtype=np.uint8)
        mask[segm > 0] = 1
        image_bgr = self.mask_visualizer.visualize(image_bgr, mask, matrix, bbox_xywh)


def _extract_i_from_iuvarr(iuv_arr):
    return iuv_arr[0, :, :]


def _extract_u_from_iuvarr(iuv_arr):
    return iuv_arr[1, :, :]


def _extract_v_from_iuvarr(iuv_arr):
    return iuv_arr[2, :, :]


class DensePoseResultsMplContourVisualizer(DensePoseResultsVisualizer):
    def __init__(self, levels=10, **kwargs):
        self.levels = levels
        self.plot_args = kwargs

    def create_visualization_context(self, image_bgr: Image):
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

        context = {}
        context["image_bgr"] = image_bgr
        dpi = 100
        height_inches = float(image_bgr.shape[0]) / dpi
        width_inches = float(image_bgr.shape[1]) / dpi
        fig = plt.figure(figsize=(width_inches, height_inches), dpi=dpi)
        plt.axes([0, 0, 1, 1])
        plt.axis("off")
        context["fig"] = fig
        canvas = FigureCanvas(fig)
        context["canvas"] = canvas
        extent = (0, image_bgr.shape[1], image_bgr.shape[0], 0)
        plt.imshow(image_bgr[:, :, ::-1], extent=extent)
        return context

    def context_to_image_bgr(self, context):
        fig = context["fig"]
        w, h = map(int, fig.get_size_inches() * fig.get_dpi())
        canvas = context["canvas"]
        canvas.draw()
        image_1d = np.fromstring(canvas.tostring_rgb(), dtype="uint8")
        image_rgb = image_1d.reshape(h, w, 3)
        image_bgr = image_rgb[:, :, ::-1].copy()
        return image_bgr

    def visualize_iuv_arr(self, context, iuv_arr: np.ndarray, bbox_xywh: Boxes) -> None:
        import matplotlib.pyplot as plt

        u = _extract_u_from_iuvarr(iuv_arr).astype(float) / 255.0
        v = _extract_v_from_iuvarr(iuv_arr).astype(float) / 255.0
        extent = (
            bbox_xywh[0],
            bbox_xywh[0] + bbox_xywh[2],
            bbox_xywh[1],
            bbox_xywh[1] + bbox_xywh[3],
        )
        plt.contour(u, self.levels, extent=extent, **self.plot_args)
        plt.contour(v, self.levels, extent=extent, **self.plot_args)


class DensePoseResultsCustomContourVisualizer(DensePoseResultsVisualizer):
    """
    Contour visualization using marching squares
    """

    def __init__(self, levels=10, **kwargs):
        # TODO: colormap is hardcoded
        cmap = cv2.COLORMAP_PARULA
        if isinstance(levels, int):
            self.levels = np.linspace(0, 1, levels)
        else:
            self.levels = levels
        if "linewidths" in kwargs:
            self.linewidths = kwargs["linewidths"]
        else:
            self.linewidths = [1] * len(self.levels)
        self.plot_args = kwargs
        img_colors_bgr = cv2.applyColorMap((self.levels * 255).astype(np.uint8), cmap)
        self.level_colors_bgr = [
            [int(v) for v in img_color_bgr.ravel()] for img_color_bgr in img_colors_bgr
        ]

    def visualize_iuv_arr(self, context, iuv_arr: np.ndarray, bbox_xywh: Boxes) -> None:
        image_bgr = self.get_image_bgr_from_context(context)
        segm = _extract_i_from_iuvarr(iuv_arr)
        u = _extract_u_from_iuvarr(iuv_arr).astype(float) / 255.0
        v = _extract_v_from_iuvarr(iuv_arr).astype(float) / 255.0
        self._contours(image_bgr, u, segm, bbox_xywh)
        self._contours(image_bgr, v, segm, bbox_xywh)

    def _contours(self, image_bgr, arr, segm, bbox_xywh):
        for part_idx in range(1, DensePoseDataRelative.N_PART_LABELS + 1):
            mask = segm == part_idx
            if not np.any(mask):
                continue
            arr_min = np.amin(arr[mask])
            arr_max = np.amax(arr[mask])
            I, J = np.nonzero(mask)
            i0 = np.amin(I)
            i1 = np.amax(I) + 1
            j0 = np.amin(J)
            j1 = np.amax(J) + 1
            if (j1 == j0 + 1) or (i1 == i0 + 1):
                continue
            Nw = arr.shape[1] - 1
            Nh = arr.shape[0] - 1
            for level_idx, level in enumerate(self.levels):
                if (level < arr_min) or (level > arr_max):
                    continue
                vp = arr[i0:i1, j0:j1] >= level
                bin_codes = vp[:-1, :-1] + vp[1:, :-1] * 2 + vp[1:, 1:] * 4 + vp[:-1, 1:] * 8
                mp = mask[i0:i1, j0:j1]
                bin_mask_codes = mp[:-1, :-1] + mp[1:, :-1] * 2 + mp[1:, 1:] * 4 + mp[:-1, 1:] * 8
                it = np.nditer(bin_codes, flags=["multi_index"])
                color_bgr = self.level_colors_bgr[level_idx]
                linewidth = self.linewidths[level_idx]
                while not it.finished:
                    if (it[0] != 0) and (it[0] != 15):
                        i, j = it.multi_index
                        if bin_mask_codes[i, j] != 0:
                            self._draw_line(
                                image_bgr,
                                arr,
                                mask,
                                level,
                                color_bgr,
                                linewidth,
                                it[0],
                                it.multi_index,
                                bbox_xywh,
                                Nw,
                                Nh,
                                (i0, j0),
                            )
                    it.iternext()

    def _draw_line(
        self,
        image_bgr,
        arr,
        mask,
        v,
        color_bgr,
        linewidth,
        bin_code,
        multi_idx,
        bbox_xywh,
        Nw,
        Nh,
        offset,
    ):
        lines = self._bin_code_2_lines(arr, v, bin_code, multi_idx, Nw, Nh, offset)
        x0, y0, w, h = bbox_xywh
        x1 = x0 + w
        y1 = y0 + h
        for line in lines:
            x0r, y0r = line[0]
            x1r, y1r = line[1]
            pt0 = (int(x0 + x0r * (x1 - x0)), int(y0 + y0r * (y1 - y0)))
            pt1 = (int(x0 + x1r * (x1 - x0)), int(y0 + y1r * (y1 - y0)))
            cv2.line(image_bgr, pt0, pt1, color_bgr, linewidth)

    def _bin_code_2_lines(self, arr, v, bin_code, multi_idx, Nw, Nh, offset):
        i0, j0 = offset
        i, j = multi_idx
        i += i0
        j += j0
        v0, v1, v2, v3 = arr[i, j], arr[i + 1, j], arr[i + 1, j + 1], arr[i, j + 1]
        x0i = float(j) / Nw
        y0j = float(i) / Nh
        He = 1.0 / Nh
        We = 1.0 / Nw
        if (bin_code == 1) or (bin_code == 14):
            a = (v - v0) / (v1 - v0)
            b = (v - v0) / (v3 - v0)
            pt1 = (x0i, y0j + a * He)
            pt2 = (x0i + b * We, y0j)
            return [(pt1, pt2)]
        elif (bin_code == 2) or (bin_code == 13):
            a = (v - v0) / (v1 - v0)
            b = (v - v1) / (v2 - v1)
            pt1 = (x0i, y0j + a * He)
            pt2 = (x0i + b * We, y0j + He)
            return [(pt1, pt2)]
        elif (bin_code == 3) or (bin_code == 12):
            a = (v - v0) / (v3 - v0)
            b = (v - v1) / (v2 - v1)
            pt1 = (x0i + a * We, y0j)
            pt2 = (x0i + b * We, y0j + He)
            return [(pt1, pt2)]
        elif (bin_code == 4) or (bin_code == 11):
            a = (v - v1) / (v2 - v1)
            b = (v - v3) / (v2 - v3)
            pt1 = (x0i + a * We, y0j + He)
            pt2 = (x0i + We, y0j + b * He)
            return [(pt1, pt2)]
        elif (bin_code == 6) or (bin_code == 9):
            a = (v - v0) / (v1 - v0)
            b = (v - v3) / (v2 - v3)
            pt1 = (x0i, y0j + a * He)
            pt2 = (x0i + We, y0j + b * He)
            return [(pt1, pt2)]
        elif (bin_code == 7) or (bin_code == 8):
            a = (v - v0) / (v3 - v0)
            b = (v - v3) / (v2 - v3)
            pt1 = (x0i + a * We, y0j)
            pt2 = (x0i + We, y0j + b * He)
            return [(pt1, pt2)]
        elif bin_code == 5:
            a1 = (v - v0) / (v1 - v0)
            b1 = (v - v1) / (v2 - v1)
            pt11 = (x0i, y0j + a1 * He)
            pt12 = (x0i + b1 * We, y0j + He)
            a2 = (v - v0) / (v3 - v0)
            b2 = (v - v3) / (v2 - v3)
            pt21 = (x0i + a2 * We, y0j)
            pt22 = (x0i + We, y0j + b2 * He)
            return [(pt11, pt12), (pt21, pt22)]
        elif bin_code == 10:
            a1 = (v - v0) / (v3 - v0)
            b1 = (v - v0) / (v1 - v0)
            pt11 = (x0i + a1 * We, y0j)
            pt12 = (x0i, y0j + b1 * He)
            a2 = (v - v1) / (v2 - v1)
            b2 = (v - v3) / (v2 - v3)
            pt21 = (x0i + a2 * We, y0j + He)
            pt22 = (x0i + We, y0j + b2 * He)
            return [(pt11, pt12), (pt21, pt22)]
        return []


try:
    import matplotlib

    matplotlib.use("Agg")
    DensePoseResultsContourVisualizer = DensePoseResultsMplContourVisualizer
except ModuleNotFoundError:
    logger = logging.getLogger(__name__)
    logger.warning("Could not import matplotlib, using custom contour visualizer")
    DensePoseResultsContourVisualizer = DensePoseResultsCustomContourVisualizer


class DensePoseResultsFineSegmentationVisualizer(DensePoseMaskedColormapResultsVisualizer):
    def __init__(self, inplace=True, cmap=cv2.COLORMAP_PARULA, alpha=0.7, **kwargs):
        super(DensePoseResultsFineSegmentationVisualizer, self).__init__(
            _extract_i_from_iuvarr,
            _extract_i_from_iuvarr,
            inplace,
            cmap,
            alpha,
            val_scale=255.0 / DensePoseDataRelative.N_PART_LABELS,
            **kwargs,
        )


class DensePoseResultsUVisualizer(DensePoseMaskedColormapResultsVisualizer):
    def __init__(self, inplace=True, cmap=cv2.COLORMAP_PARULA, alpha=0.7, **kwargs):
        super(DensePoseResultsUVisualizer, self).__init__(
            _extract_u_from_iuvarr,
            _extract_i_from_iuvarr,
            inplace,
            cmap,
            alpha,
            val_scale=1.0,
            **kwargs,
        )


class DensePoseResultsVVisualizer(DensePoseMaskedColormapResultsVisualizer):
    def __init__(self, inplace=True, cmap=cv2.COLORMAP_PARULA, alpha=0.7, **kwargs):
        super(DensePoseResultsVVisualizer, self).__init__(
            _extract_v_from_iuvarr,
            _extract_i_from_iuvarr,
            inplace,
            cmap,
            alpha,
            val_scale=1.0,
            **kwargs,
        )
