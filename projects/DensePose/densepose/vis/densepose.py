# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import logging
import numpy as np
from typing import Iterable, Optional, Tuple
import cv2

from ..structures import DensePoseDataRelative, DensePoseOutput, DensePoseResult
from .base import Boxes, Image, MatrixVisualizer, PointsVisualizer


class DensePoseResultsVisualizer(object):
    def visualize(self, image_bgr: Image, densepose_result: Optional[DensePoseResult]) -> Image:
        if densepose_result is None:
            return image_bgr
        context = self.create_visualization_context(image_bgr)
        for i, result_encoded_w_shape in enumerate(densepose_result.results):
            iuv_arr = DensePoseResult.decode_png_data(*result_encoded_w_shape)
            bbox_xywh = densepose_result.boxes_xywh[i]
            self.visualize_iuv_arr(context, iuv_arr, bbox_xywh)
        image_bgr = self.context_to_image_bgr(context)
        return image_bgr


class DensePoseMaskedColormapResultsVisualizer(DensePoseResultsVisualizer):
    def __init__(
        self,
        data_extractor,
        segm_extractor,
        inplace=True,
        cmap=cv2.COLORMAP_PARULA,
        alpha=0.7,
        val_scale=1.0,
    ):
        self.mask_visualizer = MatrixVisualizer(
            inplace=inplace, cmap=cmap, val_scale=val_scale, alpha=alpha
        )
        self.data_extractor = data_extractor
        self.segm_extractor = segm_extractor

    def create_visualization_context(self, image_bgr: Image):
        return image_bgr

    def context_to_image_bgr(self, context):
        return context

    def get_image_bgr_from_context(self, context):
        return context

    def visualize_iuv_arr(self, context, iuv_arr, bbox_xywh):
        image_bgr = self.get_image_bgr_from_context(context)
        matrix = self.data_extractor(iuv_arr)
        segm = self.segm_extractor(iuv_arr)
        mask = np.zeros(matrix.shape, dtype=np.uint8)
        mask[segm > 0] = 1
        image_bgr = self.mask_visualizer.visualize(image_bgr, mask, matrix, bbox_xywh)
        return image_bgr


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

    def visualize_iuv_arr(self, context, iuv_arr: np.ndarray, bbox_xywh: Boxes) -> Image:
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

    def create_visualization_context(self, image_bgr: Image):
        return image_bgr

    def context_to_image_bgr(self, context):
        return context

    def get_image_bgr_from_context(self, context):
        return context

    def visualize_iuv_arr(self, context, iuv_arr: np.ndarray, bbox_xywh: Boxes) -> Image:
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
    def __init__(self, inplace=True, cmap=cv2.COLORMAP_PARULA, alpha=0.7):
        super(DensePoseResultsFineSegmentationVisualizer, self).__init__(
            _extract_i_from_iuvarr,
            _extract_i_from_iuvarr,
            inplace,
            cmap,
            alpha,
            val_scale=255.0 / DensePoseDataRelative.N_PART_LABELS,
        )


class DensePoseResultsUVisualizer(DensePoseMaskedColormapResultsVisualizer):
    def __init__(self, inplace=True, cmap=cv2.COLORMAP_PARULA, alpha=0.7):
        super(DensePoseResultsUVisualizer, self).__init__(
            _extract_u_from_iuvarr, _extract_i_from_iuvarr, inplace, cmap, alpha, val_scale=1.0
        )


class DensePoseResultsVVisualizer(DensePoseMaskedColormapResultsVisualizer):
    def __init__(self, inplace=True, cmap=cv2.COLORMAP_PARULA, alpha=0.7):
        super(DensePoseResultsVVisualizer, self).__init__(
            _extract_v_from_iuvarr, _extract_i_from_iuvarr, inplace, cmap, alpha, val_scale=1.0
        )


class DensePoseOutputsFineSegmentationVisualizer(object):
    def __init__(self, inplace=True, cmap=cv2.COLORMAP_PARULA, alpha=0.7):
        self.mask_visualizer = MatrixVisualizer(
            inplace=inplace,
            cmap=cmap,
            val_scale=255.0 / DensePoseDataRelative.N_PART_LABELS,
            alpha=alpha,
        )

    def visualize(
        self, image_bgr: Image, dp_output_with_bboxes: Optional[Tuple[DensePoseOutput, Boxes]]
    ) -> Image:
        if dp_output_with_bboxes is None:
            return image_bgr
        densepose_output, bboxes_xywh = dp_output_with_bboxes
        S = densepose_output.S
        I = densepose_output.I  # noqa
        U = densepose_output.U
        V = densepose_output.V
        N = S.size(0)
        assert N == I.size(0), (
            "densepose outputs S {} and I {}"
            " should have equal first dim size".format(S.size(), I.size())
        )
        assert N == U.size(0), (
            "densepose outputs S {} and U {}"
            " should have equal first dim size".format(S.size(), U.size())
        )
        assert N == V.size(0), (
            "densepose outputs S {} and V {}"
            " should have equal first dim size".format(S.size(), V.size())
        )
        assert N == len(bboxes_xywh), (
            "number of bounding boxes {}"
            " should be equal to first dim size of outputs {}".format(len(bboxes_xywh), N)
        )
        for n in range(N):
            Sn = S[n].argmax(dim=0)
            In = I[n].argmax(dim=0) * (Sn > 0).long()
            matrix = In.cpu().numpy().astype(np.uint8)
            mask = np.zeros(matrix.shape, dtype=np.uint8)
            mask[matrix > 0] = 1
            bbox_xywh = bboxes_xywh[n]
            image_bgr = self.mask_visualizer.visualize(image_bgr, mask, matrix, bbox_xywh)
        return image_bgr


class DensePoseOutputsUVisualizer(object):
    def __init__(self, inplace=True, cmap=cv2.COLORMAP_PARULA, alpha=0.7):
        self.mask_visualizer = MatrixVisualizer(
            inplace=inplace, cmap=cmap, val_scale=1.0, alpha=alpha
        )

    def visualize(
        self, image_bgr: Image, dp_output_with_bboxes: Optional[Tuple[DensePoseOutput, Boxes]]
    ) -> Image:
        if dp_output_with_bboxes is None:
            return image_bgr
        densepose_output, bboxes_xywh = dp_output_with_bboxes
        assert isinstance(
            densepose_output, DensePoseOutput
        ), "DensePoseOutput expected, {} encountered".format(type(densepose_output))
        S = densepose_output.S
        I = densepose_output.I  # noqa
        U = densepose_output.U
        V = densepose_output.V
        N = S.size(0)
        assert N == I.size(0), (
            "densepose outputs S {} and I {}"
            " should have equal first dim size".format(S.size(), I.size())
        )
        assert N == U.size(0), (
            "densepose outputs S {} and U {}"
            " should have equal first dim size".format(S.size(), U.size())
        )
        assert N == V.size(0), (
            "densepose outputs S {} and V {}"
            " should have equal first dim size".format(S.size(), V.size())
        )
        assert N == len(bboxes_xywh), (
            "number of bounding boxes {}"
            " should be equal to first dim size of outputs {}".format(len(bboxes_xywh), N)
        )
        for n in range(N):
            Sn = S[n].argmax(dim=0)
            In = I[n].argmax(dim=0) * (Sn > 0).long()
            segmentation = In.cpu().numpy().astype(np.uint8)
            mask = np.zeros(segmentation.shape, dtype=np.uint8)
            mask[segmentation > 0] = 1
            Un = U[n].cpu().numpy().astype(np.float32)
            Uvis = np.zeros(segmentation.shape, dtype=np.float32)
            for partId in range(Un.shape[0]):
                Uvis[segmentation == partId] = Un[partId][segmentation == partId].clip(0, 1) * 255
                bbox_xywh = bboxes_xywh[n]
            image_bgr = self.mask_visualizer.visualize(image_bgr, mask, Uvis, bbox_xywh)
        return image_bgr


class DensePoseOutputsVVisualizer(object):
    def __init__(self, inplace=True, cmap=cv2.COLORMAP_PARULA, alpha=0.7):
        self.mask_visualizer = MatrixVisualizer(
            inplace=inplace, cmap=cmap, val_scale=1.0, alpha=alpha
        )

    def visualize(
        self, image_bgr: Image, dp_output_with_bboxes: Optional[Tuple[DensePoseOutput, Boxes]]
    ) -> Image:
        if dp_output_with_bboxes is None:
            return image_bgr
        densepose_output, bboxes_xywh = dp_output_with_bboxes
        assert isinstance(
            densepose_output, DensePoseOutput
        ), "DensePoseOutput expected, {} encountered".format(type(densepose_output))
        S = densepose_output.S
        I = densepose_output.I  # noqa
        U = densepose_output.U
        V = densepose_output.V
        N = S.size(0)
        assert N == I.size(0), (
            "densepose outputs S {} and I {}"
            " should have equal first dim size".format(S.size(), I.size())
        )
        assert N == U.size(0), (
            "densepose outputs S {} and U {}"
            " should have equal first dim size".format(S.size(), U.size())
        )
        assert N == V.size(0), (
            "densepose outputs S {} and V {}"
            " should have equal first dim size".format(S.size(), V.size())
        )
        assert N == len(bboxes_xywh), (
            "number of bounding boxes {}"
            " should be equal to first dim size of outputs {}".format(len(bboxes_xywh), N)
        )
        for n in range(N):
            Sn = S[n].argmax(dim=0)
            In = I[n].argmax(dim=0) * (Sn > 0).long()
            segmentation = In.cpu().numpy().astype(np.uint8)
            mask = np.zeros(segmentation.shape, dtype=np.uint8)
            mask[segmentation > 0] = 1
            Vn = V[n].cpu().numpy().astype(np.float32)
            Vvis = np.zeros(segmentation.shape, dtype=np.float32)
            for partId in range(Vn.size(0)):
                Vvis[segmentation == partId] = Vn[partId][segmentation == partId].clip(0, 1) * 255
            bbox_xywh = bboxes_xywh[n]
            image_bgr = self.mask_visualizer.visualize(image_bgr, mask, Vvis, bbox_xywh)
        return image_bgr


class DensePoseDataCoarseSegmentationVisualizer(object):
    """
    Visualizer for ground truth segmentation
    """

    def __init__(self, inplace=True, cmap=cv2.COLORMAP_PARULA, alpha=0.7):
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
    def __init__(self, densepose_data_to_value_fn=None, cmap=cv2.COLORMAP_PARULA):
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
    def __init__(self):
        super(DensePoseDataPointsUVisualizer, self).__init__(
            densepose_data_to_value_fn=_densepose_data_u_for_cmap
        )


class DensePoseDataPointsVVisualizer(DensePoseDataPointsVisualizer):
    def __init__(self):
        super(DensePoseDataPointsVVisualizer, self).__init__(
            densepose_data_to_value_fn=_densepose_data_v_for_cmap
        )


class DensePoseDataPointsIVisualizer(DensePoseDataPointsVisualizer):
    def __init__(self):
        super(DensePoseDataPointsIVisualizer, self).__init__(
            densepose_data_to_value_fn=_densepose_data_i_for_cmap
        )
