# Copyright (c) Facebook, Inc. and its affiliates.
import logging
import numpy as np
import cv2
import torch

Image = np.ndarray
Boxes = torch.Tensor


class MatrixVisualizer(object):
    """
    Base visualizer for matrix data
    """

    def __init__(
        self,
        inplace=True,
        cmap=cv2.COLORMAP_PARULA,
        val_scale=1.0,
        alpha=0.7,
        interp_method_matrix=cv2.INTER_LINEAR,
        interp_method_mask=cv2.INTER_NEAREST,
    ):
        self.inplace = inplace
        self.cmap = cmap
        self.val_scale = val_scale
        self.alpha = alpha
        self.interp_method_matrix = interp_method_matrix
        self.interp_method_mask = interp_method_mask

    def visualize(self, image_bgr, mask, matrix, bbox_xywh):
        self._check_image(image_bgr)
        self._check_mask_matrix(mask, matrix)
        if self.inplace:
            image_target_bgr = image_bgr
        else:
            image_target_bgr = image_bgr * 0
        x, y, w, h = [int(v) for v in bbox_xywh]
        if w <= 0 or h <= 0:
            return image_bgr
        mask, matrix = self._resize(mask, matrix, w, h)
        mask_bg = np.tile((mask == 0)[:, :, np.newaxis], [1, 1, 3])
        matrix_scaled = matrix.astype(np.float32) * self.val_scale
        _EPSILON = 1e-6
        if np.any(matrix_scaled > 255 + _EPSILON):
            logger = logging.getLogger(__name__)
            logger.warning(
                f"Matrix has values > {255 + _EPSILON} after " f"scaling, clipping to [0..255]"
            )
        matrix_scaled_8u = matrix_scaled.clip(0, 255).astype(np.uint8)
        matrix_vis = cv2.applyColorMap(matrix_scaled_8u, self.cmap)
        matrix_vis[mask_bg] = image_target_bgr[y : y + h, x : x + w, :][mask_bg]
        image_target_bgr[y : y + h, x : x + w, :] = (
            image_target_bgr[y : y + h, x : x + w, :] * (1.0 - self.alpha) + matrix_vis * self.alpha
        )
        return image_target_bgr.astype(np.uint8)

    def _resize(self, mask, matrix, w, h):
        if (w != mask.shape[1]) or (h != mask.shape[0]):
            mask = cv2.resize(mask, (w, h), self.interp_method_mask)
        if (w != matrix.shape[1]) or (h != matrix.shape[0]):
            matrix = cv2.resize(matrix, (w, h), self.interp_method_matrix)
        return mask, matrix

    def _check_image(self, image_rgb):
        assert len(image_rgb.shape) == 3
        assert image_rgb.shape[2] == 3
        assert image_rgb.dtype == np.uint8

    def _check_mask_matrix(self, mask, matrix):
        assert len(matrix.shape) == 2
        assert len(mask.shape) == 2
        assert mask.dtype == np.uint8


class RectangleVisualizer(object):

    _COLOR_GREEN = (18, 127, 15)

    def __init__(self, color=_COLOR_GREEN, thickness=1):
        self.color = color
        self.thickness = thickness

    def visualize(self, image_bgr, bbox_xywh, color=None, thickness=None):
        x, y, w, h = bbox_xywh
        color = color or self.color
        thickness = thickness or self.thickness
        cv2.rectangle(image_bgr, (int(x), int(y)), (int(x + w), int(y + h)), color, thickness)
        return image_bgr


class PointsVisualizer(object):

    _COLOR_GREEN = (18, 127, 15)

    def __init__(self, color_bgr=_COLOR_GREEN, r=5):
        self.color_bgr = color_bgr
        self.r = r

    def visualize(self, image_bgr, pts_xy, colors_bgr=None, rs=None):
        for j, pt_xy in enumerate(pts_xy):
            x, y = pt_xy
            color_bgr = colors_bgr[j] if colors_bgr is not None else self.color_bgr
            r = rs[j] if rs is not None else self.r
            cv2.circle(image_bgr, (x, y), r, color_bgr, -1)
        return image_bgr


class TextVisualizer(object):

    _COLOR_GRAY = (218, 227, 218)
    _COLOR_WHITE = (255, 255, 255)

    def __init__(
        self,
        font_face=cv2.FONT_HERSHEY_SIMPLEX,
        font_color_bgr=_COLOR_GRAY,
        font_scale=0.35,
        font_line_type=cv2.LINE_AA,
        font_line_thickness=1,
        fill_color_bgr=_COLOR_WHITE,
        fill_color_transparency=1.0,
        frame_color_bgr=_COLOR_WHITE,
        frame_color_transparency=1.0,
        frame_thickness=1,
    ):
        self.font_face = font_face
        self.font_color_bgr = font_color_bgr
        self.font_scale = font_scale
        self.font_line_type = font_line_type
        self.font_line_thickness = font_line_thickness
        self.fill_color_bgr = fill_color_bgr
        self.fill_color_transparency = fill_color_transparency
        self.frame_color_bgr = frame_color_bgr
        self.frame_color_transparency = frame_color_transparency
        self.frame_thickness = frame_thickness

    def visualize(self, image_bgr, txt, topleft_xy):
        txt_w, txt_h = self.get_text_size_wh(txt)
        topleft_xy = tuple(map(int, topleft_xy))
        x, y = topleft_xy
        if self.frame_color_transparency < 1.0:
            t = self.frame_thickness
            image_bgr[y - t : y + txt_h + t, x - t : x + txt_w + t, :] = (
                image_bgr[y - t : y + txt_h + t, x - t : x + txt_w + t, :]
                * self.frame_color_transparency
                + np.array(self.frame_color_bgr) * (1.0 - self.frame_color_transparency)
            ).astype(float)
        if self.fill_color_transparency < 1.0:
            image_bgr[y : y + txt_h, x : x + txt_w, :] = (
                image_bgr[y : y + txt_h, x : x + txt_w, :] * self.fill_color_transparency
                + np.array(self.fill_color_bgr) * (1.0 - self.fill_color_transparency)
            ).astype(float)
        cv2.putText(
            image_bgr,
            txt,
            topleft_xy,
            self.font_face,
            self.font_scale,
            self.font_color_bgr,
            self.font_line_thickness,
            self.font_line_type,
        )
        return image_bgr

    def get_text_size_wh(self, txt):
        ((txt_w, txt_h), _) = cv2.getTextSize(
            txt, self.font_face, self.font_scale, self.font_line_thickness
        )
        return txt_w, txt_h


class CompoundVisualizer(object):
    def __init__(self, visualizers):
        self.visualizers = visualizers

    def visualize(self, image_bgr, data):
        assert len(data) == len(
            self.visualizers
        ), "The number of datas {} should match the number of visualizers" " {}".format(
            len(data), len(self.visualizers)
        )
        image = image_bgr
        for i, visualizer in enumerate(self.visualizers):
            image = visualizer.visualize(image, data[i])
        return image

    def __str__(self):
        visualizer_str = ", ".join([str(v) for v in self.visualizers])
        return "Compound Visualizer [{}]".format(visualizer_str)
