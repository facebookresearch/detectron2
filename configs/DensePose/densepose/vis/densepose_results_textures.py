# Copyright (c) Facebook, Inc. and its affiliates.

# pyre-unsafe
import numpy as np
from typing import List, Optional, Tuple
import torch

from detectron2.data.detection_utils import read_image

from ..structures import DensePoseChartResult
from .base import Boxes, Image
from .densepose_results import DensePoseResultsVisualizer


def get_texture_atlas(path: Optional[str]) -> Optional[np.ndarray]:
    if path is None:
        return None

    # Reading images like that downsamples 16-bit images to 8-bit
    # If 16-bit images are needed, we can replace that by cv2.imread with the
    # cv2.IMREAD_UNCHANGED flag (with cv2 we also need it to keep alpha channels)
    # The rest of the pipeline would need to be adapted to 16-bit images too
    bgr_image = read_image(path)
    rgb_image = np.copy(bgr_image)  # Convert BGR -> RGB
    rgb_image[:, :, :3] = rgb_image[:, :, 2::-1]  # Works with alpha channel
    return rgb_image


class DensePoseResultsVisualizerWithTexture(DensePoseResultsVisualizer):
    """
    texture_atlas: An image, size 6N * 4N, with N * N squares for each of the 24 body parts.
            It must follow the grid found at https://github.com/facebookresearch/DensePose/blob/master/DensePoseData/demo_data/texture_atlas_200.png  # noqa
            For each body part, U is proportional to the x coordinate, and (1 - V) to y
    """

    def __init__(self, texture_atlas, **kwargs):
        self.texture_atlas = texture_atlas
        self.body_part_size = texture_atlas.shape[0] // 6
        assert self.body_part_size == texture_atlas.shape[1] // 4

    def visualize(
        self,
        image_bgr: Image,
        results_and_boxes_xywh: Tuple[Optional[List[DensePoseChartResult]], Optional[Boxes]],
    ) -> Image:
        densepose_result, boxes_xywh = results_and_boxes_xywh
        if densepose_result is None or boxes_xywh is None:
            return image_bgr

        boxes_xywh = boxes_xywh.int().cpu().numpy()
        texture_image, alpha = self.get_texture()
        for i, result in enumerate(densepose_result):
            iuv_array = torch.cat((result.labels[None], result.uv.clamp(0, 1)))
            x, y, w, h = boxes_xywh[i]
            bbox_image = image_bgr[y : y + h, x : x + w]
            image_bgr[y : y + h, x : x + w] = self.generate_image_with_texture(
                texture_image, alpha, bbox_image, iuv_array.cpu().numpy()
            )
        return image_bgr

    def get_texture(self):
        N = self.body_part_size
        texture_image = np.zeros([24, N, N, self.texture_atlas.shape[-1]])
        for i in range(4):
            for j in range(6):
                texture_image[(6 * i + j), :, :, :] = self.texture_atlas[
                    N * j : N * (j + 1), N * i : N * (i + 1), :
                ]

        if texture_image.shape[-1] == 4:  # Image with alpha channel
            alpha = texture_image[:, :, :, -1] / 255.0
            texture_image = texture_image[:, :, :, :3]
        else:
            alpha = texture_image.sum(axis=-1) > 0

        return texture_image, alpha

    def generate_image_with_texture(self, texture_image, alpha, bbox_image_bgr, iuv_array):

        I, U, V = iuv_array
        generated_image_bgr = bbox_image_bgr.copy()

        for PartInd in range(1, 25):
            x, y = np.where(I == PartInd)
            x_index = (U[x, y] * (self.body_part_size - 1)).astype(int)
            y_index = ((1 - V[x, y]) * (self.body_part_size - 1)).astype(int)
            part_alpha = np.expand_dims(alpha[PartInd - 1, y_index, x_index], -1)
            generated_image_bgr[I == PartInd] = (
                generated_image_bgr[I == PartInd] * (1 - part_alpha)
                + texture_image[PartInd - 1, y_index, x_index] * part_alpha
            )

        return generated_image_bgr.astype(np.uint8)
