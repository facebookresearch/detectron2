# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from dataclasses import dataclass
from typing import Tuple
import torch


@dataclass
class DensePoseChartResult:
    """
    DensePose results for chart-based methods represented by labels and inner
    coordinates (U, V) of individual charts. Each chart is a 2D manifold
    that has an associated label and is parameterized by two coordinates U and V.
    Both U and V take values in [0, 1].
    Thus the results are represented by two tensors:
    - labels (tensor [H, W] of uint8): contains estimated label for each pixel of
        the detection bounding box of size (H, W)
    - uv (tensor [2, H, W] of float): contains estimated U and V coordinates
        for each pixel of the detection bounding box of size (H, W)
    """

    labels: torch.Tensor
    uv: torch.Tensor


@dataclass
class DensePoseChartResultQuantized:
    """
    DensePose results for chart-based methods represented by labels and quantized
    inner coordinates (U, V) of individual charts. Each chart is a 2D manifold
    that has an associated label and is parameterized by two coordinates U and V.
    Both U and V take values in [0, 1].
    Quantized coordinates Uq and Vq have uint8 values which are obtained as:
      Uq = U * 255 (hence 0 <= Uq <= 255)
      Vq = V * 255 (hence 0 <= Vq <= 255)
    Thus the results are represented by one tensor:
    - labels_uv_uint8 (tensor [3, H, W] of uint8): contains estimated label
        and quantized coordinates Uq and Vq for each pixel of the detection
        bounding box of size (H, W)
    """

    labels_uv_uint8: torch.Tensor


@dataclass
class DensePoseChartResultCompressed:
    """
    DensePose results for chart-based methods represented by a PNG-encoded string.
    The tensor of quantized DensePose results of size [3, H, W] is considered
    as an image with 3 color channels. PNG compression is applied and the result
    is stored as a Base64-encoded string. The following attributes are defined:
    - shape_chw (tuple of 3 int): contains shape of the result tensor
        (number of channels, height, width)
    - labels_uv_str (str): contains Base64-encoded results tensor of size
        [3, H, W] compressed with PNG compression methods
    """

    shape_chw: Tuple[int, int, int]
    labels_uv_str: str


def quantize_densepose_chart_result(result: DensePoseChartResult) -> DensePoseChartResultQuantized:
    """
    Applies quantization to DensePose chart-based result.

    Args:
        result (DensePoseChartResult): DensePose chart-based result
    Return:
        Quantized DensePose chart-based result (DensePoseChartResultQuantized)
    """
    h, w = result.labels.shape
    labels_uv_uint8 = torch.zeros([3, h, w], dtype=torch.uint8, device=result.labels.device)
    labels_uv_uint8[0] = result.labels
    labels_uv_uint8[1:] = (result.uv * 255).clamp(0, 255).byte()
    return DensePoseChartResultQuantized(labels_uv_uint8=labels_uv_uint8)


def compress_quantized_densepose_chart_result(
    result: DensePoseChartResultQuantized,
) -> DensePoseChartResultCompressed:
    """
    Compresses quantized DensePose chart-based result

    Args:
        result (DensePoseChartResultQuantized): quantized DensePose chart-based result
    Return:
        Compressed DensePose chart-based result (DensePoseChartResultCompressed)
    """
    import base64
    import numpy as np
    from io import BytesIO
    from PIL import Image

    labels_uv_uint8_np_chw = result.labels_uv_uint8.cpu().numpy()
    labels_uv_uint8_np_hwc = np.moveaxis(labels_uv_uint8_np_chw, 0, -1)
    im = Image.fromarray(labels_uv_uint8_np_hwc)
    fstream = BytesIO()
    im.save(fstream, format="png", optimize=True)
    labels_uv_str = base64.encodebytes(fstream.getvalue()).decode()
    shape_chw = labels_uv_uint8_np_chw.shape
    return DensePoseChartResultCompressed(labels_uv_str=labels_uv_str, shape_chw=shape_chw)


def decompress_compressed_densepose_chart_result(
    result: DensePoseChartResultCompressed,
) -> DensePoseChartResultQuantized:
    """
    Decompresses DensePose chart-based result encoded into a base64 string

    Args:
        result (DensePoseChartResultCompressed): compressed DensePose chart result
    Return:
        Quantized DensePose chart-based result (DensePoseChartResultQuantized)
    """
    import base64
    import numpy as np
    from io import BytesIO
    from PIL import Image

    fstream = BytesIO(base64.decodebytes(result.labels_uv_str.encode()))
    im = Image.open(fstream)
    labels_uv_uint8_np_chw = np.moveaxis(np.array(im.getdata(), dtype=np.uint8), -1, 0)
    return DensePoseChartResultQuantized(
        labels_uv_uint8=torch.from_numpy(labels_uv_uint8_np_chw.reshape(result.shape_chw))
    )
