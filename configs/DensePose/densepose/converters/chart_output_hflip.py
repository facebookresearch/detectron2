# Copyright (c) Facebook, Inc. and its affiliates.

# pyre-unsafe
from dataclasses import fields
import torch

from densepose.structures import DensePoseChartPredictorOutput, DensePoseTransformData


def densepose_chart_predictor_output_hflip(
    densepose_predictor_output: DensePoseChartPredictorOutput,
    transform_data: DensePoseTransformData,
) -> DensePoseChartPredictorOutput:
    """
    Change  to take into account a Horizontal flip.
    """
    if len(densepose_predictor_output) > 0:

        PredictorOutput = type(densepose_predictor_output)
        output_dict = {}

        for field in fields(densepose_predictor_output):
            field_value = getattr(densepose_predictor_output, field.name)
            # flip tensors
            if isinstance(field_value, torch.Tensor):
                setattr(densepose_predictor_output, field.name, torch.flip(field_value, [3]))

        densepose_predictor_output = _flip_iuv_semantics_tensor(
            densepose_predictor_output, transform_data
        )
        densepose_predictor_output = _flip_segm_semantics_tensor(
            densepose_predictor_output, transform_data
        )

        for field in fields(densepose_predictor_output):
            output_dict[field.name] = getattr(densepose_predictor_output, field.name)

        return PredictorOutput(**output_dict)
    else:
        return densepose_predictor_output


def _flip_iuv_semantics_tensor(
    densepose_predictor_output: DensePoseChartPredictorOutput,
    dp_transform_data: DensePoseTransformData,
) -> DensePoseChartPredictorOutput:
    point_label_symmetries = dp_transform_data.point_label_symmetries
    uv_symmetries = dp_transform_data.uv_symmetries

    N, C, H, W = densepose_predictor_output.u.shape
    u_loc = (densepose_predictor_output.u[:, 1:, :, :].clamp(0, 1) * 255).long()
    v_loc = (densepose_predictor_output.v[:, 1:, :, :].clamp(0, 1) * 255).long()
    Iindex = torch.arange(C - 1, device=densepose_predictor_output.u.device)[
        None, :, None, None
    ].expand(N, C - 1, H, W)
    densepose_predictor_output.u[:, 1:, :, :] = uv_symmetries["U_transforms"][Iindex, v_loc, u_loc]
    densepose_predictor_output.v[:, 1:, :, :] = uv_symmetries["V_transforms"][Iindex, v_loc, u_loc]

    for el in ["fine_segm", "u", "v"]:
        densepose_predictor_output.__dict__[el] = densepose_predictor_output.__dict__[el][
            :, point_label_symmetries, :, :
        ]
    return densepose_predictor_output


def _flip_segm_semantics_tensor(
    densepose_predictor_output: DensePoseChartPredictorOutput, dp_transform_data
):
    if densepose_predictor_output.coarse_segm.shape[1] > 2:
        densepose_predictor_output.coarse_segm = densepose_predictor_output.coarse_segm[
            :, dp_transform_data.mask_label_symmetries, :, :
        ]
    return densepose_predictor_output
