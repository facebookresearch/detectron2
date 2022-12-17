# Copyright (c) Facebook, Inc. and its affiliates.

from dataclasses import dataclass
from typing import Union
import torch
from torchvision.transforms.functional import rotate, resize
from torchvision.transforms import InterpolationMode
from typing import Any


@dataclass
class DensePoseChartPredictorOutput:
    """
    Predictor output that contains segmentation and inner coordinates predictions for predefined
    body parts:
     * coarse segmentation, a tensor of shape [N, K, Hout, Wout]
     * fine segmentation, a tensor of shape [N, C, Hout, Wout]
     * U coordinates, a tensor of shape [N, C, Hout, Wout]
     * V coordinates, a tensor of shape [N, C, Hout, Wout]
    where
     - N is the number of instances
     - K is the number of coarse segmentation channels (
         2 = foreground / background,
         15 = one of 14 body parts / background)
     - C is the number of fine segmentation channels (
         24 fine body parts / background)
     - Hout and Wout are height and width of predictions
    """

    coarse_segm: torch.Tensor
    fine_segm: torch.Tensor
    u: torch.Tensor
    v: torch.Tensor
    crt_segm: torch.Tensor
    crt_sigma: torch.Tensor

    def __len__(self):
        """
        Number of instances (N) in the output
        """
        return self.coarse_segm.size(0)

    def __getitem__(
        self, item: Union[int, slice, torch.BoolTensor]
    ) -> "DensePoseChartPredictorOutput":
        """
        Get outputs for the selected instance(s)

        Args:
            item (int or slice or tensor): selected items
        """

        def slice_if_not_none(data, item):
            if data is None:
                return None
            if isinstance(item, int):
                return data[item].unsqueeze(0)
            return data[item]

        if isinstance(item, int):
            return DensePoseChartPredictorOutput(
                coarse_segm=self.coarse_segm[item].unsqueeze(0),
                fine_segm=self.fine_segm[item].unsqueeze(0),
                u=self.u[item].unsqueeze(0),
                v=self.v[item].unsqueeze(0),
                crt_segm=slice_if_not_none(self.crt_segm, item),
                crt_sigma=slice_if_not_none(self.crt_sigma, item),
            )
        else:
            return DensePoseChartPredictorOutput(
                coarse_segm=self.coarse_segm[item],
                fine_segm=self.fine_segm[item],
                u=self.u[item],
                v=self.v[item],
                crt_segm=slice_if_not_none(self.crt_segm, item),
                crt_sigma=slice_if_not_none(self.crt_sigma, item)
            )

    def to(self, device: torch.device):
        """
        Transfers all tensors to the given device
        """

        def to_device_if_tensor(var: Any):
            if isinstance(var, torch.Tensor):
                return var.to(device)
            return var

        coarse_segm = self.coarse_segm.to(device)
        fine_segm = self.fine_segm.to(device)
        u = self.u.to(device)
        v = self.v.to(device)
        crt_segm = to_device_if_tensor(self.crt_segm)
        crt_sigma = to_device_if_tensor(self.crt_sigma)

        return DensePoseChartPredictorOutput(coarse_segm=coarse_segm, fine_segm=fine_segm, u=u, v=v, crt_segm=crt_segm,
                                             crt_sigma=crt_sigma)

    def rotate(self, labeled_boxes, angle):
        mark = 0
        for i in range(len(angle)):
            ag = angle[i]
            boxes_length = len(labeled_boxes[i])
            if ag == 0:
                mark += boxes_length
                continue
            # do angle for pseudo labels
            boxes = labeled_boxes[i].tensor
            h, w = boxes[:, 3] - boxes[:, 1], boxes[:, 2] - boxes[:, 0]
            for j in range(boxes_length):
                self.coarse_segm[j + mark] = get_rotated_result(self.coarse_segm[j + mark], h[j], w[j], ag)
                self.fine_segm[j + mark] = get_rotated_result(self.fine_segm[j + mark], h[j], w[j], ag)
                self.u[j + mark] = get_rotated_result(self.u[j + mark], h[j], w[j], ag)
                self.v[j + mark] = get_rotated_result(self.v[j + mark], h[j], w[j], ag)
                self.crt_segm[j + mark] = get_rotated_result(self.crt_segm[j + mark], h[j], w[j], ag)
                self.crt_sigma[j + mark] = get_rotated_result(self.crt_sigma[j + mark], h[j], w[j], ag)
            mark += boxes_length


def get_rotated_result(img, h, w, angle):
    img = resize(img, (int(h), int(w)))
    img = rotate(img, angle, expand=True, interpolation=InterpolationMode.BILINEAR)
    return resize(img, (112, 112))
