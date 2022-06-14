# Copyright (c) Facebook, Inc. and its affiliates.

from dataclasses import dataclass
from typing import Union
import torch


@dataclass
class DistributionPredictorOutput:
    """
    Predictor output that contains segmentation and inner coordinates predictions for predefined
    body parts:
     * coarse segmentation, a tensor of shape [N, K, Hout, Wout]
     * fine segmentation, a tensor of shape [N, C, Hout, Wout]
     * U coordinates, a tensor of shape [N, C * Cb, Hout, Wout]
     * V coordinates, a tensor of shape [N, C * Cb, Hout, Wout]
    where
     - N is the number of instances
     - K is the number of coarse segmentation channels (
         2 = foreground / background,
         15 = one of 14 body parts / background)
     - C is the number of fine segmentation channels (
         24 fine body parts / background)
     - Cb is the number of block
     - Hout and Wout are height and width of predictions
    """

    coarse_segm: torch.Tensor
    fine_segm: torch.Tensor
    u_dis: torch.Tensor
    v_dis: torch.Tensor

    def __len__(self):
        """
        Number of instances (N) in the output
        """
        return self.coarse_segm.size(0)

    def __getitem__(
        self, item: Union[int, slice, torch.BoolTensor]
    ) -> "DistributionPredictorOutput":
        """
        Get outputs for the selected instance(s)

        Args:
            item (int or slice or tensor): selected items
        """
        if isinstance(item, int):
            return DistributionPredictorOutput(
                coarse_segm=self.coarse_segm[item].unsqueeze(0),
                fine_segm=self.fine_segm[item].unsqueeze(0),
                u_dis=self.u_dis[item].unsqueeze(0),
                v_dis=self.v_dis[item].unsqueeze(0),
            )
        else:
            return DistributionPredictorOutput(
                coarse_segm=self.coarse_segm[item],
                fine_segm=self.fine_segm[item],
                u_dis=self.u_dis[item],
                v_dis=self.v_dis[item],
            )

    def to(self, device: torch.device):
        """
        Transfers all tensors to the given device
        """
        coarse_segm = self.coarse_segm.to(device)
        fine_segm = self.fine_segm.to(device)
        u_dis = self.u_dis.to(device)
        v_dis = self.v_dis.to(device)
        return DistributionPredictorOutput(coarse_segm=coarse_segm, fine_segm=fine_segm, u_dis=u_dis, v_dis=v_dis)
