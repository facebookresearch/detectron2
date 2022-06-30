# Copyright (c) Facebook, Inc. and its affiliates.

from dataclasses import dataclass
from typing import Union
import torch


@dataclass
class BlockPredictorOutput:
    """
    Predictor output that contains segmentation and inner coordinates predictions for predefined
    body parts:
     * coarse segmentation, a tensor of shape [N, K, Hout, Wout]
     * fine segmentation, a tensor of shape [N, C, Hout, Wout]
     * U coordinates, two tensor of shape [N, C * Cb, Hout, Wout]
     * V coordinates, two tensor of shape [N, C * Cb, Hout, Wout]
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
    u_cls: torch.Tensor
    u_offset: torch.Tensor
    v_cls: torch.Tensor
    v_offset: torch.Tensor

    def __len__(self):
        """
        Number of instances (N) in the output
        """
        return self.coarse_segm.size(0)

    def __getitem__(
        self, item: Union[int, slice, torch.BoolTensor]
    ) -> "BlockPredictorOutput":
        """
        Get outputs for the selected instance(s)

        Args:
            item (int or slice or tensor): selected items
        """
        if isinstance(item, int):
            return BlockPredictorOutput(
                coarse_segm=self.coarse_segm[item].unsqueeze(0),
                fine_segm=self.fine_segm[item].unsqueeze(0),
                u_cls=self.u_cls[item].unsqueeze(0),
                u_offset=self.u_offset[item].unsqueeze(0),
                v_cls=self.v_cls[item].unsqueeze(0),
                v_offset=self.v_offset[item].unsqueeze(0)
            )
        else:
            return BlockPredictorOutput(
                coarse_segm=self.coarse_segm[item],
                fine_segm=self.fine_segm[item],
                u_cls=self.u_cls[item],
                u_offset=self.u_offset[item],
                v_cls=self.v_cls[item],
                v_offset=self.v_offset[item]
            )

    def to(self, device: torch.device):
        """
        Transfers all tensors to the given device
        """
        coarse_segm = self.coarse_segm.to(device)
        fine_segm = self.fine_segm.to(device)
        u_cls = self.u_cls.to(device)
        u_offset = self.u_offset.to(device)
        v_cls = self.v_cls.to(device)
        v_offset = self.v_offset.to(device)
        return BlockPredictorOutput(coarse_segm=coarse_segm, fine_segm=fine_segm,
                                    u_cls=u_cls, u_offset=u_offset, v_cls=v_cls, v_offset=v_offset)