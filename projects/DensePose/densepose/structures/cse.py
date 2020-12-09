# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from dataclasses import dataclass
from typing import Union
import torch


@dataclass
class DensePoseEmbeddingPredictorOutput:
    """
    Predictor output that contains embedding and coarse segmentation data:
     * embedding: float tensor of size [N, D, H, W], contains estimated embeddings
     * coarse_segm: float tensor of size [N, K, H, W]
    Here D = MODEL.ROI_DENSEPOSE_HEAD.CSE.EMBED_SIZE
         K = MODEL.ROI_DENSEPOSE_HEAD.NUM_COARSE_SEGM_CHANNELS
    """

    embedding: torch.Tensor
    coarse_segm: torch.Tensor

    def __len__(self):
        """
        Number of instances (N) in the output
        """
        return self.coarse_segm.size(0)

    def __getitem__(
        self, item: Union[int, slice, torch.BoolTensor]
    ) -> "DensePoseEmbeddingPredictorOutput":
        """
        Get outputs for the selected instance(s)

        Args:
            item (int or slice or tensor): selected items
        """
        if isinstance(item, int):
            return DensePoseEmbeddingPredictorOutput(
                coarse_segm=self.coarse_segm[item].unsqueeze(0),
                embedding=self.embedding[item].unsqueeze(0),
            )
        else:
            return DensePoseEmbeddingPredictorOutput(
                coarse_segm=self.coarse_segm[item], embedding=self.embedding[item]
            )
