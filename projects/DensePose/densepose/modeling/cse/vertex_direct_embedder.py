# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import pickle
import torch
from torch import nn

from detectron2.utils.file_io import PathManager

from .utils import normalize_embeddings


class VertexDirectEmbedder(nn.Module):
    """
    Class responsible for embedding vertices. Vertex embeddings take
    the form of a tensor of size [N, D], where
        N = number of vertices
        D = number of dimensions in the embedding space
    """

    def __init__(self, num_vertices: int, embed_dim: int):
        """
        Initialize embedder, set random embeddings

        Args:
            num_vertices (int): number of vertices to embed
            embed_dim (int): number of dimensions in the embedding space
        """
        super(VertexDirectEmbedder, self).__init__()
        self.embeddings = nn.Parameter(torch.Tensor(num_vertices, embed_dim))
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self):
        """
        Reset embeddings to random values
        """
        self.embeddings.zero_()

    def forward(self) -> torch.Tensor:
        """
        Produce vertex embeddings, a tensor of shape [N, D] where:
            N = number of vertices
            D = number of dimensions in the embedding space

        Return:
           Full vertex embeddings, a tensor of shape [N, D]
        """
        return normalize_embeddings(self.embeddings)

    @torch.no_grad()
    def load(self, fpath: str):
        """
        Load data from a file

        Args:
            fpath (str): file path to load data from
        """
        with PathManager.open(fpath, "rb") as hFile:
            data = pickle.load(hFile)
            for name in ["embeddings"]:
                if name in data:
                    getattr(self, name).copy_(
                        torch.tensor(data[name]).float().to(device=getattr(self, name).device)
                    )
