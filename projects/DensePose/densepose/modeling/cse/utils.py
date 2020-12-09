# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import torch


def squared_euclidean_distance_matrix(pts1: torch.Tensor, pts2: torch.Tensor) -> torch.Tensor:
    """
    Get squared Euclidean Distance Matrix
    Computes pairwise squared Euclidean distances between points

    Args:
        pts1: Tensor [M x D], M is the number of points, D is feature dimensionality
        pts2: Tensor [N x D], N is the number of points, D is feature dimensionality

    Return:
        Tensor [M, N]: matrix of squared Euclidean distances; at index (m, n)
            it contains || pts1[m] - pts2[n] ||^2
    """
    edm = torch.mm(-2 * pts1, pts2.t())
    edm += (pts1 * pts1).sum(1, keepdim=True) + (pts2 * pts2).sum(1, keepdim=True).t()
    return edm.contiguous()


def normalize_embeddings(embeddings: torch.Tensor, epsilon: float = 1e-6) -> torch.Tensor:
    """
    Normalize N D-dimensional embedding vectors arranged in a tensor [N, D]

    Args:
        embeddings (tensor [N, D]): N D-dimensional embedding vectors
        epsilon (float): minimum value for a vector norm
    Return:
        Normalized embeddings (tensor [N, D]), such that L2 vector norms are all equal to 1.
    """
    return embeddings / torch.clamp(embeddings.norm(p=None, dim=1, keepdim=True), min=epsilon)
