# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import random
from typing import Tuple
import torch
from torch import nn
from torch.nn import functional as F

from detectron2.config import CfgNode

from densepose.structures.mesh import create_mesh

from .utils import sample_random_indices


class ShapeToShapeCycleLoss(nn.Module):
    """
    Cycle Loss for Shapes.
    Inspired by:
    "Mapping in a Cycle: Sinkhorn Regularized Unsupervised Learning for Point Cloud Shapes".
    """

    def __init__(self, cfg: CfgNode):
        super().__init__()
        self.shape_names = list(cfg.MODEL.ROI_DENSEPOSE_HEAD.CSE.EMBEDDERS.keys())
        self.all_shape_pairs = [
            (x, y) for i, x in enumerate(self.shape_names) for y in self.shape_names[i + 1 :]
        ]
        random.shuffle(self.all_shape_pairs)
        self.cur_pos = 0
        self.norm_p = cfg.MODEL.ROI_DENSEPOSE_HEAD.CSE.SHAPE_TO_SHAPE_CYCLE_LOSS.NORM_P
        self.temperature = cfg.MODEL.ROI_DENSEPOSE_HEAD.CSE.SHAPE_TO_SHAPE_CYCLE_LOSS.TEMPERATURE
        self.max_num_vertices = (
            cfg.MODEL.ROI_DENSEPOSE_HEAD.CSE.SHAPE_TO_SHAPE_CYCLE_LOSS.MAX_NUM_VERTICES
        )

    def _sample_random_pair(self) -> Tuple[str, str]:
        """
        Produce a random pair of different mesh names

        Return:
            tuple(str, str): a pair of different mesh names
        """
        if self.cur_pos >= len(self.all_shape_pairs):
            random.shuffle(self.all_shape_pairs)
            self.cur_pos = 0
        shape_pair = self.all_shape_pairs[self.cur_pos]
        self.cur_pos += 1
        return shape_pair

    def forward(self, embedder: nn.Module):
        """
        Do a forward pass with a random pair (src, dst) pair of shapes
        Args:
            embedder (nn.Module): module that computes vertex embeddings for different meshes
        """
        src_mesh_name, dst_mesh_name = self._sample_random_pair()
        return self._forward_one_pair(embedder, src_mesh_name, dst_mesh_name)

    def fake_value(self, embedder: nn.Module):
        losses = []
        for mesh_name in embedder.mesh_names:
            losses.append(embedder(mesh_name).sum() * 0)
        return torch.mean(torch.stack(losses))

    def _get_embeddings_and_geodists_for_mesh(
        self, embedder: nn.Module, mesh_name: str
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Produces embeddings and geodesic distance tensors for a given mesh. May subsample
        the mesh, if it contains too many vertices (controlled by
        SHAPE_CYCLE_LOSS_MAX_NUM_VERTICES parameter).
        Args:
            embedder (nn.Module): module that computes embeddings for mesh vertices
            mesh_name (str): mesh name
        Return:
            embeddings (torch.Tensor of size [N, D]): embeddings for selected mesh
                vertices (N = number of selected vertices, D = embedding space dim)
            geodists (torch.Tensor of size [N, N]): geodesic distances for the selected
                mesh vertices (N = number of selected vertices)
        """
        embeddings = embedder(mesh_name)
        indices = sample_random_indices(
            embeddings.shape[0], self.max_num_vertices, embeddings.device
        )
        mesh = create_mesh(mesh_name, embeddings.device)
        geodists = mesh.geodists
        if indices is not None:
            embeddings = embeddings[indices]
            geodists = geodists[torch.meshgrid(indices, indices)]
        return embeddings, geodists

    def _forward_one_pair(
        self, embedder: nn.Module, mesh_name_1: str, mesh_name_2: str
    ) -> torch.Tensor:
        """
        Do a forward pass with a selected pair of meshes
        Args:
            embedder (nn.Module): module that computes vertex embeddings for different meshes
            mesh_name_1 (str): first mesh name
            mesh_name_2 (str): second mesh name
        Return:
            Tensor containing the loss value
        """
        embeddings_1, geodists_1 = self._get_embeddings_and_geodists_for_mesh(embedder, mesh_name_1)
        embeddings_2, geodists_2 = self._get_embeddings_and_geodists_for_mesh(embedder, mesh_name_2)
        sim_matrix_12 = embeddings_1.mm(embeddings_2.T)

        c_12 = F.softmax(sim_matrix_12 / self.temperature, dim=1)
        c_21 = F.softmax(sim_matrix_12.T / self.temperature, dim=1)
        c_11 = c_12.mm(c_21)
        c_22 = c_21.mm(c_12)

        loss_cycle_11 = torch.norm(geodists_1 * c_11, p=self.norm_p)
        loss_cycle_22 = torch.norm(geodists_2 * c_22, p=self.norm_p)

        return loss_cycle_11 + loss_cycle_22
