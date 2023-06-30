# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from typing import Any, List
import torch
from torch import nn
from torch.nn import functional as F

from detectron2.config import CfgNode
from detectron2.structures import Instances

from densepose.data.meshes.catalog import MeshCatalog
from densepose.modeling.cse.utils import normalize_embeddings, squared_euclidean_distance_matrix

from .embed_utils import PackedCseAnnotations
from .mask import extract_data_for_mask_loss_from_matches


def _create_pixel_dist_matrix(grid_size: int) -> torch.Tensor:
    rows = torch.arange(grid_size)
    cols = torch.arange(grid_size)
    # at index `i` contains [row, col], where
    # row = i // grid_size
    # col = i % grid_size
    pix_coords = (
        torch.stack(torch.meshgrid(rows, cols), -1).reshape((grid_size * grid_size, 2)).float()
    )
    return squared_euclidean_distance_matrix(pix_coords, pix_coords)


def _sample_fg_pixels_randperm(fg_mask: torch.Tensor, sample_size: int) -> torch.Tensor:
    fg_mask_flattened = fg_mask.reshape((-1,))
    num_pixels = int(fg_mask_flattened.sum().item())
    fg_pixel_indices = fg_mask_flattened.nonzero(as_tuple=True)[0]
    if (sample_size <= 0) or (num_pixels <= sample_size):
        return fg_pixel_indices
    sample_indices = torch.randperm(num_pixels, device=fg_mask.device)[:sample_size]
    return fg_pixel_indices[sample_indices]


def _sample_fg_pixels_multinomial(fg_mask: torch.Tensor, sample_size: int) -> torch.Tensor:
    fg_mask_flattened = fg_mask.reshape((-1,))
    num_pixels = int(fg_mask_flattened.sum().item())
    if (sample_size <= 0) or (num_pixels <= sample_size):
        return fg_mask_flattened.nonzero(as_tuple=True)[0]
    return fg_mask_flattened.float().multinomial(sample_size, replacement=False)


class PixToShapeCycleLoss(nn.Module):
    """
    Cycle loss for pixel-vertex correspondence
    """

    def __init__(self, cfg: CfgNode):
        super().__init__()
        self.shape_names = list(cfg.MODEL.ROI_DENSEPOSE_HEAD.CSE.EMBEDDERS.keys())
        self.embed_size = cfg.MODEL.ROI_DENSEPOSE_HEAD.CSE.EMBED_SIZE
        self.norm_p = cfg.MODEL.ROI_DENSEPOSE_HEAD.CSE.PIX_TO_SHAPE_CYCLE_LOSS.NORM_P
        self.use_all_meshes_not_gt_only = (
            cfg.MODEL.ROI_DENSEPOSE_HEAD.CSE.PIX_TO_SHAPE_CYCLE_LOSS.USE_ALL_MESHES_NOT_GT_ONLY
        )
        self.num_pixels_to_sample = (
            cfg.MODEL.ROI_DENSEPOSE_HEAD.CSE.PIX_TO_SHAPE_CYCLE_LOSS.NUM_PIXELS_TO_SAMPLE
        )
        self.pix_sigma = cfg.MODEL.ROI_DENSEPOSE_HEAD.CSE.PIX_TO_SHAPE_CYCLE_LOSS.PIXEL_SIGMA
        self.temperature_pix_to_vertex = (
            cfg.MODEL.ROI_DENSEPOSE_HEAD.CSE.PIX_TO_SHAPE_CYCLE_LOSS.TEMPERATURE_PIXEL_TO_VERTEX
        )
        self.temperature_vertex_to_pix = (
            cfg.MODEL.ROI_DENSEPOSE_HEAD.CSE.PIX_TO_SHAPE_CYCLE_LOSS.TEMPERATURE_VERTEX_TO_PIXEL
        )
        self.pixel_dists = _create_pixel_dist_matrix(cfg.MODEL.ROI_DENSEPOSE_HEAD.HEATMAP_SIZE)

    def forward(
        self,
        proposals_with_gt: List[Instances],
        densepose_predictor_outputs: Any,
        packed_annotations: PackedCseAnnotations,
        embedder: nn.Module,
    ):
        """
        Args:
            proposals_with_gt (list of Instances): detections with associated
                ground truth data; each item corresponds to instances detected
                on 1 image; the number of items corresponds to the number of
                images in a batch
            densepose_predictor_outputs: an object of a dataclass that contains predictor
                outputs with estimated values; assumed to have the following attributes:
                * embedding - embedding estimates, tensor of shape [N, D, S, S], where
                  N = number of instances (= sum N_i, where N_i is the number of
                      instances on image i)
                  D = embedding space dimensionality (MODEL.ROI_DENSEPOSE_HEAD.CSE.EMBED_SIZE)
                  S = output size (width and height)
            packed_annotations (PackedCseAnnotations): contains various data useful
                for loss computation, each data is packed into a single tensor
            embedder (nn.Module): module that computes vertex embeddings for different meshes
        """
        pix_embeds = densepose_predictor_outputs.embedding
        if self.pixel_dists.device != pix_embeds.device:
            # should normally be done only once
            self.pixel_dists = self.pixel_dists.to(device=pix_embeds.device)
        with torch.no_grad():
            mask_loss_data = extract_data_for_mask_loss_from_matches(
                proposals_with_gt, densepose_predictor_outputs.coarse_segm
            )
        # GT masks - tensor of shape [N, S, S] of int64
        masks_gt = mask_loss_data.masks_gt.long()  # pyre-ignore[16]
        assert len(pix_embeds) == len(masks_gt), (
            f"Number of instances with embeddings {len(pix_embeds)} != "
            f"number of instances with GT masks {len(masks_gt)}"
        )
        losses = []
        mesh_names = (
            self.shape_names
            if self.use_all_meshes_not_gt_only
            else [
                MeshCatalog.get_mesh_name(mesh_id.item())
                for mesh_id in packed_annotations.vertex_mesh_ids_gt.unique()
            ]
        )
        for pixel_embeddings, mask_gt in zip(pix_embeds, masks_gt):
            # pixel_embeddings [D, S, S]
            # mask_gt [S, S]
            for mesh_name in mesh_names:
                mesh_vertex_embeddings = embedder(mesh_name)
                # pixel indices [M]
                pixel_indices_flattened = _sample_fg_pixels_randperm(
                    mask_gt, self.num_pixels_to_sample
                )
                # pixel distances [M, M]
                pixel_dists = self.pixel_dists.to(pixel_embeddings.device)[
                    torch.meshgrid(pixel_indices_flattened, pixel_indices_flattened)
                ]
                # pixel embeddings [M, D]
                pixel_embeddings_sampled = normalize_embeddings(
                    pixel_embeddings.reshape((self.embed_size, -1))[:, pixel_indices_flattened].T
                )
                # pixel-vertex similarity [M, K]
                sim_matrix = pixel_embeddings_sampled.mm(mesh_vertex_embeddings.T)
                c_pix_vertex = F.softmax(sim_matrix / self.temperature_pix_to_vertex, dim=1)
                c_vertex_pix = F.softmax(sim_matrix.T / self.temperature_vertex_to_pix, dim=1)
                c_cycle = c_pix_vertex.mm(c_vertex_pix)
                loss_cycle = torch.norm(pixel_dists * c_cycle, p=self.norm_p)
                losses.append(loss_cycle)

        if len(losses) == 0:
            return pix_embeds.sum() * 0
        return torch.stack(losses, dim=0).mean()

    def fake_value(self, densepose_predictor_outputs: Any, embedder: nn.Module):
        losses = [
            embedder(mesh_name).sum() * 0 for mesh_name in embedder.mesh_names
        ]
        losses.append(densepose_predictor_outputs.embedding.sum() * 0)
        return torch.mean(torch.stack(losses))
