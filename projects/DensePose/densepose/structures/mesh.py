# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import pickle
from functools import lru_cache
from typing import Dict, Optional, Tuple
import torch

from detectron2.utils.file_io import PathManager

from densepose.data.meshes.catalog import MeshCatalog


class Mesh:
    def __init__(
        self,
        vertices: torch.Tensor,
        faces: torch.Tensor,
        geodists: Optional[torch.Tensor] = None,
        symmetry: Optional[Dict[str, torch.Tensor]] = None,
    ):
        """
        Args:
            vertices (tensor [N, 3] of float32): vertex coordinates in 3D
            faces (tensor [M, 3] of long): triangular face represented as 3
                vertex indices
            geodists (tensor [N, N] of float32): geodesic distances from
                vertex `i` to vertex `j` (optional, default: None)
            symmetry (dict: str -> tensor): various mesh symmetry data:
                - "vertex_transforms": vertex mapping under horizontal flip,
                  tensor of size [N] of type long; vertex `i` is mapped to
                  vertex `tensor[i]` (optional, default: None)
        """
        self.vertices = vertices
        self.faces = faces
        self.geodists = geodists
        self.symmetry = symmetry
        assert self.vertices.device == self.faces.device
        assert geodists is None or self.vertices.device == self.geodists.device
        assert symmetry is None or all(
            self.vertices.device == self.symmetry[key].device for key in self.symmetry
        )
        self.device = self.vertices.device

    def to(self, device: torch.device):
        return Mesh(
            self.vertices.to(device),
            self.faces.to(device),
            self.geodists.to(device) if self.geodists is not None else None,
            {key: value.to(device) for key, value in self.symmetry.items()}
            if self.symmetry is not None
            else None,
        )

    def get_geodists(self):
        if self.geodists is None:
            self.geodists = self._compute_geodists()
        return self.geodists

    def _compute_geodists(self):
        # TODO: compute using Laplace-Beltrami
        geodists = None
        return geodists


def load_mesh_data(mesh_fpath: str) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    with PathManager.open(mesh_fpath, "rb") as hFile:
        mesh_data = pickle.load(hFile)
        vertices = torch.as_tensor(mesh_data["vertices"], dtype=torch.float)
        faces = torch.as_tensor(mesh_data["faces"], dtype=torch.long)
        return vertices, faces
    return None, None


def load_mesh_geodists(geodists_fpath: str) -> Optional[torch.Tensor]:
    geodists_fpath_local = PathManager.get_local_path(geodists_fpath, timeout_sec=600)
    with PathManager.open(geodists_fpath_local, "rb") as hFile:
        return torch.as_tensor(pickle.load(hFile), dtype=torch.float)


@lru_cache()
def load_mesh_symmetry(symmetry_fpath: str) -> Optional[Dict[str, torch.Tensor]]:
    with PathManager.open(symmetry_fpath, "rb") as hFile:
        symmetry_loaded = pickle.load(hFile)
        symmetry = {
            "vertex_transforms": torch.as_tensor(
                symmetry_loaded["vertex_transforms"], dtype=torch.long
            ),
        }
        return symmetry


@lru_cache()
def create_mesh(mesh_name: str, device: torch.device):
    mesh_info = MeshCatalog[mesh_name]
    vertices, faces = load_mesh_data(mesh_info.data)
    geodists = load_mesh_geodists(mesh_info.geodists) if mesh_info.geodists is not None else None
    symmetry = load_mesh_symmetry(mesh_info.symmetry) if mesh_info.symmetry is not None else None
    mesh = Mesh(vertices, faces, geodists, symmetry)
    return mesh.to(device)
