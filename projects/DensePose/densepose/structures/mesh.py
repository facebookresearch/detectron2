# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

# pyre-unsafe

import pickle
from functools import lru_cache
from typing import Dict, Optional, Tuple
import torch

from detectron2.utils.file_io import PathManager

from densepose.data.meshes.catalog import MeshCatalog, MeshInfo


def _maybe_copy_to_device(
    attribute: Optional[torch.Tensor], device: torch.device
) -> Optional[torch.Tensor]:
    if attribute is None:
        return None
    return attribute.to(device)


class Mesh:
    def __init__(
        self,
        vertices: Optional[torch.Tensor] = None,
        faces: Optional[torch.Tensor] = None,
        geodists: Optional[torch.Tensor] = None,
        symmetry: Optional[Dict[str, torch.Tensor]] = None,
        texcoords: Optional[torch.Tensor] = None,
        mesh_info: Optional[MeshInfo] = None,
        device: Optional[torch.device] = None,
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
            texcoords (tensor [N, 2] of float32): texture coordinates, i.e. global
                and normalized mesh UVs (optional, default: None)
            mesh_info (MeshInfo type): necessary to load the attributes on-the-go,
                can be used instead of passing all the variables one by one
            device (torch.device): device of the Mesh. If not provided, will use
                the device of the vertices
        """
        self._vertices = vertices
        self._faces = faces
        self._geodists = geodists
        self._symmetry = symmetry
        self._texcoords = texcoords
        self.mesh_info = mesh_info
        self.device = device

        assert self._vertices is not None or self.mesh_info is not None

        all_fields = [self._vertices, self._faces, self._geodists, self._texcoords]

        if self.device is None:
            for field in all_fields:
                if field is not None:
                    self.device = field.device
                    break
            if self.device is None and symmetry is not None:
                for key in symmetry:
                    self.device = symmetry[key].device
                    break
            self.device = torch.device("cpu") if self.device is None else self.device

        assert all([var.device == self.device for var in all_fields if var is not None])
        if symmetry:
            assert all(symmetry[key].device == self.device for key in symmetry)
        if texcoords and vertices:
            assert len(vertices) == len(texcoords)

    def to(self, device: torch.device):
        device_symmetry = self._symmetry
        if device_symmetry:
            device_symmetry = {key: value.to(device) for key, value in device_symmetry.items()}
        return Mesh(
            _maybe_copy_to_device(self._vertices, device),
            _maybe_copy_to_device(self._faces, device),
            _maybe_copy_to_device(self._geodists, device),
            device_symmetry,
            _maybe_copy_to_device(self._texcoords, device),
            self.mesh_info,
            device,
        )

    @property
    def vertices(self):
        if self._vertices is None and self.mesh_info is not None:
            self._vertices = load_mesh_data(self.mesh_info.data, "vertices", self.device)
        return self._vertices

    @property
    def faces(self):
        if self._faces is None and self.mesh_info is not None:
            self._faces = load_mesh_data(self.mesh_info.data, "faces", self.device)
        return self._faces

    @property
    def geodists(self):
        if self._geodists is None and self.mesh_info is not None:
            self._geodists = load_mesh_auxiliary_data(self.mesh_info.geodists, self.device)
        return self._geodists

    @property
    def symmetry(self):
        if self._symmetry is None and self.mesh_info is not None:
            self._symmetry = load_mesh_symmetry(self.mesh_info.symmetry, self.device)
        return self._symmetry

    @property
    def texcoords(self):
        if self._texcoords is None and self.mesh_info is not None:
            self._texcoords = load_mesh_auxiliary_data(self.mesh_info.texcoords, self.device)
        return self._texcoords

    def get_geodists(self):
        if self.geodists is None:
            self.geodists = self._compute_geodists()
        return self.geodists

    def _compute_geodists(self):
        # TODO: compute using Laplace-Beltrami
        geodists = None
        return geodists


def load_mesh_data(
    mesh_fpath: str, field: str, device: Optional[torch.device] = None
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    with PathManager.open(mesh_fpath, "rb") as hFile:
        # pyre-fixme[7]: Expected `Tuple[Optional[Tensor], Optional[Tensor]]` but
        #  got `Tensor`.
        return torch.as_tensor(pickle.load(hFile)[field], dtype=torch.float).to(device)
    return None


def load_mesh_auxiliary_data(
    fpath: str, device: Optional[torch.device] = None
) -> Optional[torch.Tensor]:
    fpath_local = PathManager.get_local_path(fpath)
    with PathManager.open(fpath_local, "rb") as hFile:
        return torch.as_tensor(pickle.load(hFile), dtype=torch.float).to(device)
    return None


@lru_cache()
def load_mesh_symmetry(
    symmetry_fpath: str, device: Optional[torch.device] = None
) -> Optional[Dict[str, torch.Tensor]]:
    with PathManager.open(symmetry_fpath, "rb") as hFile:
        symmetry_loaded = pickle.load(hFile)
        symmetry = {
            "vertex_transforms": torch.as_tensor(
                symmetry_loaded["vertex_transforms"], dtype=torch.long
            ).to(device),
        }
        return symmetry
    return None


@lru_cache()
def create_mesh(mesh_name: str, device: Optional[torch.device] = None) -> Mesh:
    return Mesh(mesh_info=MeshCatalog[mesh_name], device=device)
