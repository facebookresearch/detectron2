# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import logging
from collections import UserDict
from dataclasses import dataclass
from typing import Iterable, Optional

from ..utils import maybe_prepend_base_path


@dataclass
class MeshInfo:
    name: str
    data: str
    geodists: Optional[str] = None
    symmetry: Optional[str] = None
    texcoords: Optional[str] = None


class _MeshCatalog(UserDict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mesh_ids = {}
        self.mesh_names = {}
        self.max_mesh_id = -1

    def __setitem__(self, key, value):
        if key in self:
            logger = logging.getLogger(__name__)
            logger.warning(
                f"Overwriting mesh catalog entry '{key}': old value {self[key]}"
                f", new value {value}"
            )
            mesh_id = self.mesh_ids[key]
        else:
            self.max_mesh_id += 1
            mesh_id = self.max_mesh_id
        super().__setitem__(key, value)
        self.mesh_ids[key] = mesh_id
        self.mesh_names[mesh_id] = key

    def get_mesh_id(self, shape_name: str) -> int:
        return self.mesh_ids[shape_name]

    def get_mesh_name(self, mesh_id: int) -> str:
        return self.mesh_names[mesh_id]


MeshCatalog = _MeshCatalog()


def register_mesh(mesh_info: MeshInfo, base_path: Optional[str]):
    MeshCatalog[mesh_info.name] = MeshInfo(
        name=mesh_info.name,
        data=maybe_prepend_base_path(base_path, mesh_info.data),
        geodists=(
            maybe_prepend_base_path(base_path, mesh_info.geodists)
            if mesh_info.geodists is not None
            else None
        ),
        symmetry=(
            maybe_prepend_base_path(base_path, mesh_info.symmetry)
            if mesh_info.symmetry is not None
            else None
        ),
        texcoords=(
            maybe_prepend_base_path(base_path, mesh_info.texcoords)
            if mesh_info.texcoords is not None
            else None
        ),
    )


def register_meshes(mesh_infos: Iterable[MeshInfo], base_path: Optional[str]):
    for mesh_info in mesh_infos:
        register_mesh(mesh_info, base_path)
