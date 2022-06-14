# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import json
import logging
from typing import List, Optional
import torch
from torch import nn

from detectron2.utils.file_io import PathManager

from densepose.structures.mesh import create_mesh


class MeshAlignmentEvaluator:
    """
    Class for evaluation of 3D mesh alignment based on the learned vertex embeddings
    """

    def __init__(self, embedder: nn.Module, mesh_names: Optional[List[str]]):
        self.embedder = embedder
        # use the provided mesh names if not None and not an empty list
        self.mesh_names = mesh_names if mesh_names else embedder.mesh_names
        self.logger = logging.getLogger(__name__)
        with PathManager.open(
            "https://dl.fbaipublicfiles.com/densepose/data/cse/mesh_keyvertices_v0.json", "r"
        ) as f:
            self.mesh_keyvertices = json.load(f)

    def evaluate(self):
        ge_per_mesh = {}
        gps_per_mesh = {}
        for mesh_name_1 in self.mesh_names:
            avg_errors = []
            avg_gps = []
            embeddings_1 = self.embedder(mesh_name_1)
            keyvertices_1 = self.mesh_keyvertices[mesh_name_1]
            keyvertex_names_1 = list(keyvertices_1.keys())
            keyvertex_indices_1 = [keyvertices_1[name] for name in keyvertex_names_1]
            for mesh_name_2 in self.mesh_names:
                if mesh_name_1 == mesh_name_2:
                    continue
                embeddings_2 = self.embedder(mesh_name_2)
                keyvertices_2 = self.mesh_keyvertices[mesh_name_2]
                sim_matrix_12 = embeddings_1[keyvertex_indices_1].mm(embeddings_2.T)
                vertices_2_matching_keyvertices_1 = sim_matrix_12.argmax(axis=1)
                mesh_2 = create_mesh(mesh_name_2, embeddings_2.device)
                geodists = mesh_2.geodists[
                    vertices_2_matching_keyvertices_1,
                    [keyvertices_2[name] for name in keyvertex_names_1],
                ]
                Current_Mean_Distances = 0.255
                gps = (-(geodists ** 2) / (2 * (Current_Mean_Distances ** 2))).exp()
                avg_errors.append(geodists.mean().item())
                avg_gps.append(gps.mean().item())

            ge_mean = torch.as_tensor(avg_errors).mean().item()
            gps_mean = torch.as_tensor(avg_gps).mean().item()
            ge_per_mesh[mesh_name_1] = ge_mean
            gps_per_mesh[mesh_name_1] = gps_mean
        ge_mean_global = torch.as_tensor(list(ge_per_mesh.values())).mean().item()
        gps_mean_global = torch.as_tensor(list(gps_per_mesh.values())).mean().item()
        per_mesh_metrics = {
            "GE": ge_per_mesh,
            "GPS": gps_per_mesh,
        }
        return ge_mean_global, gps_mean_global, per_mesh_metrics
