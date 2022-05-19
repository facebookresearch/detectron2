# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import logging
import numpy as np
import pickle
from enum import Enum
from typing import Optional
import torch
from torch import nn

from detectron2.config import CfgNode
from detectron2.utils.file_io import PathManager

from .vertex_direct_embedder import VertexDirectEmbedder
from .vertex_feature_embedder import VertexFeatureEmbedder


class EmbedderType(Enum):
    """
    Embedder type which defines how vertices are mapped into the embedding space:
     - "vertex_direct": direct vertex embedding
     - "vertex_feature": embedding vertex features
    """

    VERTEX_DIRECT = "vertex_direct"
    VERTEX_FEATURE = "vertex_feature"


def create_embedder(embedder_spec: CfgNode, embedder_dim: int) -> nn.Module:
    """
    Create an embedder based on the provided configuration

    Args:
        embedder_spec (CfgNode): embedder configuration
        embedder_dim (int): embedding space dimensionality
    Return:
        An embedder instance for the specified configuration
        Raises ValueError, in case of unexpected  embedder type
    """
    embedder_type = EmbedderType(embedder_spec.TYPE)
    if embedder_type == EmbedderType.VERTEX_DIRECT:
        embedder = VertexDirectEmbedder(
            num_vertices=embedder_spec.NUM_VERTICES,
            embed_dim=embedder_dim,
        )
        if embedder_spec.INIT_FILE != "":
            embedder.load(embedder_spec.INIT_FILE)
    elif embedder_type == EmbedderType.VERTEX_FEATURE:
        embedder = VertexFeatureEmbedder(
            num_vertices=embedder_spec.NUM_VERTICES,
            feature_dim=embedder_spec.FEATURE_DIM,
            embed_dim=embedder_dim,
            train_features=embedder_spec.FEATURES_TRAINABLE,
        )
        if embedder_spec.INIT_FILE != "":
            embedder.load(embedder_spec.INIT_FILE)
    else:
        raise ValueError(f"Unexpected embedder type {embedder_type}")

    if not embedder_spec.IS_TRAINABLE:
        embedder.requires_grad_(False)

    return embedder


class Embedder(nn.Module):
    """
    Embedder module that serves as a container for embedders to use with different
    meshes. Extends Module to automatically save / load state dict.
    """

    DEFAULT_MODEL_CHECKPOINT_PREFIX = "roi_heads.embedder."

    def __init__(self, cfg: CfgNode):
        """
        Initialize mesh embedders. An embedder for mesh `i` is stored in a submodule
        "embedder_{i}".

        Args:
            cfg (CfgNode): configuration options
        """
        super(Embedder, self).__init__()
        self.mesh_names = set()
        embedder_dim = cfg.MODEL.ROI_DENSEPOSE_HEAD.CSE.EMBED_SIZE
        logger = logging.getLogger(__name__)
        for mesh_name, embedder_spec in cfg.MODEL.ROI_DENSEPOSE_HEAD.CSE.EMBEDDERS.items():
            logger.info(f"Adding embedder embedder_{mesh_name} with spec {embedder_spec}")
            self.add_module(f"embedder_{mesh_name}", create_embedder(embedder_spec, embedder_dim))
            self.mesh_names.add(mesh_name)
        if cfg.MODEL.WEIGHTS != "":
            self.load_from_model_checkpoint(cfg.MODEL.WEIGHTS)

    def load_from_model_checkpoint(self, fpath: str, prefix: Optional[str] = None):
        if prefix is None:
            prefix = Embedder.DEFAULT_MODEL_CHECKPOINT_PREFIX
        state_dict = None
        if fpath.endswith(".pkl"):
            with PathManager.open(fpath, "rb") as hFile:
                state_dict = pickle.load(hFile, encoding="latin1")  # pyre-ignore[6]
        else:
            with PathManager.open(fpath, "rb") as hFile:
                state_dict = torch.load(hFile, map_location=torch.device("cpu"))
        if state_dict is not None and "model" in state_dict:
            state_dict_local = {}
            for key in state_dict["model"]:
                if key.startswith(prefix):
                    v_key = state_dict["model"][key]
                    if isinstance(v_key, np.ndarray):
                        v_key = torch.from_numpy(v_key)
                    state_dict_local[key[len(prefix) :]] = v_key
            # non-strict loading to finetune on different meshes
            self.load_state_dict(state_dict_local, strict=False)

    def forward(self, mesh_name: str) -> torch.Tensor:
        """
        Produce vertex embeddings for the specific mesh; vertex embeddings are
        a tensor of shape [N, D] where:
            N = number of vertices
            D = number of dimensions in the embedding space
        Args:
            mesh_name (str): name of a mesh for which to obtain vertex embeddings
        Return:
            Vertex embeddings, a tensor of shape [N, D]
        """
        return getattr(self, f"embedder_{mesh_name}")()

    def has_embeddings(self, mesh_name: str) -> bool:
        return hasattr(self, f"embedder_{mesh_name}")
