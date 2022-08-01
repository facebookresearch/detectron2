# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import imp
import logging
import numpy as np
import pickle
from enum import Enum
from typing import Optional
import torch
from torch import nn

from detectron2.config import CfgNode
from detectron2.layers import Conv2d
from detectron2.utils.file_io import PathManager
from ..utils import initialize_module_params

class Corrector(nn.Module):

    DEFAULT_MODEL_CHECKPOINT_PREFIX = "roi_heads.corrector."

    def __init__(self, cfg: CfgNode, input_channels: int):
        """
        Initialize mesh correctors. An corrector for mesh `i` is stored in a submodule
        "corrector_{i}".

        Args:
            cfg (CfgNode): configuration options
        """
        super(Corrector, self).__init__()
        hidden_dim = cfg.MODEL.SEMI.COR.CONV_HEAD_DIM
        kernel_size          = cfg.MODEL.SEMI.COR.CONV_HEAD_KERNEL
        self.n_stacked_convs = cfg.MODEL.SEMI.COR.NUM_STACKED_CONVS
        # fmt: on
        pad_size = kernel_size // 2
        n_channels = input_channels
        for i in range(self.n_stacked_convs):
            layer = Conv2d(n_channels, hidden_dim, kernel_size, stride=1, padding=pad_size)
            layer_name = self._get_layer_name(i)
            self.add_module(layer_name, layer)
            n_channels = hidden_dim
        self.n_out_channels = n_channels
        initialize_module_params(self)

        logger = logging.getLogger(__name__)
        for mesh_name, embedder_spec in cfg.MODEL.ROI_DENSEPOSE_HEAD.CSE.EMBEDDERS.items():
            logger.info(f"Adding embedder embedder_{mesh_name} with spec {embedder_spec}")
            self.add_module(f"embedder_{mesh_name}", create_embedder(embedder_spec, embedder_dim))
            self.mesh_names.add(mesh_name)

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
            # pyre-fixme[6]: Expected `OrderedDict[typing.Any, typing.Any]` for 1st
            #  param but got `Dict[typing.Any, typing.Any]`.
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
