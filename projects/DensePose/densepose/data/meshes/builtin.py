# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from .catalog import register_meshes

DENSEPOSE_MESHES_DIR = "meshes"

MESHES = []

register_meshes(MESHES, DENSEPOSE_MESHES_DIR)
