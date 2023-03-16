# Copyright (c) Facebook, Inc. and its affiliates.
from .config import add_panoptic_deeplab_config
from .dataset_mapper import PanopticDeeplabDatasetMapper
from .panoptic_seg import (
    INS_EMBED_BRANCHES_REGISTRY,
    PanopticDeepLab,
    PanopticDeepLabInsEmbedHead,
    PanopticDeepLabSemSegHead,
    build_ins_embed_branch,
)
