[isort]
line_length=100
multi_line_output=3
include_trailing_comma=True
known_standard_library=numpy,setuptools,mock
skip=./datasets,docs
skip_glob=*/__init__.py,**/configs/**,**/tests/config/**
known_myself=detectron2
known_third_party=fvcore,matplotlib,cv2,torch,torchvision,PIL,pycocotools,yacs,termcolor,cityscapesscripts,tabulate,tqdm,scipy,lvis,psutil,pkg_resources,caffe2,onnx,panopticapi,black,isort,av,iopath,omegaconf,hydra,yaml,pydoc,submitit,cloudpickle,packaging,timm,pandas,fairscale,pytorch3d,pytorch_lightning
no_lines_before=STDLIB,THIRDPARTY
sections=FUTURE,STDLIB,THIRDPARTY,myself,FIRSTPARTY,LOCALFOLDER
default_section=FIRSTPARTY

[mypy]
python_version=3.7
ignore_missing_imports = True
warn_unused_configs = True
disallow_untyped_defs = True
check_untyped_defs = True
warn_unused_ignores = True
warn_redundant_casts = True
show_column_numbers = True
follow_imports = silent
allow_redefinition = True
*; Require all functions to be annotated"*"
disallow_incomplete_defs = True
import numpy as np

import torch	import torchimport torch.nn as nn	import torch.nn as nn

from timm.models.layers import DropPath, Mlp, trunc_normal_	

from .backbone import Backbone	from .backbone import Backbone

from .utils import (	from .utils import (

@@ -123,8 +122,8 @@ def __init__(

            self.rel_pos_w = nn.Parameter(torch.zeros(rel_dim, head_dim))	            self.rel_pos_w = nn.Parameter(torch.zeros(rel_dim, head_dim))

            if not rel_pos_zero_init:	            if not rel_pos_zero_init:

                trunc_normal_(self.rel_pos_h, std=0.02)	                nn.init.trunc_normal_(self.rel_pos_h, std=0.02)

                trunc_normal_(self.rel_pos_w, std=0.02)	                nn.init.trunc_normal_(self.rel_pos_w, std=0.02)

    def forward(self, x):	    def forward(self, x):

        B, H, W, _ = x.shape	        B, H, W, _ = x.shape

@@ -235,6 +234,8 @@ def __init__(

            input_size=input_size,	            input_size=input_size,

        )	        )

        from timm.models.layers import DropPath, Mlp

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()	        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm_layer(dim_out)	        self.norm2 = norm_layer(dim_out)

        self.mlp = Mlp(	        self.mlp = Mlp(

@@ -414,13 +415,13 @@ def __init__(

        self._last_block_indexes = last_block_indexes	        self._last_block_indexes = last_block_indexes

        if self.pos_embed is not None:	        if self.pos_embed is not None:

            trunc_normal_(self.pos_embed, std=0.02)	            nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.apply(self._init_weights)	        self.apply(self._init_weights)

    def _init_weights(self, m):	    def _init_weights(self, m):

        if isinstance(m, nn.Linear):	        if isinstance(m, nn.Linear):

            trunc_normal_(m.weight, std=0.02)	            nn.init.trunc_normal_(m.weight, std=0.02)

            if isinstance(m, nn.Linear) and m.bias is not None:	            if isinstance(m, nn.Linear) and m.bias is not None:

                nn.init.constant_(m.bias, 0)	                nn.init.constant_(m.bias, 0)

        elif isinstance(m, nn.LayerNorm):	        elif isinstance(m, nn.LayerNorm):

  24  

detectron2/modeling/backbone/swin.py
