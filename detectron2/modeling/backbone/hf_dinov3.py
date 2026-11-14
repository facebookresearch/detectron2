# Copyright (c) Facebook, Inc. and its affiliates.
from typing import Any, Dict, Optional

import torch

from detectron2.layers import ShapeSpec

from .backbone import Backbone

__all__ = ["HFDINOv3ViT"]


class HFDINOv3ViT(Backbone):
    """
    Hugging Face DINOv3 ViT backbone adapter for ViTDet-style models.

    The adapter exposes the DINOv3 patch tokens as a single spatial feature map
    with stride equal to the patch size, matching the contract expected by
    :class:`SimpleFeaturePyramid`.
    """

    def __init__(
        self,
        model_name: str = "facebook/dinov3-vitb16-pretrain-lvd1689m",
        *,
        out_feature: str = "last_feat",
        pretrained: bool = True,
        freeze: bool = True,
        config_kwargs: Optional[Dict[str, Any]] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        config_kwargs = {} if config_kwargs is None else dict(config_kwargs)
        model_kwargs = {} if model_kwargs is None else dict(model_kwargs)

        try:
            from transformers import DINOv3ViTConfig, DINOv3ViTModel
        except ImportError as exc:
            raise ImportError(
                "HFDINOv3ViT requires Hugging Face Transformers. "
                "Install it with `pip install transformers`."
            ) from exc

        if pretrained:
            self.model = DINOv3ViTModel.from_pretrained(model_name, **model_kwargs)
        else:
            self.model = DINOv3ViTModel(DINOv3ViTConfig(**config_kwargs))

        self.out_feature = out_feature
        self.freeze_encoder = freeze

        cfg = self.model.config
        self.patch_size = int(cfg.patch_size)
        self.num_prefix_tokens = 1 + int(getattr(cfg, "num_register_tokens", 0))
        self._out_features = [out_feature]
        self._out_feature_channels = {out_feature: int(cfg.hidden_size)}
        self._out_feature_strides = {out_feature: self.patch_size}

        if self.freeze_encoder:
            for parameter in self.model.parameters():
                parameter.requires_grad = False
            self.model.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_encoder:
            self.model.eval()
        return self

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        height, width = x.shape[-2:]
        if height % self.patch_size != 0 or width % self.patch_size != 0:
            raise ValueError(
                f"Input height/width must be divisible by patch_size={self.patch_size}; "
                f"got {(height, width)}."
            )

        outputs = self.model(pixel_values=x, return_dict=True)
        hidden = outputs.last_hidden_state

        if hidden.dim() == 4:
            features = hidden
        else:
            grid_h, grid_w = height // self.patch_size, width // self.patch_size
            expected_tokens = grid_h * grid_w
            if hidden.shape[1] == expected_tokens:
                patch_tokens = hidden
            elif hidden.shape[1] >= self.num_prefix_tokens + expected_tokens:
                patch_tokens = hidden[
                    :, self.num_prefix_tokens : self.num_prefix_tokens + expected_tokens, :
                ]
            else:
                raise ValueError(
                    "DINOv3 output token count cannot be reshaped to a spatial feature map: "
                    f"got {hidden.shape[1]} tokens, expected {expected_tokens} patch tokens "
                    f"after {self.num_prefix_tokens} prefix tokens."
                )
            features = patch_tokens.reshape(x.shape[0], grid_h, grid_w, -1).permute(0, 3, 1, 2)

        return {self.out_feature: features}

    def output_shape(self):
        return {
            self.out_feature: ShapeSpec(
                channels=self._out_feature_channels[self.out_feature],
                stride=self._out_feature_strides[self.out_feature],
            )
        }
