# Copyright (c) Facebook, Inc. and its affiliates.
from typing import Dict, List, Optional

import torch

from detectron2.structures import Instances

from .rcnn import GeneralizedRCNN

__all__ = ["GeneralizedRCNNWithBBoxOnlyEval"]


class GeneralizedRCNNWithBBoxOnlyEval(GeneralizedRCNN):
    """
    GeneralizedRCNN variant that trains all configured heads, but skips mask
    prediction during inference/evaluation.

    This is useful for bbox-only COCO evaluation of mask models on very large
    images, where postprocessing predicted masks would materialize full-size
    bitmasks for every detection.
    """

    def inference(
        self,
        batched_inputs: List[Dict[str, torch.Tensor]],
        detected_instances: Optional[List[Instances]] = None,
        do_postprocess: bool = True,
    ):
        old_mask_on = getattr(self.roi_heads, "mask_on", None)
        if old_mask_on is None:
            return super().inference(batched_inputs, detected_instances, do_postprocess)

        self.roi_heads.mask_on = False
        try:
            return super().inference(batched_inputs, detected_instances, do_postprocess)
        finally:
            self.roi_heads.mask_on = old_mask_on
