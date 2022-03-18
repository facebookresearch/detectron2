# Copyright (c) Facebook, Inc. and its affiliates.

from dataclasses import make_dataclass
from functools import lru_cache
from typing import Any, Optional
import torch


@lru_cache(maxsize=None)
def decorate_predictor_output_class_with_confidences(BasePredictorOutput: type) -> type:
    """
    Create a new output class from an existing one by adding new attributes
    related to confidence estimation:
    - sigma_1 (tensor)
    - sigma_2 (tensor)
    - kappa_u (tensor)
    - kappa_v (tensor)
    - fine_segm_confidence (tensor)
    - coarse_segm_confidence (tensor)

    Details on confidence estimation parameters can be found in:
    N. Neverova, D. Novotny, A. Vedaldi "Correlated Uncertainty for Learning
        Dense Correspondences from Noisy Labels", p. 918--926, in Proc. NIPS 2019
    A. Sanakoyeu et al., Transferring Dense Pose to Proximal Animal Classes, CVPR 2020

    The new class inherits the provided `BasePredictorOutput` class,
    it's name is composed of the name of the provided class and
    "WithConfidences" suffix.

    Args:
        BasePredictorOutput (type): output type to which confidence data
            is to be added, assumed to be a dataclass
    Return:
        New dataclass derived from the provided one that has attributes
        for confidence estimation
    """

    PredictorOutput = make_dataclass(
        BasePredictorOutput.__name__ + "WithConfidences",
        fields=[  # pyre-ignore[6]
            ("sigma_1", Optional[torch.Tensor], None),
            ("sigma_2", Optional[torch.Tensor], None),
            ("kappa_u", Optional[torch.Tensor], None),
            ("kappa_v", Optional[torch.Tensor], None),
            ("fine_segm_confidence", Optional[torch.Tensor], None),
            ("coarse_segm_confidence", Optional[torch.Tensor], None),
        ],
        bases=(BasePredictorOutput,),
    )

    # add possibility to index PredictorOutput

    def slice_if_not_none(data, item):
        if data is None:
            return None
        if isinstance(item, int):
            return data[item].unsqueeze(0)
        return data[item]

    def PredictorOutput_getitem(self, item):
        PredictorOutput = type(self)
        base_predictor_output_sliced = super(PredictorOutput, self).__getitem__(item)
        return PredictorOutput(
            **base_predictor_output_sliced.__dict__,
            coarse_segm_confidence=slice_if_not_none(self.coarse_segm_confidence, item),
            fine_segm_confidence=slice_if_not_none(self.fine_segm_confidence, item),
            sigma_1=slice_if_not_none(self.sigma_1, item),
            sigma_2=slice_if_not_none(self.sigma_2, item),
            kappa_u=slice_if_not_none(self.kappa_u, item),
            kappa_v=slice_if_not_none(self.kappa_v, item),
        )

    PredictorOutput.__getitem__ = PredictorOutput_getitem

    def PredictorOutput_to(self, device: torch.device):
        """
        Transfers all tensors to the given device
        """
        PredictorOutput = type(self)
        base_predictor_output_to = super(PredictorOutput, self).to(device)  # pyre-ignore[16]

        def to_device_if_tensor(var: Any):
            if isinstance(var, torch.Tensor):
                return var.to(device)
            return var

        return PredictorOutput(
            **base_predictor_output_to.__dict__,
            sigma_1=to_device_if_tensor(self.sigma_1),
            sigma_2=to_device_if_tensor(self.sigma_2),
            kappa_u=to_device_if_tensor(self.kappa_u),
            kappa_v=to_device_if_tensor(self.kappa_v),
            fine_segm_confidence=to_device_if_tensor(self.fine_segm_confidence),
            coarse_segm_confidence=to_device_if_tensor(self.coarse_segm_confidence),
        )

    PredictorOutput.to = PredictorOutput_to
    return PredictorOutput
