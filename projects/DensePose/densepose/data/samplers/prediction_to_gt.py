# Copyright (c) Facebook, Inc. and its affiliates.

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

from detectron2.structures import Instances

ModelOutput = Dict[str, Any]
SampledData = Dict[str, Any]


@dataclass
class _Sampler:
    """
    Sampler registry entry that contains:
     - src (str): source field to sample from (deleted after sampling)
     - dst (Optional[str]): destination field to sample to, if not None
     - func (Optional[Callable: Any -> Any]): function that performs sampling,
         if None, reference copy is performed
    """

    src: str
    dst: Optional[str]
    func: Optional[Callable[[Any], Any]]


class PredictionToGroundTruthSampler:
    """
    Sampler implementation that converts predictions to GT using registered
    samplers for different fields of `Instances`.
    """

    def __init__(self, dataset_name: str = ""):
        self.dataset_name = dataset_name
        self._samplers = {}
        self.register_sampler("pred_boxes", "gt_boxes", None)
        self.register_sampler("pred_classes", "gt_classes", None)
        self.register_sampler("scores")

    def __call__(self, model_output: ModelOutput) -> SampledData:
        """
        Transform model output into ground truth data through sampling

        Args:
          model_output (Dict[str, Any]): model output
        Returns:
          Dict[str, Any]: sampled data
        """
        for model_output_i in model_output:
            instances: Instances = model_output_i["instances"]
            # transform data in each field
            for _, sampler in self._samplers.items():
                if not instances.has(sampler.src) or sampler.dst is None:
                    continue
                if sampler.func is None:
                    instances.set(sampler.dst, instances.get(sampler.src))
                else:
                    instances.set(sampler.dst, sampler.func(instances))
            # delete model output data that was transformed
            for _, sampler in self._samplers.items():
                if sampler.src != sampler.dst and instances.has(sampler.src):
                    instances.remove(sampler.src)
            model_output_i["dataset"] = self.dataset_name
        return model_output

    def register_sampler(
        self,
        prediction_attr: str,
        gt_attr: Optional[str] = None,
        func: Optional[Callable[[Any], Any]] = None,
    ):
        """
        Register sampler for a field

        Args:
          prediction_attr (str): field to replace with a sampled value
          gt_attr (Optional[str]): field to store the sampled value to, if not None
          func (Optional[Callable: Any -> Any]): sampler function
        """
        self._samplers[prediction_attr] = _Sampler(src=prediction_attr, dst=gt_attr, func=func)
