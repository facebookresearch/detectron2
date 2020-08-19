# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from collections import OrderedDict

from detectron2.checkpoint import DetectionCheckpointer


def _rename_HRNet_weights(weights):
    # We detect and  rename HRNet weights for DensePose. 1956 and 1716 are values that are
    # common to all HRNet pretrained weights, and should be enough to accurately identify them
    if (
        len(weights["model"].keys()) == 1956
        and len([k for k in weights["model"].keys() if k.startswith("stage")]) == 1716
    ):
        hrnet_weights = OrderedDict()
        for k in weights["model"].keys():
            hrnet_weights["backbone.bottom_up." + str(k)] = weights["model"][k]
        return {"model": hrnet_weights}
    else:
        return weights


class DensePoseCheckpointer(DetectionCheckpointer):
    """
    Same as :class:`DetectionCheckpointer`, but is able to handle HRNet weights
    """

    def __init__(self, model, save_dir="", *, save_to_disk=None, **checkpointables):
        super().__init__(model, save_dir, save_to_disk=save_to_disk, **checkpointables)

    def _load_file(self, filename: str) -> object:
        """
        Adding hrnet support
        """
        weights = super()._load_file(filename)
        return _rename_HRNet_weights(weights)
