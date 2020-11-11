import torch

from detectron2 import model_zoo
from detectron2.modeling import build_model


"""
Internal utilities for tests.
"""


def get_model_no_weights(config_path):
    """
    Like model_zoo.get, but do not load any weights (even pretrained)
    """
    cfg = model_zoo.get_config(config_path)
    if not torch.cuda.is_available():
        cfg.MODEL.DEVICE = "cpu"
    return build_model(cfg)
