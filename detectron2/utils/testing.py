import torch

from detectron2 import model_zoo
from detectron2.modeling import build_model


"""
Internal utilities for tests. Don't use except for writing tests.
"""


def get_model_no_weights(config_path):
    """
    Like model_zoo.get, but do not load any weights (even pretrained)
    """
    cfg = model_zoo.get_config(config_path)
    if not torch.cuda.is_available():
        cfg.MODEL.DEVICE = "cpu"
    return build_model(cfg)


def random_boxes(num_boxes, max_coord=100, device="cpu"):
    """
    Create a random Nx4 boxes tensor, with coordinates < max_coord.
    """
    boxes = torch.rand(num_boxes, 4, device=device) * (max_coord * 0.5)
    # Note: the implementation of this function in torchvision is:
    # boxes[:, 2:] += torch.rand(N, 2) * 100
    # but it does not guarantee non-negative widths/heights constraints:
    # boxes[:, 2] >= boxes[:, 0] and boxes[:, 3] >= boxes[:, 1]:
    boxes[:, 2:] += boxes[:, :2]
    return boxes
