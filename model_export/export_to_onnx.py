import json
from typing import List, Dict, Tuple

import onnx
import torch
import numpy as np
from PIL import Image
from torch import Tensor
from torch._C._onnx import TrainingMode

from detectron2.config import get_cfg
from detectron2.data import detection_utils
from detectron2.export import scripting_with_instances, TracingAdapter
from detectron2.modeling import build_model, detector_postprocess
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.structures import ImageList, Instances, BitMasks, PolygonMasks, Boxes
import detectron2.data.transforms as T

cfg = get_cfg()
# Load cfg from a file
cfg.merge_from_file("config.yml")

model = build_model(cfg)

weights_file = "model_final.pth"
DetectionCheckpointer(model).load(weights_file)

# fields = {'image_size': torch.Size, 'gt_boxes': Boxes, 'gt_classes': torch.Tensor, 'gt_masks': PolygonMasks}
# scripting_with_instances(model, fields)

# image = torch.tensor(np.asarray(Image.open('roof_1.png')).transpose((2, 0, 1)), dtype=torch.float32)
original_image = detection_utils.read_image('/Users/ranhomri/tensorleap/data/effizency-datasets/train/CH_1.png',
                                            format=cfg.INPUT.FORMAT)
aug = T.ResizeShortestEdge(
    [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
)
image = aug.get_transform(original_image).apply_image(original_image)
image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))

inputs = {"image": image}

batched_input = [inputs]


model.eval()
eval_output = model(batched_input)
#
#
# def inference(model, inputs):
#     # use do_postprocess=False so it returns ROI mask
#     inst = model.inference(inputs, do_postprocess=True)[0]
#     return [{"instances": inst}]
#
#
# # traceable_model = TracingAdapter(model, batched_input, inference)
# dynamic_axes = {
#     'image': {
#         0: 'batch',
#         2: 'w',
#         3: 'h'
#     }}


def get_bboxes(annotations: List[Dict]) -> np.ndarray:
    polygons = [ann['points'] for ann in annotations]
    bboxes = np.asarray([polygon2bbox(polygon) for polygon in polygons])
    return bboxes


def polygon2bbox(polygon: List[List[int]]) -> np.ndarray:
    """
    Converts a polygon to a bounding box.

    Args:
        polygon (List[List[int]]): A list of (x, y) coordinates representing a polygon.

    Returns:
        np.ndarray: A numpy array of shape (4,) representing the bounding box in (x1, y1, x2, y2) format.
    """

    # Convert polygon to numpy array
    polygon = np.array(polygon)

    # Calculate bounding box coordinates
    x1 = np.min(polygon[:, 0])
    y1 = np.min(polygon[:, 1])
    x2 = np.max(polygon[:, 0])
    y2 = np.max(polygon[:, 1])

    return np.array([x1, y1, x2, y2])


gt_file_path = '/Users/ranhomri/tensorleap/data/effizency-datasets/train/CH_1.json'
with open(gt_file_path, 'r') as f:
    gt = json.load(f)

boxes = Boxes(get_bboxes(gt['shapes']))
classes = torch.tensor([0 for ann in gt['shapes']])
polygons = [[np.asarray(m['points']).flatten()] for m in gt['shapes']]
polygons = PolygonMasks(polygons)
gt_instances = Instances(image_size=batched_input[0]['image'].shape[-2:],
                         gt_boxes=boxes,
                         gt_classes=classes,
                         gt_masks=polygons)
batched_input[0]['instances'] = gt_instances


model.train()
train_output = model(batched_input)
a = 0
