from typing import List

import torch
import numpy as np

from detectron2.config import get_cfg
from detectron2.data import detection_utils
from detectron2.modeling import build_model, detector_postprocess
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.structures import ImageList
import detectron2.data.transforms as T

cfg = get_cfg()
# Load cfg from a file
cfg.merge_from_file("config.yml")

model = build_model(cfg)

weights_file = "model_final.pth"
DetectionCheckpointer(model).load(weights_file)

# dummy_input = [torch.tensor(np.random.uniform(0, 1, (3, 800, 800)), dtype=torch.float32)]
# example_input = [torch.tensor(np.asarray(Image.open('roof_1.png').resize((800, 800))).transpose((2, 0, 1)))]
original_image = detection_utils.read_image('/Users/ranhomri/tensorleap/data/effizency-datasets/val/CH_192.png',
                                            format=cfg.INPUT.FORMAT)
aug = T.ResizeShortestEdge(
    [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
)
image = aug.get_transform(original_image).apply_image(original_image)
example_input = [torch.as_tensor(image.astype("float32").transpose(2, 0, 1))]


def preprocess_image(batched_inputs, pixel_mean, pixel_std):
    """
    Normalize, pad and batch the input images.
    """
    images = [(x - pixel_mean) / pixel_std for x in batched_inputs]

    return torch.stack(images, dim=0)


def bounding_box_to_polygon(bbox: np.ndarray) -> List[torch.Tensor]:
    x1, y1, x2, y2 = bbox
    polygon = torch.tensor([x1, y1, x1, y2, x2, y2, x2, y1], dtype=torch.float32)
    return [polygon]


def postprocess(instances, batched_inputs: List[torch.Tensor], image_sizes):
    """
    Rescale the output instances to the target size.
    """
    # note: private function; subject to changes
    processed_results = []
    for results_per_image, input_per_image, image_size in zip(
            instances, batched_inputs, image_sizes
    ):
        height = input_per_image.shape[1]
        width = input_per_image.shape[2]
        r = detector_postprocess(results_per_image, height, width)
        processed_results.append({"instances": r})
    return processed_results


class InferenceModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = model.backbone
        self.proposal_generator = model.proposal_generator
        self.roi_heads = model.roi_heads

    def forward(self, batched_input):
        images = [x for x in batched_input]
        images = ImageList.from_tensors(
            images,
            32,
            padding_constraints={'square_size': 0},
        )
        features = self.backbone(images.tensor)
        proposals, _, anchors, pred_objectness_logits, pred_anchor_deltas = self.proposal_generator(images, features,
                                                                                                    None)
        anchors = torch.cat([a.tensor for a in anchors], 0)
        pred_objectness_logits = torch.cat(pred_objectness_logits, 1)
        pred_anchor_deltas = torch.cat(pred_anchor_deltas, 1)
        results, _, cls_box_loss_predictions, mask_loss_features, mask_loss_instances = self.roi_heads(images, features,
                                                                                                       proposals, None)
        cls_loss_predictions = cls_box_loss_predictions[0]
        box_loss_predictions = cls_box_loss_predictions[1]
        mask_loss_instances = mask_loss_instances[0].get_fields()
        results = results[0].get_fields()
        return results['pred_boxes'].tensor, results['scores'], results['pred_classes'], results['pred_masks'], \
            anchors, pred_objectness_logits, pred_anchor_deltas, cls_loss_predictions, box_loss_predictions, \
            mask_loss_features, mask_loss_instances['proposal_boxes'].tensor, mask_loss_instances['objectness_logits']


pixel_mean = torch.tensor([103.53, 116.28, 123.675]).unsqueeze(1).unsqueeze(1)
pixel_std = torch.tensor([1.0, 1.0, 1.0]).unsqueeze(1).unsqueeze(1)
processed_input = preprocess_image(example_input, pixel_mean, pixel_std)

inference_model = InferenceModel().eval()

infer_output = inference_model(processed_input)

output_names = ['pred_boxes', 'scores', 'pred_classes', 'pred_masks', 'anchors', 'pred_objectness_logits',
                'pred_anchor_deltas', 'cls_loss_predictions', 'box_loss_predictions', 'mask_loss_features',
                'proposal_boxes', 'proposal_logits']

assert len(output_names) == len(infer_output)
torch.onnx.export(model=inference_model,
                  args=processed_input,
                  f='eval_model.onnx',
                  input_names=['image'],
                  output_names=output_names,
                  opset_version=11)
