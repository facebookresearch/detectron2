from detectron2.modeling.meta_arch.rcnn import GeneralizedRCNN
from detectron2.modeling.backbone import Backbone
from detectron2.structures import ImageList, Instances
from detectron2.utils.events import get_event_storage
from transformers import BartTokenizer, BartModel
import torch
import torch.nn as nn
from typing import Optional


class TextFeatureExtractor(torch.nn.Module):
    def __init__(self,  output_shape: int, expected_output_shape: tuple, device: str):
        """
        we assume that the output_shape has shape flattened shape K (make sure it)
        we assume that the expected_output_shape has shape (W, X, X)
        """
        super().__init__()
        self.expected_output_shape = expected_output_shape
        self.tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
        self.model = BartModel.from_pretrained('facebook/bart-base')
        expected_flattened_output_shape = 1
        self.device = device
        for s in expected_output_shape:
            expected_flattened_output_shape = expected_flattened_output_shape * s

        self.remapper = torch.nn.Linear(output_shape, expected_flattened_output_shape)

    def forward(self, text):
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)

        outputs = self.model(**inputs, output_hidden_states=True)
        last_hidden_state = outputs.encoder_last_hidden_state.squeeze() # could use encoded instead
        remapped_state = torch.amax(last_hidden_state, dim=0)
        return self.remapper(remapped_state).view(self.expected_output_shape)
    

class MultiModalRCNN(GeneralizedRCNN):
    def __init__(
        self,
        *,
        backbone: Backbone,
        proposal_generator: nn.Module,
        roi_heads: nn.Module,
        pixel_mean: tuple[float],
        pixel_std: tuple[float],
        input_format: Optional[str] = None,
        vis_period: int = 0,
    ):
        """
        Args:
            backbone: a backbone module, must follow detectron2's backbone interface
            proposal_generator: a module that generates proposals using backbone features
            roi_heads: a ROI head that performs per-region computation
            pixel_mean, pixel_std: list or tuple with #channels element, representing
                the per-channel mean and std to be used to normalize the input image
            input_format: describe the meaning of channels of input. Needed by visualization
            vis_period: the period to run visualization. Set to 0 to disable.
        """

        super().__init__(backbone=backbone, proposal_generator=proposal_generator, roi_heads=roi_heads, pixel_mean=pixel_mean, pixel_std=pixel_std, input_format=input_format, vis_period=vis_period)

        self.text_encoder = TextFeatureExtractor(output_shape=768, expected_output_shape=(256, 16, 16), device="cuda")

    def forward(self, batched_inputs):

        if not self.training:
            return self.inference(batched_inputs)

        images = self.preprocess_image(batched_inputs)
        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None

        features = self.backbone(images.tensor)

        text_features = self.text_encoder([batched_input['item_description'] for batched_input in batched_inputs])
        
        features['p6'] += text_features
        
        if self.proposal_generator is not None:
            proposals, proposal_losses = self.proposal_generator(images, features, gt_instances)
        else:
            assert "proposals" in batched_inputs[0]
            proposals = [x["proposals"].to(self.device) for x in batched_inputs]
            proposal_losses = {}

        _, detector_losses = self.roi_heads(images, features, proposals, gt_instances)
        if self.vis_period > 0:
            storage = get_event_storage()
            if storage.iter % self.vis_period == 0:
                self.visualize_training(batched_inputs, proposals)

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)
        return losses
    

    def inference(
        self,
        batched_inputs: list[dict[str, torch.Tensor]],
        detected_instances: Optional[list[Instances]] = None,
        do_postprocess: bool = True,
    ):
        """
        Run inference on the given inputs.

        Args:
            batched_inputs (list[dict]): same as in :meth:`forward`
            detected_instances (None or list[Instances]): if not None, it
                contains an `Instances` object per image. The `Instances`
                object contains "pred_boxes" and "pred_classes" which are
                known boxes in the image.
                The inference will then skip the detection of bounding boxes,
                and only predict other per-ROI outputs.
            do_postprocess (bool): whether to apply post-processing on the outputs.

        Returns:
            When do_postprocess=True, same as in :meth:`forward`.
            Otherwise, a list[Instances] containing raw network outputs.
        """
        assert not self.training

        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)

        text_features = self.text_encoder([batched_input['item_description'] for batched_input in batched_inputs]).unsqueeze(dim=0)

        features['p6'] += text_features

        if detected_instances is None:
            if self.proposal_generator is not None:
                proposals, _ = self.proposal_generator(images, features, None)
            else:
                assert "proposals" in batched_inputs[0]
                proposals = [x["proposals"].to(self.device) for x in batched_inputs]

            results, _ = self.roi_heads(images, features, proposals, None)
        else:
            detected_instances = [x.to(self.device) for x in detected_instances]
            results = self.roi_heads.forward_with_given_boxes(features, detected_instances)

        if do_postprocess:
            assert not torch.jit.is_scripting(), "Scripting is not supported for postprocess."
            return GeneralizedRCNN._postprocess(results, batched_inputs, images.image_sizes)
        return results