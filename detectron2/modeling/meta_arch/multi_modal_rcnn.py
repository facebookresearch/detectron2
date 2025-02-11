from detectron2.modeling.meta_arch.rcnn import GeneralizedRCNN
from detectron2.modeling.backbone import Backbone
from detectron2.structures import ImageList, Instances
from detectron2.utils.events import get_event_storage
from transformers import BartTokenizer, BartModel
from sentence_transformers import SentenceTransformer
import torch
import math
import torch.nn as nn
from torch.nn.modules.module import _IncompatibleKeys as IncompatibleKeys
from typing import Optional
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# BART STUFF:
# self.tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
# self.model = BartModel.from_pretrained('facebook/bart-base')
# def forward(self, text):
# TODO: Could use encoded (or even last token) instead of `encoder_last_hidden_state`
# tokenizer, model, etc.
# last_hidden_state = outputs.encoder_last_hidden_state

# Max-pooling over the tokens
# pooled_features = torch.amax(last_hidden_state, dim=1)
PROCESSED_IMAGE_SIZE = (768, 1024)


class TextFeatureExtractor(torch.nn.Module):
    def __init__(self,  output_shape: int, expected_output_shape: tuple, fpn_keys: list[str], device: str, interpolation_mode: str = 'bilinear'):
        """
        we assume that the output_shape has shape flattened shape K (make sure)
        we assume that the expected_output_shape has shape (W, X, X) [does it work with w, x, y??]    
        """
        super().__init__()
        self.expected_output_shape = expected_output_shape
        
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.fpn_keys = fpn_keys
        self.interpolation_mode = interpolation_mode

        self.device = device

        self.remapper = nn.Sequential(
            torch.nn.Linear(output_shape, math.prod(expected_output_shape), device=device),
            nn.LeakyReLU()
        )

        num_fpn_layers = len(self.fpn_keys)
        self.fpn_convs = nn.ModuleDict()
        for i, k in enumerate(self.fpn_keys[:-1]):
            self.fpn_convs[k] = torch.nn.ConvTranspose2d(in_channels=1, out_channels=1, kernel_size=2**(num_fpn_layers-i-1), stride=2**(num_fpn_layers-i-1), device=device)

    def forward(self, batched_inputs: list[str], feature_shapes: dict[str, tuple]):
        """Forward pass for the text encoder.

        Args:
            batched_inputs (list[str]): A list of strings, each containing a text.
            feature_shapes (dict[str, tuple]): A dictionary containing the shapes of the backbone features.

        Returns:
            dict[str, torch.Tensor]: A dictionary containing the text features.
        """
        batched_reshaped_features = torch.zeros(((len(batched_inputs),) + self.expected_output_shape), device=self.device)
        for i, text in enumerate(batched_inputs):
            embedded_features = self.model.encode(text, convert_to_tensor=True)

            batched_reshaped_features[i] = self.remapper(embedded_features).view(self.expected_output_shape)
            
        return self.construct_fpn(batched_reshaped_features, feature_shapes)
    
    def construct_fpn(self, raw_tensor: torch.Tensor, feature_shapes: dict[str, tuple]):
        """Construct the FPN features from the raw tensor.

        Args:
            raw_tensor (torch.Tensor): The raw tensor of text features.
            feature_shapes (dict[str, tuple]): A dictionary containing the shapes of the backbone features.

        Returns:
            dict[str, torch.Tensor]: A dictionary containing the FPN features with the correct shapes.
        """
        fpn_features = {k: torch.zeros(v, device = self.device) for k, v in feature_shapes.items()}
        
        for i in range(raw_tensor.shape[0]):
            for k in self.fpn_keys:
                if k in self.fpn_convs:
                    fpn_features[k][i] = nn.functional.interpolate(self.fpn_convs[k](raw_tensor[i].unsqueeze(1)), size=(feature_shapes[k][-2], feature_shapes[k][-1]), mode=self.interpolation_mode)[:, 0]

                else:
                    fpn_features[k][i] = nn.functional.interpolate(raw_tensor[i].unsqueeze(0), size=(feature_shapes[k][-2], feature_shapes[k][-1]), mode=self.interpolation_mode)

        return fpn_features
    


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
        feature_dropout_rate: float = 0.0,
        feature_noise_std: float = 0.0,
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
            feature_dropout_rate: the rate to dropout backbone features
            feature_noise_std: the standard deviation of the noise to add to backbone features
            min_size_input: the minimum size of the input image
        """

        super().__init__(backbone=backbone, proposal_generator=proposal_generator, roi_heads=roi_heads, pixel_mean=pixel_mean, pixel_std=pixel_std, input_format=input_format, vis_period=vis_period)

        # Compute the expected output shape for the text encoder
        backbone_shapes = self.backbone.output_shape()
        channels = backbone_shapes[self.backbone._out_features[-1]].channels
        largest_stride = backbone_shapes[self.backbone._out_features[-1]].stride
        smallest_feature_size = (self.backbone._square_pad // largest_stride if self.backbone._square_pad else PROCESSED_IMAGE_SIZE[0] // largest_stride,
                                self.backbone._square_pad // largest_stride if self.backbone._square_pad else PROCESSED_IMAGE_SIZE[1] // largest_stride)

        self.feature_dropout = nn.Dropout(feature_dropout_rate)
        self.feature_noise_std = feature_noise_std
        # TODO: TextFeatureExtractor as input argument
        self.text_encoder = TextFeatureExtractor(
            output_shape=384,
            expected_output_shape=(channels, int(smallest_feature_size[0]), int(smallest_feature_size[1])), 
            fpn_keys=self.backbone._out_features,
            device="cuda")

    def forward(self, batched_inputs: list[dict[str, torch.Tensor]]):
        """Forward pass for the model.

        Args:
            batched_inputs (list[dict]): A list of dictionaries, each containing an image and its corresponding text.

        Returns:
            dict: A dictionary containing the losses.
        """

        if not self.training:
            return self.inference(batched_inputs)

        images = self.preprocess_image(batched_inputs)
        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None

        features = self.backbone(images.tensor)

        feature_shapes = {k: v.shape for k, v in features.items()}
        text_features = self.text_encoder([batched_input['text'] for batched_input in batched_inputs], feature_shapes)

        for k in features:
            # Dropout features
            features[k] = self.feature_dropout(features[k])

            # Add noise
            features[k] = features[k] + torch.randn_like(features[k]) * self.feature_noise_std

            features[k] = features[k] + text_features[k]
        
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

        feature_shapes = {k: v.shape for k, v in features.items()}
        text_features = self.text_encoder([batched_input['text'] for batched_input in batched_inputs], feature_shapes)

        for k in features:
            features[k] = features[k] + text_features[k]

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
    
    def load_state_dict(self, state_dict, strict: bool = True) -> IncompatibleKeys:
        """Load the state dict of the model.
        If the state dict does not contain text encoder weights, load the rest of the model normally.
        If the state dict contains text encoder weights, load the model normally.
        Args:
            state_dict (dict): The state dict of the model.
            strict (bool, optional): Whether to strictly enforce that the keys match exactly. Defaults to True.

        Returns:
            IncompatibleKeys: The missing (exist in model but not in state_dict) and unexpected keys (exist in state_dict but not in model).
        """
        if not any(k.startswith('text_encoder.') for k in state_dict):
            # If no text encoder weights in state dict, load everything else
            # and keep the pre-trained weights
            # TODO: distinguish between text_encoder.model and text_encoder.remapper
            result: IncompatibleKeys = super().load_state_dict(state_dict, strict=False)
            # Filter out text_encoder related missing keys from the report
            missing_keys = [k for k in result.missing_keys if not k.startswith('text_encoder.model')]
            return IncompatibleKeys(missing_keys=missing_keys, unexpected_keys=result.unexpected_keys)
        
        # If text encoder weights exist, load everything normally
        return super().load_state_dict(state_dict, strict)
