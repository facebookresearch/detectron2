# Copyright (c) Facebook, Inc. and its affiliates.
import logging
from random import choice
import numpy as np
from typing import Dict, List, Optional, Tuple
import torch
from torch import nn
import torch.nn.functional as F

from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.data.detection_utils import convert_image_to_rgb
from detectron2.layers import move_device_like
from detectron2.structures import Instances, Boxes
from detectron2.utils.events import get_event_storage
from detectron2.utils.file_io import PathManager

from detectron2.modeling.backbone import Backbone, build_backbone
from detectron2.modeling.postprocessing import detector_postprocess
from detectron2.modeling.proposal_generator import build_proposal_generator
from detectron2.modeling.roi_heads import build_roi_heads
from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY

from densepose.structures.image_list import ImageList
from densepose.structures import DensePoseTransformData

__all__ = ["GeneralizedRCNNDP"]

POINT_LABEL_SYMMETRIES = [0, 1, 2, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15, 18, 17, 20, 19, 22, 21, 24, 23]

@META_ARCH_REGISTRY.register()
class GeneralizedRCNNDP(nn.Module):
    """
    Generalized R-CNN. Any models that contains the following three components:
    1. Per-image feature extraction (aka backbone)
    2. Region proposal generation
    3. Per-region feature extraction and prediction
    """

    @configurable
    def __init__(
        self,
        *,
        backbone: Backbone,
        proposal_generator: nn.Module,
        roi_heads: nn.Module,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        input_format: Optional[str] = None,
        vis_period: int = 0,
        unlabeled_threshold: int = 10000,
        total_iteration:int = 260000,
        ds = "densepose_coco_2014_train",
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
        super().__init__()
        self.backbone = backbone
        self.proposal_generator = proposal_generator
        self.roi_heads = roi_heads

        self.input_format = input_format
        self.vis_period = vis_period
        if vis_period > 0:
            assert input_format is not None, "input_format is required for visualization!"

        self.register_buffer("pixel_mean", torch.tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.tensor(pixel_std).view(-1, 1, 1), False)
        assert (
            self.pixel_mean.shape == self.pixel_std.shape
        ), f"{self.pixel_mean} and {self.pixel_std} have different shapes!"

        self.unlabeled_threshold = unlabeled_threshold
        self.iteration = 0
        self.total_iteration = total_iteration

        densepose_transform_data_fpath = PathManager.get_local_path(MetadataCatalog.get(ds).densepose_transform_src)
        self.uv_symmetries = DensePoseTransformData.load(
            densepose_transform_data_fpath
        ).uv_symmetries

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        return {
            "backbone": backbone,
            "proposal_generator": build_proposal_generator(cfg, backbone.output_shape()),
            "roi_heads": build_roi_heads(cfg, backbone.output_shape()),
            "input_format": cfg.INPUT.FORMAT,
            "vis_period": cfg.VIS_PERIOD,
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            "unlabeled_threshold": cfg.MODEL.SEMI.UNLABELED_THRESHOLD
        }

    def update_iteration(self, iteration):
        self.iteration = iteration

    @property
    def device(self):
        return self.pixel_mean.device

    def _move_to_current_device(self, x):
        return move_device_like(x, self.pixel_mean)

    def visualize_training(self, batched_inputs, proposals):
        """
        A function used to visualize images and proposals. It shows ground truth
        bounding boxes on the original image and up to 20 top-scoring predicted
        object proposals on the original image. Users can implement different
        visualization functions for different models.

        Args:
            batched_inputs (list): a list that contains input to the model.
            proposals (list): a list that contains predicted proposals. Both
                batched_inputs and proposals should have the same length.
        """
        from detectron2.utils.visualizer import Visualizer

        storage = get_event_storage()
        max_vis_prop = 20

        for input, prop in zip(batched_inputs, proposals):
            img = input["image"]
            img = convert_image_to_rgb(img.permute(1, 2, 0), self.input_format)
            v_gt = Visualizer(img, None)
            v_gt = v_gt.overlay_instances(boxes=input["instances"].gt_boxes)
            anno_img = v_gt.get_image()
            box_size = min(len(prop.proposal_boxes), max_vis_prop)
            v_pred = Visualizer(img, None)
            v_pred = v_pred.overlay_instances(
                boxes=prop.proposal_boxes[0:box_size].tensor.cpu().numpy()
            )
            prop_img = v_pred.get_image()
            vis_img = np.concatenate((anno_img, prop_img), axis=1)
            vis_img = vis_img.transpose(2, 0, 1)
            vis_name = "Left: GT bounding boxes;  Right: Predicted proposals"
            storage.put_image(vis_name, vis_img)
            break  # only visualize one image in a batch

    def forward(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances (optional): groundtruth :class:`Instances`
                * proposals (optional): :class:`Instances`, precomputed proposals.

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.

        Returns:
            list[dict]:
                Each dict is the output for one input image.
                The dict contains one key "instances" whose value is a :class:`Instances`.
                The :class:`Instances` object has the following keys:
                "pred_boxes", "pred_classes", "scores", "pred_masks", "pred_keypoints"
        """
        if not self.training:
            return self.inference(batched_inputs)

        # batched_inputs, pseudo_info = batched_inputs[:-1], batched_inputs[-1]

        for k in self.uv_symmetries.keys():
            self.uv_symmetries[k] = self.uv_symmetries[k].to(self.device)

        do_flip = choice([True, False])
        # do_flip = False

        images = self.preprocess_image(batched_inputs, do_flip=do_flip)
        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
            un_instances = [x["un_instances"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None
            un_instances = None

        features = self.backbone(images.tensor)
        batch_size = images.tensor.shape[0] // 2
        label_features = {}
        for k in features.keys():
            label_features[k] = features[k][:batch_size]
            features[k] = features[k][batch_size:]

        if self.proposal_generator is not None:
            proposals, proposal_losses = self.proposal_generator(images[:batch_size], label_features, gt_instances)
        else:
            assert "proposals" in batched_inputs[0]
            proposals = [x["proposals"].to(self.device) for x in batched_inputs]
            proposal_losses = {}

        _, detector_losses = self.roi_heads(images[:batch_size], label_features, proposals, gt_instances)

        labeled_boxes = [x.labeled_boxes for x in un_instances]
        if do_flip:
            unlabeled_boxes = [x.unlabeled_boxes.h_flip(x.image_size[1]) for x in un_instances]
        else:
            unlabeled_boxes = [x.unlabeled_boxes for x in un_instances]

        if len(labeled_boxes) <= 0 or len(unlabeled_boxes) <= 0:
            unlabeled_loss = self.get_fake_unsup_loss()
        else:
            with torch.no_grad():
                pseudo_labels = self.roi_heads.forward_with_given_boxes_train(label_features, labeled_boxes)
                pseudo_labels.rotate(labeled_boxes, [x['angle'] for x in batched_inputs])
            prediction = self.roi_heads.forward_with_given_boxes_train(features, unlabeled_boxes)
            unlabeled_loss = self.get_unlabeled_loss(pseudo_labels, prediction, do_flip=do_flip)

        if self.vis_period > 0:
            storage = get_event_storage()
            if storage.iter % self.vis_period == 0:
                self.visualize_training(batched_inputs, proposals)

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)
        losses.update(unlabeled_loss)
        return losses

    def inference(
        self,
        batched_inputs: List[Dict[str, torch.Tensor]],
        detected_instances: Optional[List[Instances]] = None,
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
            return GeneralizedRCNNDP._postprocess(results, batched_inputs, images.image_sizes)
        return results

    def preprocess_image(self, batched_inputs: List[Dict[str, torch.Tensor]], do_flip=False):
        """
        Normalize, pad and batch the input images.
        """
        images = [self._move_to_current_device(x["image"]) for x in batched_inputs]
        if self.training:
            if do_flip:
                images += [self._move_to_current_device(torch.flip(x["un_image"], dims=[-1])) for x in batched_inputs]
            else:
                images += [self._move_to_current_device(x["un_image"]) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(
            images,
            self.backbone.size_divisibility,
            padding_constraints=self.backbone.padding_constraints,
        )
        return images

    @staticmethod
    def _postprocess(instances, batched_inputs: List[Dict[str, torch.Tensor]], image_sizes):
        """
        Rescale the output instances to the target size.
        """
        # note: private function; subject to changes
        processed_results = []
        for results_per_image, input_per_image, image_size in zip(
            instances, batched_inputs, image_sizes
        ):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            r = detector_postprocess(results_per_image, height, width)
            processed_results.append({"instances": r})
        return processed_results

    def get_unlabeled_loss(self, pseudo_labels, prediction, do_flip=False):
        n_channels = 25
        # factor = np.exp(-5 * (1 - self.iteration / self.total_iteration) ** 2) * 0.1
        if (self.iteration + 1) <= 80000:
            factor = np.exp(-5 * (1 - self.iteration / 80000) ** 2) * 0.1
        # elif (self.iteration + 1) >= 220000:
        #     factor = np.exp(-12.5 * (1 - (self.iteration - 179999) / 40000) ** 2) * 0.1
        else:
            factor = 0.05
        # factor = 0.5

        # threshold = np.exp(-5 * (1 - self.iteration / self.total_iteration) ** 2) * 0.25 + 0.7
        threshold = 0.85

        est = prediction.fine_segm.permute(0, 2, 3, 1).reshape(-1, n_channels)
        coarse_est = prediction.coarse_segm.permute(0, 2, 3, 1).reshape(-1, 2)
        with torch.no_grad():
            if do_flip:
                pseudo_fine_segm = torch.flip(pseudo_labels.fine_segm, dims=[3])[:, POINT_LABEL_SYMMETRIES]
                pos_index = torch.flip(pseudo_labels.crt_segm, dims=[3])
                pos_index[:, :n_channels] = pos_index[:, POINT_LABEL_SYMMETRIES]
                pseudo_coarse_segm = torch.flip(pseudo_labels.coarse_segm, dims=[3])
            else:
                pseudo_fine_segm = pseudo_labels.fine_segm
                pos_index = pseudo_labels.crt_segm
                pseudo_coarse_segm = pseudo_labels.coarse_segm
            pseudo_fine_segm = pseudo_fine_segm.permute(0, 2, 3, 1).reshape(-1, n_channels)
            pos_index = pos_index.permute(0, 2, 3, 1).reshape(-1, n_channels + 1)
            pseudo_coarse_segm = pseudo_coarse_segm.permute(0, 2, 3, 1).reshape(-1, 2).argmax(dim=1)

            pred_index = pseudo_fine_segm.argmax(dim=1).long()

            coarse_pos_index = torch.sigmoid(pos_index[:, -1]) >= threshold
            pos_index = pos_index[torch.arange(pos_index.shape[0]), pred_index]
            pos_index = torch.sigmoid(pos_index) >= threshold

        if coarse_pos_index.sum() <= 0:
            losses = {
                "loss_unsup_coarse_segm": coarse_est.sum() * 0,
            }
        else:
            losses = {
                "loss_unsup_coarse_segm": F.cross_entropy(
                    coarse_est[coarse_pos_index], pseudo_coarse_segm[coarse_pos_index]
                ) * 5.0 * factor,
            }
        pos_index = pos_index * coarse_est.argmax(dim=1).bool() * coarse_pos_index
        pred_index = pred_index[pos_index]
        if pos_index.sum() <= 0:
            losses.update({
                "loss_unsup_fine_segm": est.sum() * 0,
                "loss_unsup_u": prediction.u.sum() * 0,
                "loss_unsup_v": prediction.v.sum() * 0,
            })
        else:
            loss = F.cross_entropy(est[pos_index], pred_index.long(), reduction='mean')
            losses.update({
                "loss_unsup_fine_segm": loss * factor
            })

            u_est = prediction.u.permute(0, 2, 3, 1).reshape(-1, n_channels)[pos_index]
            v_est = prediction.v.permute(0, 2, 3, 1).reshape(-1, n_channels)[pos_index]

            # pred_index = torch.zeros_like(u_est).scatter_(1, pred_index.unsqueeze(1), 1).bool()

            batch_indices = torch.arange(pred_index.shape[0]).to(self.device)
            u_est = u_est[batch_indices, pred_index]
            v_est = v_est[batch_indices, pred_index]

            with torch.no_grad():
                if do_flip:
                    pseudo_u = torch.flip(pseudo_labels.u, dims=[3])[:, POINT_LABEL_SYMMETRIES]
                    pseudo_v = torch.flip(pseudo_labels.v, dims=[3])[:, POINT_LABEL_SYMMETRIES]
                    pseudo_sigma = torch.flip(pseudo_labels.crt_sigma, dims=[3])[:, POINT_LABEL_SYMMETRIES]
                else:
                    pseudo_u = pseudo_labels.u
                    pseudo_v = pseudo_labels.v
                    pseudo_sigma = pseudo_labels.crt_sigma
                pseudo_u = pseudo_u.permute(0, 2, 3, 1).reshape(-1, n_channels)[pos_index]
                pseudo_v = pseudo_v.permute(0, 2, 3, 1).reshape(-1, n_channels)[pos_index]
                pseudo_sigma = pseudo_sigma.permute(0, 2, 3, 1).reshape(-1, n_channels)[pos_index]

                pseudo_u = pseudo_u[batch_indices, pred_index]#.clamp(0., 1.)
                pseudo_v = pseudo_v[batch_indices, pred_index]#.clamp(0., 1.)

                pseudo_sigma = pseudo_sigma[batch_indices, pred_index]
                pseudo_sigma = torch.pow(1 - pseudo_sigma.clamp(0., 1.), 9)
                # pseudo_sigma = (1 / (pseudo_sigma.clip(0., 1.) + 0.1))

            if do_flip:
                # good_indices = (pseudo_u >= 0.) * (pseudo_u <= 1.) * (pseudo_v >= 0.) * (pseudo_v <= 1.)
                for i in range(1, len(POINT_LABEL_SYMMETRIES)):
                    indice = pred_index == POINT_LABEL_SYMMETRIES[i]
                    u_loc = (pseudo_u[indice] * 255).clip(0, 255).long()
                    v_loc = (pseudo_v[indice] * 255).clip(0, 255).long()
                    pseudo_u[indice] = self.uv_symmetries["U_transforms"][i - 1][v_loc, u_loc]
                    pseudo_v[indice] = self.uv_symmetries["V_transforms"][i - 1][v_loc, u_loc]

                loss_u = F.mse_loss(u_est, pseudo_u, reduction='none') * pseudo_sigma
                loss_v = F.mse_loss(v_est, pseudo_v, reduction='none') * pseudo_sigma
            else:
                loss_u = F.mse_loss(u_est, pseudo_u, reduction='none') * pseudo_sigma
                loss_v = F.mse_loss(v_est, pseudo_v, reduction='none') * pseudo_sigma

            losses.update({
                "loss_unsup_u": loss_u.sum() * 0.01 * factor,
                "loss_unsup_v": loss_v.sum() * 0.01 * factor
            })

        return losses

    def get_fake_unsup_loss(self):
        return {
            "loss_unsup_coarse_segm": 0,
            "loss_unsup_fine_segm": 0,
            "loss_unsup_u": 0,
            "loss_unsup_v": 0,
        }
