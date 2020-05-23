# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import logging
import torch
from torch import nn

from detectron2.structures import ImageList
from detectron2.utils.logger import log_first_n
from detectron2.data.datasets.lvis_categories_mapper import lvis_cate_mapper, cate_id_list

from ..backbone import build_backbone
from ..postprocessing import detector_postprocess
from ..proposal_generator import build_proposal_generator
from ..roi_heads import build_roi_heads
from .build import META_ARCH_REGISTRY


__all__ = ["ParallelRCNN"]


@META_ARCH_REGISTRY.register()
class ParallelRCNN(nn.Module):
    """
    Parallel R-CNN. Any models that contains the following three components:
    1. Per-image feature extraction (aka backbone)
    2. Region proposal generation
    3. Per-region feature extraction and prediction
    This model with different head for frequent, common, rare data.
    """

    def __init__(self, cfg):
        super().__init__()

        self.device = torch.device(cfg.MODEL.DEVICE)
        self.backbone = build_backbone(cfg)
        self.index_list = [cate_id_list()]
        self.proposal_generator_f = build_proposal_generator(cfg, self.backbone.output_shape())
        self.proposal_generator_c = build_proposal_generator(cfg, self.backbone.output_shape())
        self.proposal_generator_r = build_proposal_generator(cfg, self.backbone.output_shape())
        cfg.defrost()
        all_num_classes = cfg.MODEL.ROI_HEADS.NUM_CLASSES
        cfg = self.num_class_modifier(cfg, 'f')
        self.roi_heads_f = build_roi_heads(cfg, self.backbone.output_shape())
        cfg = self.num_class_modifier(cfg, 'c')
        self.roi_heads_c = build_roi_heads(cfg, self.backbone.output_shape())
        cfg = self.num_class_modifier(cfg, 'r')
        self.roi_heads_r = build_roi_heads(cfg, self.backbone.output_shape())
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = all_num_classes
        cfg.freeze()
        
        assert len(cfg.MODEL.PIXEL_MEAN) == len(cfg.MODEL.PIXEL_STD)
        num_channels = len(cfg.MODEL.PIXEL_MEAN)
        pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(self.device).view(num_channels, 1, 1)
        pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).to(self.device).view(num_channels, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std
        self.to(self.device)

    def num_class_modifier(self, cfg, mode):
        """
        Args:
            cfg -> config
            mode -> ['f', 'c', 'r']
        """
        if mode == 'f':
            cfg.MODEL.ROI_HEADS.NUM_CLASSES = cfg.DATASETS.NUM_CLASSES_F
        if mode == 'c':
            cfg.MODEL.ROI_HEADS.NUM_CLASSES = cfg.DATASETS.NUM_CLASSES_C
        if mode == 'r':
            cfg.MODEL.ROI_HEADS.NUM_CLASSES = cfg.DATASETS.NUM_CLASSES_R
        return cfg
        
    def forward(self, batched_inputs):
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

        images = self.preprocess_image(batched_inputs)
        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        elif "targets" in batched_inputs[0]:
            log_first_n(
                logging.WARN, "'targets' in the model inputs is now renamed to 'instances'!", n=10
            )
            gt_instances = [x["targets"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None
        
        gt_instances_splited = self.split_input_to_freq(gt_instances)
        gt_instances_f = gt_instances_splited[0]
        gt_instances_c = gt_instances_splited[1]
        gt_instances_r = gt_instances_splited[2]
        
        features = self.backbone(images.tensor)
        
        ########## Frequency data 
        if self.proposal_generator_f:
            proposals_f, proposal_losses_f = self.proposal_generator_f(images, features, gt_instances_f)
        else:
            assert "proposals" in batched_inputs[0]
            proposals_f = [x["proposals"].to(self.device) for x in batched_inputs]
            proposal_losses_f = {}

        _, detector_losses_f = self.roi_heads_f(images, features, proposals_f, gt_instances_f)

        ########## Common data 
        if self.proposal_generator_c:
            proposals_c, proposal_losses_c = self.proposal_generator_c(images, features, gt_instances_c)
        else:
            assert "proposals" in batched_inputs[0]
            proposals_c = [x["proposals"].to(self.device) for x in batched_inputs]
            proposal_losses_c = {}

        _, detector_losses_c = self.roi_heads_c(images, features, proposals_c, gt_instances_c)

        ########## Rare data 
        if self.proposal_generator_r:
            proposals_r, proposal_losses_r = self.proposal_generator_r(images, features, gt_instances_r)
        else:
            assert "proposals" in batched_inputs[0]
            proposals_r = [x["proposals"].to(self.device) for x in batched_inputs]
            proposal_losses_r = {}

        _, detector_losses_r = self.roi_heads_r(images, features, proposals_r, gt_instances_r)

#         detector_losses_total = self.combine_loss([detector_losses_f, detector_losses_c, detector_losses_r])
#         proposal_losses_total = self.combine_loss([proposal_losses_f, proposal_losses_c, proposal_losses_r])    

        losses = {}
        losses.update({k + "_f": v for k, v in proposal_losses_f.items()})
        losses.update({k + "_f": v for k, v in detector_losses_f.items()})
        losses.update({k + "_c": v for k, v in proposal_losses_c.items()})
        losses.update({k + "_c": v for k, v in detector_losses_c.items()})
        losses.update({k + "_r": v for k, v in proposal_losses_r.items()})
        losses.update({k + "_r": v for k, v in detector_losses_r.items()})

        return losses

    def inference(self, batched_inputs, detected_instances=None, do_postprocess=True):
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
            same as in :meth:`forward`.
        """
        assert not self.training

        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)

        ########## Frequency data
        if detected_instances is None:
            if self.proposal_generator_f:
                proposals_f, _ = self.proposal_generator_f(images, features, None)
            else:
                assert "proposals" in batched_inputs[0]
                proposals_f = [x["proposals"].to(self.device) for x in batched_inputs]

            results_f, _ = self.roi_heads_f(images, features, proposals_f, None)
        else:
            detected_instances = [x.to(self.device) for x in detected_instances]
            results_f = self.roi_heads_f.forward_with_given_boxes(features, detected_instances)
            
        ########## Common data
        if detected_instances is None:
            if self.proposal_generator_c:
                proposals_c, _ = self.proposal_generator_c(images, features, None)
            else:
                assert "proposals" in batched_inputs[0]
                proposals_c = [x["proposals"].to(self.device) for x in batched_inputs]

            results_c, _ = self.roi_heads_c(images, features, proposals_c, None)
        else:
            detected_instances = [x.to(self.device) for x in detected_instances]
            results_c = self.roi_heads_c.forward_with_given_boxes(features, detected_instances)
            
        ########## Rare data
        if detected_instances is None:
            if self.proposal_generator_r:
                proposals_r, _ = self.proposal_generator_r(images, features, None)
            else:
                assert "proposals" in batched_inputs[0]
                proposals_r = [x["proposals"].to(self.device) for x in batched_inputs]

            results_r, _ = self.roi_heads_r(images, features, proposals_r, None)
        else:
            detected_instances = [x.to(self.device) for x in detected_instances]
            results_r = self.roi_heads_r.forward_with_given_boxes(features, detected_instances)
                
        results = self.combine_result([results_f, results_c, results_r])
        
        if do_postprocess:
            processed_results = []
            for results_per_image, input_per_image, image_size in zip(
                results, batched_inputs, images.image_sizes
            ):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                r = detector_postprocess(results_per_image, height, width)
                processed_results.append({"instances": r})
            return processed_results
        else:
            return results

    def preprocess_image(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [self.normalizer(x) for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        return images
    
    def split_input_to_freq(self, batched_inputs):
        f = []; c = []; r = []
        for idx, Int in enumerate(batched_inputs):
            Int_split = Int.frequency_split()
            f.append(Int_split[0])
            c.append(Int_split[1])
            r.append(Int_split[2])
        return [f, c, r]
    
    def combine_loss(self, loss_list):
        dic_result = {}
        loss_names = loss_list[0].keys()
        for name in loss_names:
            dic_result[name] = loss_list[0][name]

        if len(loss_list) > 1:
            for idx in range(1,len(loss_list)):
                for name in loss_names:
                    new_loss = dic_result[name] + loss_list[idx][name]
                    dic_result[name] = new_loss
        return dic_result

    def combine_result(self, result_list):
        """
        Combine the result.
        Args:
            result_list (list[result]): same as in :meth:`forward`
        Returns:
        
        """
        assert len(result_list) > 0

        results = []
        for i in range(len(result_list)):
            assert len(result_list[0]) == len(result_list[i])
        num_batchs = len(result_list[0])
        
        I_type = type(result_list[0][0])
        for idx_batch in range(num_batchs):
            tmp_list = []
            for idx, result in enumerate(result_list):
                Inst = result[idx_batch]
                Inst.classes_reindex(self.index_list[0], idx)
                tmp_list.append(Inst)
            new_instance = I_type.cat(tmp_list)
            results.append(new_instance)
        
        return results
