# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch
from fvcore.transforms import HFlipTransform

from detectron2.modeling.postprocessing import detector_postprocess
from detectron2.modeling.test_time_augmentation import GeneralizedRCNNWithTTA


class DensePoseGeneralizedRCNNWithTTA(GeneralizedRCNNWithTTA):
    def __init__(self, cfg, model, transform_data, tta_mapper=None, batch_size=1):
        """
        Args:
            cfg (CfgNode):
            model (GeneralizedRCNN): a GeneralizedRCNN to apply TTA on.
            transform_data (DensePoseTransformData): contains symmetry label
                transforms used for horizontal flip
            tta_mapper (callable): takes a dataset dict and returns a list of
                augmented versions of the dataset dict. Defaults to
                `DatasetMapperTTA(cfg)`.
            batch_size (int): batch the augmented images into this batch size for inference.
        """
        self._transform_data = transform_data.to(model.device)
        super().__init__(cfg=cfg, model=model, tta_mapper=tta_mapper, batch_size=batch_size)

    # the implementation follows closely the one from detectron2/modeling
    def _inference_one_image(self, input):
        """
        Args:
            input (dict): one dataset dict with "image" field being a CHW tensor

        Returns:
            dict: one output dict
        """
        orig_shape = (input["height"], input["width"])
        # For some reason, resize with uint8 slightly increases box AP but decreases densepose AP
        input["image"] = input["image"].to(torch.uint8)
        augmented_inputs, tfms = self._get_augmented_inputs(input)
        # Detect boxes from all augmented versions
        with self._turn_off_roi_heads(["mask_on", "keypoint_on", "densepose_on"]):
            # temporarily disable roi heads
            all_boxes, all_scores, all_classes = self._get_augmented_boxes(augmented_inputs, tfms)
        merged_instances = self._merge_detections(all_boxes, all_scores, all_classes, orig_shape)

        if self.cfg.MODEL.MASK_ON or self.cfg.MODEL.DENSEPOSE_ON:
            # Use the detected boxes to obtain new fields
            augmented_instances = self._rescale_detected_boxes(
                augmented_inputs, merged_instances, tfms
            )
            # run forward on the detected boxes
            outputs = self._batch_inference(augmented_inputs, augmented_instances)
            # Delete now useless variables to avoid being out of memory
            del augmented_inputs, augmented_instances
            # average the predictions
            if self.cfg.MODEL.MASK_ON:
                merged_instances.pred_masks = self._reduce_pred_masks(outputs, tfms)
            if self.cfg.MODEL.DENSEPOSE_ON:
                merged_instances.pred_densepose = self._reduce_pred_densepose(outputs, tfms)
            # postprocess
            merged_instances = detector_postprocess(merged_instances, *orig_shape)
            return {"instances": merged_instances}
        else:
            return {"instances": merged_instances}

    def _reduce_pred_densepose(self, outputs, tfms):
        for output, tfm in zip(outputs, tfms):
            if any(isinstance(t, HFlipTransform) for t in tfm.transforms):
                output.pred_densepose.hflip(self._transform_data)

        # Less memory-intensive averaging
        for attr in "SIUV":
            setattr(
                outputs[0].pred_densepose,
                attr,
                sum(getattr(o.pred_densepose, attr) for o in outputs) / len(outputs),
            )
        return outputs[0].pred_densepose
