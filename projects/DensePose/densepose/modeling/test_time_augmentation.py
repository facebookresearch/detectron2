# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
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
        self._transform_data = transform_data
        super().__init__(cfg=cfg, model=model, tta_mapper=tta_mapper, batch_size=batch_size)

    # the implementation follows closely the one from detectron2/modeling
    def _inference_one_image(self, input):
        """
        Args:
            input (dict): one dataset dict

        Returns:
            dict: one output dict
        """

        augmented_inputs, aug_vars = self._get_augmented_inputs(input)
        # Detect boxes from all augmented versions
        with self._turn_off_roi_heads(["mask_on", "keypoint_on", "densepose_on"]):
            # temporarily disable roi heads
            all_boxes, all_scores, all_classes = self._get_augmented_boxes(
                augmented_inputs, aug_vars
            )
        merged_instances = self._merge_detections(
            all_boxes, all_scores, all_classes, (aug_vars["height"], aug_vars["width"])
        )

        if self.cfg.MODEL.MASK_ON or self.cfg.MODEL.DENSEPOSE_ON:
            # Use the detected boxes to obtain new fields
            augmented_instances = self._rescale_detected_boxes(
                augmented_inputs, merged_instances, aug_vars
            )
            # run forward on the detected boxes
            outputs = self._batch_inference(
                augmented_inputs, augmented_instances, do_postprocess=False
            )
            # Delete now useless variables to avoid being out of memory
            del augmented_inputs, augmented_instances, merged_instances
            # average the predictions
            if self.cfg.MODEL.MASK_ON:
                outputs[0].pred_masks = self._reduce_pred_masks(outputs, aug_vars)
            if self.cfg.MODEL.DENSEPOSE_ON:
                outputs[0].pred_densepose = self._reduce_pred_densepose(outputs, aug_vars)
            # postprocess
            output = self._detector_postprocess(outputs[0], aug_vars)
            return {"instances": output}
        else:
            return {"instances": merged_instances}

    def _reduce_pred_densepose(self, outputs, aug_vars):
        for idx, output in enumerate(outputs):
            if aug_vars["do_hflip"][idx]:
                output.pred_densepose.hflip(self._transform_data)
        # Less memory-intensive averaging
        for attr in "SIUV":
            setattr(
                outputs[0].pred_densepose,
                attr,
                sum(getattr(o.pred_densepose, attr) for o in outputs) / len(outputs),
            )
        return outputs[0].pred_densepose
