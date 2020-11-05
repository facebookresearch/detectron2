# Copyright (c) Facebook, Inc. and its affiliates.
import copy
import numpy as np
import torch
from fvcore.transforms import HFlipTransform, TransformList
from torch.nn import functional as F

from detectron2.data.transforms import RandomRotation, RotationTransform, apply_transform_gens
from detectron2.modeling.postprocessing import detector_postprocess
from detectron2.modeling.test_time_augmentation import DatasetMapperTTA, GeneralizedRCNNWithTTA

from ..converters import HFlipConverter


class DensePoseDatasetMapperTTA(DatasetMapperTTA):
    def __init__(self, cfg):
        super().__init__(cfg=cfg)
        self.angles = cfg.TEST.AUG.ROTATION_ANGLES

    def __call__(self, dataset_dict):
        ret = super().__call__(dataset_dict=dataset_dict)
        numpy_image = dataset_dict["image"].permute(1, 2, 0).numpy()
        for angle in self.angles:
            rotate = RandomRotation(angle=angle, expand=True)
            new_numpy_image, tfms = apply_transform_gens([rotate], np.copy(numpy_image))
            torch_image = torch.from_numpy(np.ascontiguousarray(new_numpy_image.transpose(2, 0, 1)))
            dic = copy.deepcopy(dataset_dict)
            # In DatasetMapperTTA, there is a pre_tfm transform (resize or no-op) that is
            # added at the beginning of each TransformList. That's '.transforms[0]'.
            dic["transforms"] = TransformList(
                [ret[-1]["transforms"].transforms[0]] + tfms.transforms
            )
            dic["image"] = torch_image
            ret.append(dic)
        return ret


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

    def _get_augmented_boxes(self, augmented_inputs, tfms):
        # Heavily based on detectron2/modeling/test_time_augmentation.py
        # Only difference is that RotationTransform is excluded from bbox computation
        # 1: forward with all augmented images
        outputs = self._batch_inference(augmented_inputs)
        # 2: union the results
        all_boxes = []
        all_scores = []
        all_classes = []
        for output, tfm in zip(outputs, tfms):
            # Need to inverse the transforms on boxes, to obtain results on original image
            if not any(isinstance(t, RotationTransform) for t in tfm.transforms):
                # Some transforms can't compute bbox correctly
                pred_boxes = output.pred_boxes.tensor
                original_pred_boxes = tfm.inverse().apply_box(pred_boxes.cpu().numpy())
                all_boxes.append(torch.from_numpy(original_pred_boxes).to(pred_boxes.device))
                all_scores.extend(output.scores)
                all_classes.extend(output.pred_classes)
        all_boxes = torch.cat(all_boxes, dim=0)
        return all_boxes, all_scores, all_classes

    def _reduce_pred_densepose(self, outputs, tfms):
        # Should apply inverse transforms on densepose preds.
        # We assume only rotation, resize & flip are used. pred_masks is a scale-invariant
        # representation, so we handle the other ones specially
        for idx, (output, tfm) in enumerate(zip(outputs, tfms)):
            for t in tfm.transforms:
                for attr in ["coarse_segm", "fine_segm", "u", "v"]:
                    setattr(
                        output.pred_densepose,
                        attr,
                        _inverse_rotation(
                            getattr(output.pred_densepose, attr), output.pred_boxes.tensor, t
                        ),
                    )
            if any(isinstance(t, HFlipTransform) for t in tfm.transforms):
                output.pred_densepose = HFlipConverter.convert(
                    output.pred_densepose, self._transform_data
                )
            self._incremental_avg_dp(outputs[0].pred_densepose, output.pred_densepose, idx)
        return outputs[0].pred_densepose

    # incrementally computed average: u_(n + 1) = u_n + (x_(n+1) - u_n) / (n + 1).
    def _incremental_avg_dp(self, avg, new_el, idx):
        for attr in ["coarse_segm", "fine_segm", "u", "v"]:
            setattr(avg, attr, (getattr(avg, attr) * idx + getattr(new_el, attr)) / (idx + 1))
            if idx:
                # Deletion of the > 0 index intermediary values to prevent GPU OOM
                setattr(new_el, attr, None)
        return avg


def _inverse_rotation(densepose_attrs, boxes, transform):
    # resample outputs to image size and rotate back the densepose preds
    # on the rotated images to the space of the original image
    if len(boxes) == 0 or not isinstance(transform, RotationTransform):
        return densepose_attrs
    boxes = boxes.int().cpu().numpy()
    wh_boxes = boxes[:, 2:] - boxes[:, :2]  # bboxes in the rotated space
    inv_boxes = rotate_box_inverse(transform, boxes).astype(int)  # bboxes in original image
    wh_diff = (inv_boxes[:, 2:] - inv_boxes[:, :2] - wh_boxes) // 2  # diff between new/old bboxes
    rotation_matrix = torch.tensor([transform.rm_image]).to(device=densepose_attrs.device).float()
    rotation_matrix[:, :, -1] = 0
    # To apply grid_sample for rotation, we need to have enough space to fit the original and
    # rotated bboxes. l_bds and r_bds are the left/right bounds that will be used to
    # crop the difference once the rotation is done
    l_bds = np.maximum(0, -wh_diff)
    for i in range(len(densepose_attrs)):
        if min(wh_boxes[i]) <= 0:
            continue
        densepose_attr = densepose_attrs[[i]].clone()
        # 1. Interpolate densepose attribute to size of the rotated bbox
        densepose_attr = F.interpolate(densepose_attr, wh_boxes[i].tolist()[::-1], mode="bilinear")
        # 2. Pad the interpolated attribute so it has room for the original + rotated bbox
        densepose_attr = F.pad(densepose_attr, tuple(np.repeat(np.maximum(0, wh_diff[i]), 2)))
        # 3. Compute rotation grid and transform
        grid = F.affine_grid(rotation_matrix, size=densepose_attr.shape)
        densepose_attr = F.grid_sample(densepose_attr, grid)
        # 4. Compute right bounds and crop the densepose_attr to the size of the original bbox
        r_bds = densepose_attr.shape[2:][::-1] - l_bds[i]
        densepose_attr = densepose_attr[:, :, l_bds[i][1] : r_bds[1], l_bds[i][0] : r_bds[0]]
        if min(densepose_attr.shape) > 0:
            # Interpolate back to the original size of the densepose attribute
            densepose_attr = F.interpolate(
                densepose_attr, densepose_attrs.shape[-2:], mode="bilinear"
            )
            # Adding a very small probability to the background class to fill padded zones
            densepose_attr[:, 0] += 1e-10
            densepose_attrs[i] = densepose_attr
    return densepose_attrs


def rotate_box_inverse(rot_tfm, rotated_box):
    """
    rotated_box is a N * 4 array of [x0, y0, x1, y1] boxes
    When a bbox is rotated, it gets bigger, because we need to surround the tilted bbox
    So when a bbox is rotated then inverse-rotated, it is much bigger than the original
    This function aims to invert the rotation on the box, but also resize it to its original size
    """
    # 1. Compute the inverse rotation of the rotated bboxes (bigger than it )
    invrot_box = rot_tfm.inverse().apply_box(rotated_box)
    h, w = rotated_box[:, 3] - rotated_box[:, 1], rotated_box[:, 2] - rotated_box[:, 0]
    ih, iw = invrot_box[:, 3] - invrot_box[:, 1], invrot_box[:, 2] - invrot_box[:, 0]
    assert 2 * rot_tfm.abs_sin ** 2 != 1, "45 degrees angle can't be inverted"
    # 2. Inverse the corresponding computation in the rotation transform
    # to get the original height/width of the rotated boxes
    orig_h = (h * rot_tfm.abs_cos - w * rot_tfm.abs_sin) / (1 - 2 * rot_tfm.abs_sin ** 2)
    orig_w = (w * rot_tfm.abs_cos - h * rot_tfm.abs_sin) / (1 - 2 * rot_tfm.abs_sin ** 2)
    # 3. Resize the inverse-rotated bboxes to their original size
    invrot_box[:, 0] += (iw - orig_w) / 2
    invrot_box[:, 1] += (ih - orig_h) / 2
    invrot_box[:, 2] -= (iw - orig_w) / 2
    invrot_box[:, 3] -= (ih - orig_h) / 2

    return invrot_box
