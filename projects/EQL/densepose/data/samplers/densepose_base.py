# Copyright (c) Facebook, Inc. and its affiliates.

from typing import Any, Dict, List, Tuple
import torch
from torch.nn import functional as F

from detectron2.structures import BoxMode, Instances

from densepose.converters import ToChartResultConverter
from densepose.converters.base import IntTupleBox, make_int_box
from densepose.structures import DensePoseDataRelative, DensePoseList


class DensePoseBaseSampler:
    """
    Base DensePose sampler to produce DensePose data from DensePose predictions.
    Samples for each class are drawn according to some distribution over all pixels estimated
    to belong to that class.
    """

    def __init__(self, count_per_class: int = 8):
        """
        Constructor

        Args:
          count_per_class (int): the sampler produces at most `count_per_class`
              samples for each category
        """
        self.count_per_class = count_per_class

    def __call__(self, instances: Instances) -> DensePoseList:
        """
        Convert DensePose predictions (an instance of `DensePoseChartPredictorOutput`)
        into DensePose annotations data (an instance of `DensePoseList`)
        """
        boxes_xyxy_abs = instances.pred_boxes.tensor.clone().cpu()
        boxes_xywh_abs = BoxMode.convert(boxes_xyxy_abs, BoxMode.XYXY_ABS, BoxMode.XYWH_ABS)
        dp_datas = []
        for i in range(len(boxes_xywh_abs)):
            annotation_i = self._sample(instances[i], make_int_box(boxes_xywh_abs[i]))
            annotation_i[DensePoseDataRelative.S_KEY] = self._resample_mask(  # pyre-ignore[6]
                instances[i].pred_densepose
            )
            dp_datas.append(DensePoseDataRelative(annotation_i))
        # create densepose annotations on CPU
        dp_list = DensePoseList(dp_datas, boxes_xyxy_abs, instances.image_size)
        return dp_list

    def _sample(self, instance: Instances, bbox_xywh: IntTupleBox) -> Dict[str, List[Any]]:
        """
        Sample DensPoseDataRelative from estimation results
        """
        labels, dp_result = self._produce_labels_and_results(instance)
        annotation = {
            DensePoseDataRelative.X_KEY: [],
            DensePoseDataRelative.Y_KEY: [],
            DensePoseDataRelative.U_KEY: [],
            DensePoseDataRelative.V_KEY: [],
            DensePoseDataRelative.I_KEY: [],
        }
        x0, y0, _, _ = bbox_xywh
        n, h, w = dp_result.shape
        for part_id in range(1, DensePoseDataRelative.N_PART_LABELS + 1):
            # indices - tuple of 3 1D tensors of size k
            # 0: index along the first dimension N
            # 1: index along H dimension
            # 2: index along W dimension
            indices = torch.nonzero(labels.expand(n, h, w) == part_id, as_tuple=True)
            # values - an array of size [n, k]
            # n: number of channels (U, V, confidences)
            # k: number of points labeled with part_id
            values = dp_result[indices].view(n, -1)
            k = values.shape[1]
            count = min(self.count_per_class, k)
            if count <= 0:
                continue
            index_sample = self._produce_index_sample(values, count)
            sampled_values = values[:, index_sample]
            sampled_y = indices[1][index_sample] + 0.5
            sampled_x = indices[2][index_sample] + 0.5
            # prepare / normalize data
            x = (sampled_x / w * 256.0).cpu().tolist()
            y = (sampled_y / h * 256.0).cpu().tolist()
            u = sampled_values[0].clamp(0, 1).cpu().tolist()
            v = sampled_values[1].clamp(0, 1).cpu().tolist()
            fine_segm_labels = [part_id] * count
            # extend annotations
            annotation[DensePoseDataRelative.X_KEY].extend(x)
            annotation[DensePoseDataRelative.Y_KEY].extend(y)
            annotation[DensePoseDataRelative.U_KEY].extend(u)
            annotation[DensePoseDataRelative.V_KEY].extend(v)
            annotation[DensePoseDataRelative.I_KEY].extend(fine_segm_labels)
        return annotation

    def _produce_index_sample(self, values: torch.Tensor, count: int):
        """
        Abstract method to produce a sample of indices to select data
        To be implemented in descendants

        Args:
            values (torch.Tensor): an array of size [n, k] that contains
                estimated values (U, V, confidences);
                n: number of channels (U, V, confidences)
                k: number of points labeled with part_id
            count (int): number of samples to produce, should be positive and <= k

        Return:
            list(int): indices of values (along axis 1) selected as a sample
        """
        raise NotImplementedError

    def _produce_labels_and_results(self, instance: Instances) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Method to get labels and DensePose results from an instance

        Args:
            instance (Instances): an instance of `DensePoseChartPredictorOutput`

        Return:
            labels (torch.Tensor): shape [H, W], DensePose segmentation labels
            dp_result (torch.Tensor): shape [2, H, W], stacked DensePose results u and v
        """
        converter = ToChartResultConverter
        chart_result = converter.convert(instance.pred_densepose, instance.pred_boxes)
        labels, dp_result = chart_result.labels.cpu(), chart_result.uv.cpu()
        return labels, dp_result

    def _resample_mask(self, output: Any) -> torch.Tensor:
        """
        Convert DensePose predictor output to segmentation annotation - tensors of size
        (256, 256) and type `int64`.

        Args:
            output: DensePose predictor output with the following attributes:
             - coarse_segm: tensor of size [N, D, H, W] with unnormalized coarse
               segmentation scores
             - fine_segm: tensor of size [N, C, H, W] with unnormalized fine
               segmentation scores
        Return:
            Tensor of size (S, S) and type `int64` with coarse segmentation annotations,
            where S = DensePoseDataRelative.MASK_SIZE
        """
        sz = DensePoseDataRelative.MASK_SIZE
        S = (
            F.interpolate(output.coarse_segm, (sz, sz), mode="bilinear", align_corners=False)
            .argmax(dim=1)
            .long()
        )
        I = (
            (
                F.interpolate(
                    output.fine_segm, (sz, sz), mode="bilinear", align_corners=False
                ).argmax(dim=1)
                * (S > 0).long()
            )
            .squeeze()
            .cpu()
        )
        # Map fine segmentation results to coarse segmentation ground truth
        # TODO: extract this into separate classes
        # coarse segmentation: 1 = Torso, 2 = Right Hand, 3 = Left Hand,
        # 4 = Left Foot, 5 = Right Foot, 6 = Upper Leg Right, 7 = Upper Leg Left,
        # 8 = Lower Leg Right, 9 = Lower Leg Left, 10 = Upper Arm Left,
        # 11 = Upper Arm Right, 12 = Lower Arm Left, 13 = Lower Arm Right,
        # 14 = Head
        # fine segmentation: 1, 2 = Torso, 3 = Right Hand, 4 = Left Hand,
        # 5 = Left Foot, 6 = Right Foot, 7, 9 = Upper Leg Right,
        # 8, 10 = Upper Leg Left, 11, 13 = Lower Leg Right,
        # 12, 14 = Lower Leg Left, 15, 17 = Upper Arm Left,
        # 16, 18 = Upper Arm Right, 19, 21 = Lower Arm Left,
        # 20, 22 = Lower Arm Right, 23, 24 = Head
        FINE_TO_COARSE_SEGMENTATION = {
            1: 1,
            2: 1,
            3: 2,
            4: 3,
            5: 4,
            6: 5,
            7: 6,
            8: 7,
            9: 6,
            10: 7,
            11: 8,
            12: 9,
            13: 8,
            14: 9,
            15: 10,
            16: 11,
            17: 10,
            18: 11,
            19: 12,
            20: 13,
            21: 12,
            22: 13,
            23: 14,
            24: 14,
        }
        mask = torch.zeros((sz, sz), dtype=torch.int64, device=torch.device("cpu"))
        for i in range(DensePoseDataRelative.N_PART_LABELS):
            mask[I == i + 1] = FINE_TO_COARSE_SEGMENTATION[i + 1]
        return mask
