# Copyright (c) Facebook, Inc. and its affiliates.
import torch

from densepose.structures.data_relative import DensePoseDataRelative


class DensePoseList(object):

    _TORCH_DEVICE_CPU = torch.device("cpu")

    def __init__(self, densepose_datas, boxes_xyxy_abs, image_size_hw, device=_TORCH_DEVICE_CPU):
        assert len(densepose_datas) == len(
            boxes_xyxy_abs
        ), f"Attempt to initialize DensePoseList with {len(densepose_datas)} DensePose datas and {len(boxes_xyxy_abs)} boxes"
        self.densepose_datas = []
        for densepose_data in densepose_datas:
            assert (
                isinstance(densepose_data, DensePoseDataRelative)
                or densepose_data is None
            ), f"Attempt to initialize DensePoseList with DensePose datas of type {type(densepose_data)}, expected DensePoseDataRelative"
            densepose_data_ondevice = (
                densepose_data.to(device) if densepose_data is not None else None
            )
            self.densepose_datas.append(densepose_data_ondevice)
        self.boxes_xyxy_abs = boxes_xyxy_abs.to(device)
        self.image_size_hw = image_size_hw
        self.device = device

    def to(self, device):
        if self.device == device:
            return self
        return DensePoseList(self.densepose_datas, self.boxes_xyxy_abs, self.image_size_hw, device)

    def __iter__(self):
        return iter(self.densepose_datas)

    def __len__(self):
        return len(self.densepose_datas)

    def __repr__(self):
        s = f"{self.__class__.__name__}("
        s += f"num_instances={len(self.densepose_datas)}, "
        s += f"image_width={self.image_size_hw[1]}, "
        s += f"image_height={self.image_size_hw[0]})"
        return s

    def __getitem__(self, item):
        if isinstance(item, int):
            return self.densepose_datas[item]
        elif isinstance(item, slice):
            densepose_datas_rel = self.densepose_datas[item]
            boxes_xyxy_abs = self.boxes_xyxy_abs[item]
            return DensePoseList(
                densepose_datas_rel, boxes_xyxy_abs, self.image_size_hw, self.device
            )
        elif isinstance(item, torch.Tensor) and (item.dtype == torch.bool):
            densepose_datas_rel = [self.densepose_datas[i] for i, x in enumerate(item) if x > 0]
            boxes_xyxy_abs = self.boxes_xyxy_abs[item]
            return DensePoseList(
                densepose_datas_rel, boxes_xyxy_abs, self.image_size_hw, self.device
            )
        else:
            densepose_datas_rel = [self.densepose_datas[i] for i in item]
            boxes_xyxy_abs = self.boxes_xyxy_abs[item]
            return DensePoseList(
                densepose_datas_rel, boxes_xyxy_abs, self.image_size_hw, self.device
            )
