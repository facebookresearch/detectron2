# Copyright (c) Facebook, Inc. and its affiliates.
import numpy as np
from typing import BinaryIO, Dict, Union
import torch
from torch.nn import functional as F


class DensePoseTransformData(object):

    # Horizontal symmetry label transforms used for horizontal flip
    MASK_LABEL_SYMMETRIES = [0, 1, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 14]
    # fmt: off
    POINT_LABEL_SYMMETRIES = [ 0, 1, 2, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15, 18, 17, 20, 19, 22, 21, 24, 23]  # noqa
    # fmt: on

    def __init__(self, uv_symmetries: Dict[str, torch.Tensor], device: torch.device):
        self.mask_label_symmetries = DensePoseTransformData.MASK_LABEL_SYMMETRIES
        self.point_label_symmetries = DensePoseTransformData.POINT_LABEL_SYMMETRIES
        self.uv_symmetries = uv_symmetries
        self.device = torch.device("cpu")

    def to(self, device: torch.device, copy: bool = False) -> "DensePoseTransformData":
        """
        Convert transform data to the specified device

        Args:
            device (torch.device): device to convert the data to
            copy (bool): flag that specifies whether to copy or to reference the data
                in case the device is the same
        Return:
            An instance of `DensePoseTransformData` with data stored on the specified device
        """
        if self.device == device and not copy:
            return self
        uv_symmetry_map = {}
        for key in self.uv_symmetries:
            uv_symmetry_map[key] = self.uv_symmetries[key].to(device=device, copy=copy)
        return DensePoseTransformData(uv_symmetry_map, device)

    @staticmethod
    def load(io: Union[str, BinaryIO]):
        """
        Args:
            io: (str or binary file-like object): input file to load data from
        Returns:
            An instance of `DensePoseTransformData` with transforms loaded from the file
        """
        import scipy.io

        uv_symmetry_map = scipy.io.loadmat(io)
        uv_symmetry_map_torch = {}
        for key in ["U_transforms", "V_transforms"]:
            uv_symmetry_map_torch[key] = []
            map_src = uv_symmetry_map[key]
            map_dst = uv_symmetry_map_torch[key]
            for i in range(map_src.shape[1]):
                map_dst.append(torch.from_numpy(map_src[0, i]).to(dtype=torch.float))
            uv_symmetry_map_torch[key] = torch.stack(map_dst, dim=0)
        transform_data = DensePoseTransformData(uv_symmetry_map_torch, device=torch.device("cpu"))
        return transform_data


class DensePoseDataRelative(object):
    """
    Dense pose relative annotations that can be applied to any bounding box:
        x - normalized X coordinates [0, 255] of annotated points
        y - normalized Y coordinates [0, 255] of annotated points
        i - body part labels 0,...,24 for annotated points
        u - body part U coordinates [0, 1] for annotated points
        v - body part V coordinates [0, 1] for annotated points
        segm - 256x256 segmentation mask with values 0,...,14
    To obtain absolute x and y data wrt some bounding box one needs to first
    divide the data by 256, multiply by the respective bounding box size
    and add bounding box offset:
        x_img = x0 + x_norm * w / 256.0
        y_img = y0 + y_norm * h / 256.0
    Segmentation masks are typically sampled to get image-based masks.
    """

    # Key for normalized X coordinates in annotation dict
    X_KEY = "dp_x"
    # Key for normalized Y coordinates in annotation dict
    Y_KEY = "dp_y"
    # Key for U part coordinates in annotation dict
    U_KEY = "dp_U"
    # Key for V part coordinates in annotation dict
    V_KEY = "dp_V"
    # Key for I point labels in annotation dict
    I_KEY = "dp_I"
    # Key for segmentation mask in annotation dict
    S_KEY = "dp_masks"
    # Number of body parts in segmentation masks
    N_BODY_PARTS = 14
    # Number of parts in point labels
    N_PART_LABELS = 24
    MASK_SIZE = 256

    def __init__(self, annotation, cleanup=False):
        is_valid, reason_not_valid = DensePoseDataRelative.validate_annotation(annotation)
        assert is_valid, "Invalid DensePose annotations: {}".format(reason_not_valid)
        self.x = torch.as_tensor(annotation[DensePoseDataRelative.X_KEY])
        self.y = torch.as_tensor(annotation[DensePoseDataRelative.Y_KEY])
        self.i = torch.as_tensor(annotation[DensePoseDataRelative.I_KEY])
        self.u = torch.as_tensor(annotation[DensePoseDataRelative.U_KEY])
        self.v = torch.as_tensor(annotation[DensePoseDataRelative.V_KEY])
        self.segm = DensePoseDataRelative.extract_segmentation_mask(annotation)
        self.device = torch.device("cpu")
        if cleanup:
            DensePoseDataRelative.cleanup_annotation(annotation)

    def to(self, device):
        if self.device == device:
            return self
        new_data = DensePoseDataRelative.__new__(DensePoseDataRelative)
        new_data.x = self.x
        new_data.x = self.x.to(device)
        new_data.y = self.y.to(device)
        new_data.i = self.i.to(device)
        new_data.u = self.u.to(device)
        new_data.v = self.v.to(device)
        new_data.segm = self.segm.to(device)
        new_data.device = device
        return new_data

    @staticmethod
    def extract_segmentation_mask(annotation):
        poly_specs = annotation[DensePoseDataRelative.S_KEY]
        if isinstance(poly_specs, torch.Tensor):
            # data is already given as mask tensors, no need to decode
            return poly_specs

        import pycocotools.mask as mask_utils

        segm = torch.zeros((DensePoseDataRelative.MASK_SIZE,) * 2, dtype=torch.float32)
        for i in range(DensePoseDataRelative.N_BODY_PARTS):
            poly_i = poly_specs[i]
            if poly_i:
                mask_i = mask_utils.decode(poly_i)
                segm[mask_i > 0] = i + 1
        return segm

    @staticmethod
    def validate_annotation(annotation):
        for key in [
            DensePoseDataRelative.X_KEY,
            DensePoseDataRelative.Y_KEY,
            DensePoseDataRelative.I_KEY,
            DensePoseDataRelative.U_KEY,
            DensePoseDataRelative.V_KEY,
            DensePoseDataRelative.S_KEY,
        ]:
            if key not in annotation:
                return False, "no {key} data in the annotation".format(key=key)
        return True, None

    @staticmethod
    def cleanup_annotation(annotation):
        for key in [
            DensePoseDataRelative.X_KEY,
            DensePoseDataRelative.Y_KEY,
            DensePoseDataRelative.I_KEY,
            DensePoseDataRelative.U_KEY,
            DensePoseDataRelative.V_KEY,
            DensePoseDataRelative.S_KEY,
        ]:
            if key in annotation:
                del annotation[key]

    def apply_transform(self, transforms, densepose_transform_data):
        self._transform_pts(transforms, densepose_transform_data)
        self._transform_segm(transforms, densepose_transform_data)

    def _transform_pts(self, transforms, dp_transform_data):
        import detectron2.data.transforms as T

        # NOTE: This assumes that HorizFlipTransform is the only one that does flip
        do_hflip = sum(isinstance(t, T.HFlipTransform) for t in transforms.transforms) % 2 == 1
        if do_hflip:
            self.x = self.segm.size(1) - self.x
            self._flip_iuv_semantics(dp_transform_data)

        for t in transforms.transforms:
            if isinstance(t, T.RotationTransform):
                xy_scale = np.array((t.w, t.h)) / DensePoseDataRelative.MASK_SIZE
                xy = t.apply_coords(np.stack((self.x, self.y), axis=1) * xy_scale)
                self.x, self.y = torch.tensor(xy / xy_scale, dtype=self.x.dtype).T

    def _flip_iuv_semantics(self, dp_transform_data: DensePoseTransformData) -> None:
        i_old = self.i.clone()
        uv_symmetries = dp_transform_data.uv_symmetries
        pt_label_symmetries = dp_transform_data.point_label_symmetries
        for i in range(self.N_PART_LABELS):
            if i + 1 in i_old:
                annot_indices_i = i_old == i + 1
                if pt_label_symmetries[i + 1] != i + 1:
                    self.i[annot_indices_i] = pt_label_symmetries[i + 1]
                u_loc = (self.u[annot_indices_i] * 255).long()
                v_loc = (self.v[annot_indices_i] * 255).long()
                self.u[annot_indices_i] = uv_symmetries["U_transforms"][i][v_loc, u_loc].to(
                    device=self.u.device
                )
                self.v[annot_indices_i] = uv_symmetries["V_transforms"][i][v_loc, u_loc].to(
                    device=self.v.device
                )

    def _transform_segm(self, transforms, dp_transform_data):
        import detectron2.data.transforms as T

        # NOTE: This assumes that HorizFlipTransform is the only one that does flip
        do_hflip = sum(isinstance(t, T.HFlipTransform) for t in transforms.transforms) % 2 == 1
        if do_hflip:
            self.segm = torch.flip(self.segm, [1])
            self._flip_segm_semantics(dp_transform_data)

        for t in transforms.transforms:
            if isinstance(t, T.RotationTransform):
                self._transform_segm_rotation(t)

    def _flip_segm_semantics(self, dp_transform_data):
        old_segm = self.segm.clone()
        mask_label_symmetries = dp_transform_data.mask_label_symmetries
        for i in range(self.N_BODY_PARTS):
            if mask_label_symmetries[i + 1] != i + 1:
                self.segm[old_segm == i + 1] = mask_label_symmetries[i + 1]

    def _transform_segm_rotation(self, rotation):
        self.segm = F.interpolate(self.segm[None, None, :], (rotation.h, rotation.w)).numpy()
        self.segm = torch.tensor(rotation.apply_segmentation(self.segm[0, 0]))[None, None, :]
        self.segm = F.interpolate(self.segm, [DensePoseDataRelative.MASK_SIZE] * 2)[0, 0]


def normalized_coords_transform(x0, y0, w, h):
    """
    Coordinates transform that maps top left corner to (-1, -1) and bottom
    right corner to (1, 1). Used for torch.grid_sample to initialize the
    grid
    """

    def f(p):
        return (2 * (p[0] - x0) / w - 1, 2 * (p[1] - y0) / h - 1)

    return f


class DensePoseList(object):

    _TORCH_DEVICE_CPU = torch.device("cpu")

    def __init__(self, densepose_datas, boxes_xyxy_abs, image_size_hw, device=_TORCH_DEVICE_CPU):
        assert len(densepose_datas) == len(
            boxes_xyxy_abs
        ), "Attempt to initialize DensePoseList with {} DensePose datas " "and {} boxes".format(
            len(densepose_datas), len(boxes_xyxy_abs)
        )
        self.densepose_datas = []
        for densepose_data in densepose_datas:
            assert isinstance(densepose_data, DensePoseDataRelative) or densepose_data is None, (
                "Attempt to initialize DensePoseList with DensePose datas "
                "of type {}, expected DensePoseDataRelative".format(type(densepose_data))
            )
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
        s = self.__class__.__name__ + "("
        s += "num_instances={}, ".format(len(self.densepose_datas))
        s += "image_width={}, ".format(self.image_size_hw[1])
        s += "image_height={})".format(self.image_size_hw[0])
        return s

    def __getitem__(self, item):
        if isinstance(item, int):
            densepose_data_rel = self.densepose_datas[item]
            return densepose_data_rel
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
