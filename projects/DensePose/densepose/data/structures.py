# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import base64
import numpy as np
from io import BytesIO
from typing import BinaryIO, Dict, Union
import torch
from PIL import Image
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
        import pycocotools.mask as mask_util

        poly_specs = annotation[DensePoseDataRelative.S_KEY]
        segm = torch.zeros((DensePoseDataRelative.MASK_SIZE,) * 2, dtype=torch.float32)
        for i in range(DensePoseDataRelative.N_BODY_PARTS):
            poly_i = poly_specs[i]
            if poly_i:
                mask_i = mask_util.decode(poly_i)
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


class DensePoseOutput(object):
    def __init__(self, S, I, U, V, confidences):
        """
        Args:
            S (`torch.Tensor`): coarse segmentation tensor of size (N, A, H, W)
            I (`torch.Tensor`): fine segmentation tensor of size (N, C, H, W)
            U (`torch.Tensor`): U coordinates for each fine segmentation label of size (N, C, H, W)
            V (`torch.Tensor`): V coordinates for each fine segmentation label of size (N, C, H, W)
            confidences (dict of str -> `torch.Tensor`) estimated confidence model parameters
        """
        self.S = S
        self.I = I  # noqa: E741
        self.U = U
        self.V = V
        self.confidences = confidences
        self._check_output_dims(S, I, U, V)

    def _check_output_dims(self, S, I, U, V):
        assert (
            len(S.size()) == 4
        ), "Segmentation output should have 4 " "dimensions (NCHW), but has size {}".format(
            S.size()
        )
        assert (
            len(I.size()) == 4
        ), "Segmentation output should have 4 " "dimensions (NCHW), but has size {}".format(
            S.size()
        )
        assert (
            len(U.size()) == 4
        ), "Segmentation output should have 4 " "dimensions (NCHW), but has size {}".format(
            S.size()
        )
        assert (
            len(V.size()) == 4
        ), "Segmentation output should have 4 " "dimensions (NCHW), but has size {}".format(
            S.size()
        )
        assert len(S) == len(I), (
            "Number of output segmentation planes {} "
            "should be equal to the number of output part index "
            "planes {}".format(len(S), len(I))
        )
        assert S.size()[2:] == I.size()[2:], (
            "Output segmentation plane size {} "
            "should be equal to the output part index "
            "plane size {}".format(S.size()[2:], I.size()[2:])
        )
        assert I.size() == U.size(), (
            "Part index output shape {} "
            "should be the same as U coordinates output shape {}".format(I.size(), U.size())
        )
        assert I.size() == V.size(), (
            "Part index output shape {} "
            "should be the same as V coordinates output shape {}".format(I.size(), V.size())
        )

    def resize(self, image_size_hw):
        # do nothing - outputs are invariant to resize
        pass

    def _crop(self, S, I, U, V, bbox_old_xywh, bbox_new_xywh):
        """
        Resample S, I, U, V from bbox_old to the cropped bbox_new
        """
        x0old, y0old, wold, hold = bbox_old_xywh
        x0new, y0new, wnew, hnew = bbox_new_xywh
        tr_coords = normalized_coords_transform(x0old, y0old, wold, hold)
        topleft = (x0new, y0new)
        bottomright = (x0new + wnew, y0new + hnew)
        topleft_norm = tr_coords(topleft)
        bottomright_norm = tr_coords(bottomright)
        hsize = S.size(1)
        wsize = S.size(2)
        grid = torch.meshgrid(
            torch.arange(
                topleft_norm[1],
                bottomright_norm[1],
                (bottomright_norm[1] - topleft_norm[1]) / hsize,
            )[:hsize],
            torch.arange(
                topleft_norm[0],
                bottomright_norm[0],
                (bottomright_norm[0] - topleft_norm[0]) / wsize,
            )[:wsize],
        )
        grid = torch.stack(grid, dim=2).to(S.device)
        assert (
            grid.size(0) == hsize
        ), "Resampled grid expected " "height={}, actual height={}".format(hsize, grid.size(0))
        assert grid.size(1) == wsize, "Resampled grid expected " "width={}, actual width={}".format(
            wsize, grid.size(1)
        )
        S_new = F.grid_sample(
            S.unsqueeze(0),
            torch.unsqueeze(grid, 0),
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        ).squeeze(0)
        I_new = F.grid_sample(
            I.unsqueeze(0),
            torch.unsqueeze(grid, 0),
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        ).squeeze(0)
        U_new = F.grid_sample(
            U.unsqueeze(0),
            torch.unsqueeze(grid, 0),
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        ).squeeze(0)
        V_new = F.grid_sample(
            V.unsqueeze(0),
            torch.unsqueeze(grid, 0),
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        ).squeeze(0)
        return S_new, I_new, U_new, V_new

    def crop(self, indices_cropped, bboxes_old, bboxes_new):
        """
        Crop outputs for selected bounding boxes to the new bounding boxes.
        """
        # VK: cropping is ignored for now
        # for i, ic in enumerate(indices_cropped):
        #    self.S[ic], self.I[ic], self.U[ic], self.V[ic] = \
        #        self._crop(self.S[ic], self.I[ic], self.U[ic], self.V[ic],
        #        bboxes_old[i], bboxes_new[i])
        pass

    def hflip(self, transform_data: DensePoseTransformData) -> None:
        """
        Change S, I, U and V to take into account a Horizontal flip.
        """
        if self.I.shape[0] > 0:
            for el in "SIUV":
                self.__dict__[el] = torch.flip(self.__dict__[el], [3])
            self._flip_iuv_semantics_tensor(transform_data)
            self._flip_segm_semantics_tensor(transform_data)

    def _flip_iuv_semantics_tensor(self, dp_transform_data: DensePoseTransformData) -> None:
        point_label_symmetries = dp_transform_data.point_label_symmetries
        uv_symmetries = dp_transform_data.uv_symmetries

        N, C, H, W = self.U.shape
        u_loc = (self.U[:, 1:, :, :].clamp(0, 1) * 255).long()
        v_loc = (self.V[:, 1:, :, :].clamp(0, 1) * 255).long()
        Iindex = torch.arange(C - 1, device=self.U.device)[None, :, None, None].expand(
            N, C - 1, H, W
        )
        self.U[:, 1:, :, :] = uv_symmetries["U_transforms"][Iindex, v_loc, u_loc]
        self.V[:, 1:, :, :] = uv_symmetries["V_transforms"][Iindex, v_loc, u_loc]

        for el in "IUV":
            self.__dict__[el] = self.__dict__[el][:, point_label_symmetries, :, :]

    def _flip_segm_semantics_tensor(self, dp_transform_data):
        if self.S.shape[1] == DensePoseDataRelative.N_BODY_PARTS + 1:
            self.S = self.S[:, dp_transform_data.mask_label_symmetries, :, :]

    def to_result(self, boxes_xywh):
        """
        Convert DensePose outputs to results format. Results are more compact,
        but cannot be resampled any more
        """
        result = DensePoseResult(boxes_xywh, self.S, self.I, self.U, self.V)
        return result

    def __getitem__(self, item):
        if isinstance(item, int):
            S_selected = self.S[item].unsqueeze(0)
            I_selected = self.I[item].unsqueeze(0)
            U_selected = self.U[item].unsqueeze(0)
            V_selected = self.V[item].unsqueeze(0)
            conf_selected = {}
            for key in self.confidences:
                conf_selected[key] = self.confidences[key][item].unsqueeze(0)
        else:
            S_selected = self.S[item]
            I_selected = self.I[item]
            U_selected = self.U[item]
            V_selected = self.V[item]
            conf_selected = {}
            for key in self.confidences:
                conf_selected[key] = self.confidences[key][item]
        return DensePoseOutput(S_selected, I_selected, U_selected, V_selected, conf_selected)

    def __str__(self):
        s = "DensePoseOutput S {}, I {}, U {}, V {}".format(
            list(self.S.size()), list(self.I.size()), list(self.U.size()), list(self.V.size())
        )
        s_conf = "confidences: [{}]".format(
            ", ".join([f"{key} {list(self.confidences[key].size())}" for key in self.confidences])
        )
        return ", ".join([s, s_conf])

    def __len__(self):
        return self.S.size(0)


class DensePoseResult(object):
    def __init__(self, boxes_xywh, S, I, U, V):
        self.results = []
        self.boxes_xywh = boxes_xywh.cpu().tolist()
        assert len(boxes_xywh.size()) == 2
        assert boxes_xywh.size(1) == 4
        for i, box_xywh in enumerate(boxes_xywh):
            result_i = self._output_to_result(box_xywh, S[[i]], I[[i]], U[[i]], V[[i]])
            result_numpy_i = result_i.cpu().numpy()
            result_encoded_i = DensePoseResult.encode_png_data(result_numpy_i)
            result_encoded_with_shape_i = (result_numpy_i.shape, result_encoded_i)
            self.results.append(result_encoded_with_shape_i)

    def __str__(self):
        s = "DensePoseResult: N={} [{}]".format(
            len(self.results), ", ".join([str(list(r[0])) for r in self.results])
        )
        return s

    def _output_to_result(self, box_xywh, S, I, U, V):
        x, y, w, h = box_xywh
        w = max(int(w), 1)
        h = max(int(h), 1)
        result = torch.zeros([3, h, w], dtype=torch.uint8, device=U.device)
        assert (
            len(S.size()) == 4
        ), "AnnIndex tensor size should have {} " "dimensions but has {}".format(4, len(S.size()))
        s_bbox = F.interpolate(S, (h, w), mode="bilinear", align_corners=False).argmax(dim=1)
        assert (
            len(I.size()) == 4
        ), "IndexUV tensor size should have {} " "dimensions but has {}".format(4, len(S.size()))
        i_bbox = (
            F.interpolate(I, (h, w), mode="bilinear", align_corners=False).argmax(dim=1)
            * (s_bbox > 0).long()
        ).squeeze(0)
        assert len(U.size()) == 4, "U tensor size should have {} " "dimensions but has {}".format(
            4, len(U.size())
        )
        u_bbox = F.interpolate(U, (h, w), mode="bilinear", align_corners=False)
        assert len(V.size()) == 4, "V tensor size should have {} " "dimensions but has {}".format(
            4, len(V.size())
        )
        v_bbox = F.interpolate(V, (h, w), mode="bilinear", align_corners=False)
        result[0] = i_bbox
        for part_id in range(1, u_bbox.size(1)):
            result[1][i_bbox == part_id] = (
                (u_bbox[0, part_id][i_bbox == part_id] * 255).clamp(0, 255).to(torch.uint8)
            )
            result[2][i_bbox == part_id] = (
                (v_bbox[0, part_id][i_bbox == part_id] * 255).clamp(0, 255).to(torch.uint8)
            )
        assert (
            result.size(1) == h
        ), "Results height {} should be equal" "to bounding box height {}".format(result.size(1), h)
        assert (
            result.size(2) == w
        ), "Results width {} should be equal" "to bounding box width {}".format(result.size(2), w)
        return result

    @staticmethod
    def encode_png_data(arr):
        """
        Encode array data as a PNG image using the highest compression rate
        @param arr [in] Data stored in an array of size (3, M, N) of type uint8
        @return Base64-encoded string containing PNG-compressed data
        """
        assert len(arr.shape) == 3, "Expected a 3D array as an input," " got a {0}D array".format(
            len(arr.shape)
        )
        assert arr.shape[0] == 3, "Expected first array dimension of size 3," " got {0}".format(
            arr.shape[0]
        )
        assert arr.dtype == np.uint8, "Expected an array of type np.uint8, " " got {0}".format(
            arr.dtype
        )
        data = np.moveaxis(arr, 0, -1)
        im = Image.fromarray(data)
        fstream = BytesIO()
        im.save(fstream, format="png", optimize=True)
        s = base64.encodebytes(fstream.getvalue()).decode()
        return s

    @staticmethod
    def decode_png_data(shape, s):
        """
        Decode array data from a string that contains PNG-compressed data
        @param Base64-encoded string containing PNG-compressed data
        @return Data stored in an array of size (3, M, N) of type uint8
        """
        fstream = BytesIO(base64.decodebytes(s.encode()))
        im = Image.open(fstream)
        data = np.moveaxis(np.array(im.getdata(), dtype=np.uint8), -1, 0)
        return data.reshape(shape)

    def __len__(self):
        return len(self.results)

    def __getitem__(self, item):
        result_encoded = self.results[item]
        bbox_xywh = self.boxes_xywh[item]
        return result_encoded, bbox_xywh


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
