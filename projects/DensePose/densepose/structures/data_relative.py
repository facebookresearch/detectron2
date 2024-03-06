# Copyright (c) Facebook, Inc. and its affiliates.

# pyre-unsafe
import numpy as np
import torch
from torch.nn import functional as F

from densepose.data.meshes.catalog import MeshCatalog
from densepose.structures.mesh import load_mesh_symmetry
from densepose.structures.transform_data import DensePoseTransformData


class DensePoseDataRelative:
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
    # Key for U part coordinates in annotation dict (used in chart-based annotations)
    U_KEY = "dp_U"
    # Key for V part coordinates in annotation dict (used in chart-based annotations)
    V_KEY = "dp_V"
    # Key for I point labels in annotation dict (used in chart-based annotations)
    I_KEY = "dp_I"
    # Key for segmentation mask in annotation dict
    S_KEY = "dp_masks"
    # Key for vertex ids (used in continuous surface embeddings annotations)
    VERTEX_IDS_KEY = "dp_vertex"
    # Key for mesh id (used in continuous surface embeddings annotations)
    MESH_NAME_KEY = "ref_model"
    # Number of body parts in segmentation masks
    N_BODY_PARTS = 14
    # Number of parts in point labels
    N_PART_LABELS = 24
    MASK_SIZE = 256

    def __init__(self, annotation, cleanup=False):
        self.x = torch.as_tensor(annotation[DensePoseDataRelative.X_KEY])
        self.y = torch.as_tensor(annotation[DensePoseDataRelative.Y_KEY])
        if (
            DensePoseDataRelative.I_KEY in annotation
            and DensePoseDataRelative.U_KEY in annotation
            and DensePoseDataRelative.V_KEY in annotation
        ):
            self.i = torch.as_tensor(annotation[DensePoseDataRelative.I_KEY])
            self.u = torch.as_tensor(annotation[DensePoseDataRelative.U_KEY])
            self.v = torch.as_tensor(annotation[DensePoseDataRelative.V_KEY])
        if (
            DensePoseDataRelative.VERTEX_IDS_KEY in annotation
            and DensePoseDataRelative.MESH_NAME_KEY in annotation
        ):
            self.vertex_ids = torch.as_tensor(
                annotation[DensePoseDataRelative.VERTEX_IDS_KEY], dtype=torch.long
            )
            self.mesh_id = MeshCatalog.get_mesh_id(annotation[DensePoseDataRelative.MESH_NAME_KEY])
        if DensePoseDataRelative.S_KEY in annotation:
            self.segm = DensePoseDataRelative.extract_segmentation_mask(annotation)
        self.device = torch.device("cpu")
        if cleanup:
            DensePoseDataRelative.cleanup_annotation(annotation)

    def to(self, device):
        if self.device == device:
            return self
        new_data = DensePoseDataRelative.__new__(DensePoseDataRelative)
        new_data.x = self.x.to(device)
        new_data.y = self.y.to(device)
        for attr in ["i", "u", "v", "vertex_ids", "segm"]:
            if hasattr(self, attr):
                setattr(new_data, attr, getattr(self, attr).to(device))
        if hasattr(self, "mesh_id"):
            new_data.mesh_id = self.mesh_id
        new_data.device = device
        return new_data

    @staticmethod
    def extract_segmentation_mask(annotation):
        import pycocotools.mask as mask_utils

        # TODO: annotation instance is accepted if it contains either
        # DensePose segmentation or instance segmentation. However, here we
        # only rely on DensePose segmentation
        poly_specs = annotation[DensePoseDataRelative.S_KEY]
        if isinstance(poly_specs, torch.Tensor):
            # data is already given as mask tensors, no need to decode
            return poly_specs
        segm = torch.zeros((DensePoseDataRelative.MASK_SIZE,) * 2, dtype=torch.float32)
        if isinstance(poly_specs, dict):
            if poly_specs:
                mask = mask_utils.decode(poly_specs)
                segm[mask > 0] = 1
        else:
            for i in range(len(poly_specs)):
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
        ]:
            if key not in annotation:
                return False, "no {key} data in the annotation".format(key=key)
        valid_for_iuv_setting = all(
            key in annotation
            for key in [
                DensePoseDataRelative.I_KEY,
                DensePoseDataRelative.U_KEY,
                DensePoseDataRelative.V_KEY,
            ]
        )
        valid_for_cse_setting = all(
            key in annotation
            for key in [
                DensePoseDataRelative.VERTEX_IDS_KEY,
                DensePoseDataRelative.MESH_NAME_KEY,
            ]
        )
        if not valid_for_iuv_setting and not valid_for_cse_setting:
            return (
                False,
                "expected either {} (IUV setting) or {} (CSE setting) annotations".format(
                    ", ".join(
                        [
                            DensePoseDataRelative.I_KEY,
                            DensePoseDataRelative.U_KEY,
                            DensePoseDataRelative.V_KEY,
                        ]
                    ),
                    ", ".join(
                        [
                            DensePoseDataRelative.VERTEX_IDS_KEY,
                            DensePoseDataRelative.MESH_NAME_KEY,
                        ]
                    ),
                ),
            )
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
            DensePoseDataRelative.VERTEX_IDS_KEY,
            DensePoseDataRelative.MESH_NAME_KEY,
        ]:
            if key in annotation:
                del annotation[key]

    def apply_transform(self, transforms, densepose_transform_data):
        self._transform_pts(transforms, densepose_transform_data)
        if hasattr(self, "segm"):
            self._transform_segm(transforms, densepose_transform_data)

    def _transform_pts(self, transforms, dp_transform_data):
        import detectron2.data.transforms as T

        # NOTE: This assumes that HorizFlipTransform is the only one that does flip
        do_hflip = sum(isinstance(t, T.HFlipTransform) for t in transforms.transforms) % 2 == 1
        if do_hflip:
            self.x = self.MASK_SIZE - self.x
            if hasattr(self, "i"):
                self._flip_iuv_semantics(dp_transform_data)
            if hasattr(self, "vertex_ids"):
                self._flip_vertices()

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

    def _flip_vertices(self):
        mesh_info = MeshCatalog[MeshCatalog.get_mesh_name(self.mesh_id)]
        mesh_symmetry = (
            load_mesh_symmetry(mesh_info.symmetry) if mesh_info.symmetry is not None else None
        )
        self.vertex_ids = mesh_symmetry["vertex_transforms"][self.vertex_ids]

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
