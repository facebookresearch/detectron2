# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import numpy as np
from typing import List, Optional, Tuple
import cv2
import torch
import torch.nn.functional as F

from detectron2.utils.file_io import PathManager

from densepose.modeling import build_densepose_embedder

from ..modeling.cse.utils import squared_euclidean_distance_matrix
from ..structures import DensePoseEmbeddingPredictorOutput
from .base import Boxes, Image, MatrixVisualizer


def get_smpl_euclidean_vertex_embedding():
    embed_path = PathManager.get_local_path(
        "https://dl.fbaipublicfiles.com/densepose/data/cse/mds_d=256.npy"
    )
    embed_map, _ = np.load(embed_path, allow_pickle=True)
    return torch.tensor(embed_map).float()


DEFAULT_CLASS_TO_MESH_NAME = {0: "smpl_27554", 1: "cat_5001", 2: "dog_5002"}


class DensePoseOutputsVertexVisualizer(object):
    def __init__(
        self,
        inplace=True,
        cmap=cv2.COLORMAP_JET,
        alpha=0.7,
        cfg=None,
        device="cuda",
        default_class=0,
        class_to_mesh_name=DEFAULT_CLASS_TO_MESH_NAME,
        **kwargs,
    ):
        self.mask_visualizer = MatrixVisualizer(
            inplace=inplace, cmap=cmap, val_scale=1.0, alpha=alpha
        )
        self.cfg = cfg
        self.device = torch.device(device)
        self.class_to_mesh_name = class_to_mesh_name
        self.default_class = default_class

    def visualize(
        self,
        image_bgr: Image,
        outputs_boxes_xywh_classes: Tuple[
            Optional[DensePoseEmbeddingPredictorOutput], Optional[Boxes], Optional[List[int]]
        ],
    ) -> Image:
        if outputs_boxes_xywh_classes[0] is None:
            return image_bgr

        embedder = build_densepose_embedder(self.cfg)

        embed_map_rescaled = get_smpl_euclidean_vertex_embedding()[:, 0]
        embed_map_rescaled -= embed_map_rescaled.min()
        embed_map_rescaled /= embed_map_rescaled.max()

        S, E, N, bboxes_xywh, pred_classes = self.extract_and_check_outputs_and_boxes(
            outputs_boxes_xywh_classes
        )

        mesh_vertex_embeddings = {
            p: embedder(self.class_to_mesh_name[p]).to(self.device) for p in np.unique(pred_classes)
        }

        for n in range(N):
            x, y, w, h = bboxes_xywh[n].int().tolist()
            closest_vertices, mask = self.get_closest_vertices_mask_from_ES(
                E[[n]], S[[n]], h, w, mesh_vertex_embeddings[pred_classes[n]]
            )
            vis = (embed_map_rescaled[closest_vertices].clip(0, 1) * 255.0).cpu().numpy()
            mask_numpy = mask.cpu().numpy().astype(dtype=np.uint8)
            image_bgr = self.mask_visualizer.visualize(image_bgr, mask_numpy, vis, [x, y, w, h])

        return image_bgr

    def extract_and_check_outputs_and_boxes(self, outputs_boxes_xywh_classes):

        densepose_output, bboxes_xywh, pred_classes = outputs_boxes_xywh_classes

        if pred_classes is None:
            pred_classes = [self.default_class] * len(bboxes_xywh)

        assert isinstance(
            densepose_output, DensePoseEmbeddingPredictorOutput
        ), "DensePoseEmbeddingPredictorOutput expected, {} encountered".format(
            type(densepose_output)
        )

        S = densepose_output.coarse_segm
        E = densepose_output.embedding
        N = S.size(0)
        assert N == E.size(
            0
        ), "CSE coarse_segm {} and embeddings {}" " should have equal first dim size".format(
            S.size(), E.size()
        )
        assert N == len(
            bboxes_xywh
        ), "number of bounding boxes {}" " should be equal to first dim size of outputs {}".format(
            len(bboxes_xywh), N
        )
        assert N == len(pred_classes), (
            "number of predicted classes {}"
            " should be equal to first dim size of outputs {}".format(len(bboxes_xywh), N)
        )

        return S, E, N, bboxes_xywh, pred_classes

    def get_closest_vertices_mask_from_ES(self, En, Sn, h, w, mesh_vertex_embedding):
        embedding_resized = F.interpolate(En, size=(h, w), mode="bilinear")[0].to(self.device)
        coarse_segm_resized = F.interpolate(Sn, size=(h, w), mode="bilinear")[0].to(self.device)
        mask = coarse_segm_resized.argmax(0) > 0
        closest_vertices = torch.zeros(mask.shape, dtype=torch.long, device=self.device)
        all_embeddings = embedding_resized[:, mask].t()
        size_chunk = 10_000  # Chunking to avoid possible OOM
        edm = []
        for chunk in range((len(all_embeddings) - 1) // size_chunk + 1):
            chunk_embeddings = all_embeddings[size_chunk * chunk : size_chunk * (chunk + 1)]
            edm.append(squared_euclidean_distance_matrix(chunk_embeddings, mesh_vertex_embedding))
        closest_vertices[mask] = torch.cat(edm).argmin(dim=1)
        return closest_vertices, mask
