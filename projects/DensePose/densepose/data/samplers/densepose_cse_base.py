# Copyright (c) Facebook, Inc. and its affiliates.

from typing import Any, Dict, List, Tuple
import torch
from torch.nn import functional as F

from detectron2.config import CfgNode
from detectron2.structures import Instances

from densepose.converters.base import IntTupleBox
from densepose.data.utils import get_class_to_mesh_name_mapping
from densepose.modeling.cse.utils import squared_euclidean_distance_matrix
from densepose.structures import DensePoseDataRelative

from .densepose_base import DensePoseBaseSampler


class DensePoseCSEBaseSampler(DensePoseBaseSampler):
    """
    Base DensePose sampler to produce DensePose data from DensePose predictions.
    Samples for each class are drawn according to some distribution over all pixels estimated
    to belong to that class.
    """

    def __init__(
        self,
        cfg: CfgNode,
        use_gt_categories: bool,
        embedder: torch.nn.Module,
        count_per_class: int = 8,
    ):
        """
        Constructor

        Args:
          cfg (CfgNode): the config of the model
          embedder (torch.nn.Module): necessary to compute mesh vertex embeddings
          count_per_class (int): the sampler produces at most `count_per_class`
              samples for each category
        """
        super().__init__(count_per_class)
        self.embedder = embedder
        self.class_to_mesh_name = get_class_to_mesh_name_mapping(cfg)
        self.use_gt_categories = use_gt_categories

    def _sample(self, instance: Instances, bbox_xywh: IntTupleBox) -> Dict[str, List[Any]]:
        """
        Sample DensPoseDataRelative from estimation results
        """
        if self.use_gt_categories:
            instance_class = instance.dataset_classes.tolist()[0]
        else:
            instance_class = instance.pred_classes.tolist()[0]
        mesh_name = self.class_to_mesh_name[instance_class]

        annotation = {
            DensePoseDataRelative.X_KEY: [],
            DensePoseDataRelative.Y_KEY: [],
            DensePoseDataRelative.VERTEX_IDS_KEY: [],
            DensePoseDataRelative.MESH_NAME_KEY: mesh_name,
        }

        mask, embeddings, other_values = self._produce_mask_and_results(instance, bbox_xywh)
        indices = torch.nonzero(mask, as_tuple=True)
        selected_embeddings = embeddings.permute(1, 2, 0)[indices]
        values = other_values[:, indices[0], indices[1]]
        k = values.shape[1]

        count = min(self.count_per_class, k)
        if count <= 0:
            return annotation

        index_sample = self._produce_index_sample(values, count)
        closest_vertices = squared_euclidean_distance_matrix(
            selected_embeddings[index_sample], self.embedder(mesh_name)
        )
        closest_vertices = torch.argmin(closest_vertices, dim=1)

        sampled_y = indices[0][index_sample] + 0.5
        sampled_x = indices[1][index_sample] + 0.5
        # prepare / normalize data
        _, _, w, h = bbox_xywh
        x = (sampled_x / w * 256.0).cpu().tolist()
        y = (sampled_y / h * 256.0).cpu().tolist()
        # extend annotations
        annotation[DensePoseDataRelative.X_KEY].extend(x)
        annotation[DensePoseDataRelative.Y_KEY].extend(y)
        annotation[DensePoseDataRelative.VERTEX_IDS_KEY].extend(closest_vertices.cpu().tolist())
        return annotation

    def _produce_mask_and_results(
        self, instance: Instances, bbox_xywh: IntTupleBox
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Method to get labels and DensePose results from an instance

        Args:
            instance (Instances): an instance of `DensePoseEmbeddingPredictorOutput`
            bbox_xywh (IntTupleBox): the corresponding bounding box

        Return:
            mask (torch.Tensor): shape [H, W], DensePose segmentation mask
            embeddings (Tuple[torch.Tensor]): a tensor of shape [D, H, W],
                DensePose CSE Embeddings
            other_values (Tuple[torch.Tensor]): a tensor of shape [0, H, W],
                for potential other values
        """
        densepose_output = instance.pred_densepose
        S = densepose_output.coarse_segm
        E = densepose_output.embedding
        _, _, w, h = bbox_xywh
        embeddings = F.interpolate(E, size=(h, w), mode="bilinear")[0]
        coarse_segm_resized = F.interpolate(S, size=(h, w), mode="bilinear")[0]
        mask = coarse_segm_resized.argmax(0) > 0
        other_values = torch.empty((0, h, w), device=E.device)
        return mask, embeddings, other_values

    def _resample_mask(self, output: Any) -> torch.Tensor:
        """
        Convert DensePose predictor output to segmentation annotation - tensors of size
        (256, 256) and type `int64`.

        Args:
            output: DensePose predictor output with the following attributes:
             - coarse_segm: tensor of size [N, D, H, W] with unnormalized coarse
               segmentation scores
        Return:
            Tensor of size (S, S) and type `int64` with coarse segmentation annotations,
            where S = DensePoseDataRelative.MASK_SIZE
        """
        sz = DensePoseDataRelative.MASK_SIZE
        mask = (
            F.interpolate(output.coarse_segm, (sz, sz), mode="bilinear", align_corners=False)
            .argmax(dim=1)
            .long()
            .squeeze()
            .cpu()
        )
        return mask
