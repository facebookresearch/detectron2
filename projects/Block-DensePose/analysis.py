import logging
import time
import datetime
from contextlib import ExitStack
import copy
import numpy as np
from typing import Any, Dict, List, Tuple
from scipy.io import loadmat
import pickle
import os
import scipy.spatial.distance as ssd

from datetime import timedelta

from detectron2.config.config import CfgNode
import torch
from torch import nn

import detectron2.utils.comm as comm
from detectron2.config import get_cfg
from detectron2.engine import DEFAULT_TIMEOUT, default_argument_parser, default_setup, launch
from detectron2.utils.file_io import PathManager
from detectron2.utils.logger import setup_logger, log_every_n_seconds
from detectron2.evaluation.evaluator import inference_context
from detectron2.data import MetadataCatalog
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.layers import ROIAlign
from detectron2.structures import BoxMode

from densepose import add_densepose_config
from densepose.engine import Trainer
from densepose.modeling.densepose_checkpoint import DensePoseCheckpointer
from densepose.data import build_detection_test_loader
from densepose.data.dataset_mapper import build_augmentation
from densepose.structures import DensePoseTransformData, DensePoseDataRelative, DensePoseList
from densepose.modeling.losses.utils import (
    ChartBasedAnnotationsAccumulator,
    extract_packed_annotations_from_matches,
    BilinearInterpolationHelper,
)


def setup(args):
    cfg = get_cfg()
    add_densepose_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    # Setup logger for "densepose" module
    setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="densepose")
    return cfg


class Mapper:
    def __init__(self, cfg):
        self.augmentation = build_augmentation(cfg, False)

        self.img_format = cfg.INPUT.FORMAT
        self.mask_on = (
                cfg.MODEL.MASK_ON or (
                cfg.MODEL.DENSEPOSE_ON
                and cfg.MODEL.ROI_DENSEPOSE_HEAD.COARSE_SEGM_TRAINED_BY_MASKS)
        )
        self.keypoint_on = cfg.MODEL.KEYPOINT_ON
        self.densepose_on = cfg.MODEL.DENSEPOSE_ON
        assert not cfg.MODEL.LOAD_PROPOSALS, "not supported yet"
        # fmt: on
        if self.keypoint_on:
            # Flip only makes sense in training
            self.keypoint_hflip_indices = utils.create_keypoint_hflip_indices(cfg.DATASETS.TRAIN)
        else:
            self.keypoint_hflip_indices = None

        if self.densepose_on:
            densepose_transform_srcs = [
                MetadataCatalog.get(ds).densepose_transform_src
                for ds in cfg.DATASETS.TRAIN + cfg.DATASETS.TEST
            ]
            assert len(densepose_transform_srcs) > 0
            densepose_transform_data_fpath = PathManager.get_local_path(densepose_transform_srcs[0])
            self.densepose_transform_data = DensePoseTransformData.load(
                densepose_transform_data_fpath
            )

    def __call__(self, dataset_dict):
        dataset_dict = copy.deepcopy(dataset_dict)
        image = utils.read_image(dataset_dict["file_name"], format=self.img_format)
        utils.check_image_size(dataset_dict, image)

        image, transforms = T.apply_transform_gens(self.augmentation, image)
        image_shape = image.shape[:2]  # h, w
        dataset_dict["image"] = torch.as_tensor(image.transpose(2, 0, 1).astype("float32"))

        for anno in dataset_dict["annotations"]:
            if not self.mask_on:
                anno.pop("segmentation", None)
            if not self.keypoint_on:
                anno.pop("keypoints", None)

        # USER: Implement additional transformations if you have other types of data
        # USER: Don't call transpose_densepose if you don't need
        annos = [
            self._transform_densepose(
                utils.transform_instance_annotations(
                    obj, transforms, image_shape, keypoint_hflip_indices=self.keypoint_hflip_indices
                ),
                transforms,
            )
            for obj in dataset_dict.pop("annotations")
            if obj.get("iscrowd", 0) == 0
        ]

        if self.mask_on:
            self._add_densepose_masks_as_segmentation(annos, image_shape)

        instances = utils.annotations_to_instances(annos, image_shape, mask_format="bitmask")
        densepose_annotations = [obj.get("densepose") for obj in annos]
        if densepose_annotations and not all(v is None for v in densepose_annotations):
            instances.gt_densepose = DensePoseList(
                densepose_annotations, instances.gt_boxes, image_shape
            )

        dataset_dict["instances"] = instances[instances.gt_boxes.nonempty()]
        return dataset_dict

    def _transform_densepose(self, annotation, transforms):
        if not self.densepose_on:
            return annotation

        # Handle densepose annotations
        is_valid, reason_not_valid = DensePoseDataRelative.validate_annotation(annotation)
        if is_valid:
            densepose_data = DensePoseDataRelative(annotation, cleanup=True)
            densepose_data.apply_transform(transforms, self.densepose_transform_data)
            annotation["densepose"] = densepose_data
        else:
            # logger = logging.getLogger(__name__)
            # logger.debug("Could not load DensePose annotation: {}".format(reason_not_valid))
            DensePoseDataRelative.cleanup_annotation(annotation)
            # NOTE: annotations for certain instances may be unavailable.
            # 'None' is accepted by the DensePostList data structure.
            annotation["densepose"] = None
        return annotation

    def _add_densepose_masks_as_segmentation(
            self, annotations: List[Dict[str, Any]], image_shape_hw: Tuple[int, int]
    ):
        for obj in annotations:
            if ("densepose" not in obj) or ("segmentation" in obj):
                continue
            # DP segmentation: torch.Tensor [S, S] of float32, S=256
            segm_dp = torch.zeros_like(obj["densepose"].segm)
            segm_dp[obj["densepose"].segm > 0] = 1
            segm_h, segm_w = segm_dp.shape
            bbox_segm_dp = torch.tensor((0, 0, segm_h - 1, segm_w - 1), dtype=torch.float32)
            # image bbox
            x0, y0, x1, y1 = (
                v.item() for v in BoxMode.convert(obj["bbox"], obj["bbox_mode"], BoxMode.XYXY_ABS)
            )
            segm_aligned = (
                ROIAlign((y1 - y0, x1 - x0), 1.0, 0, aligned=True)
                    .forward(segm_dp.view(1, 1, *segm_dp.shape), bbox_segm_dp)
                    .squeeze()
            )
            image_mask = torch.zeros(*image_shape_hw, dtype=torch.float32)
            image_mask[y0:y1, x0:x1] = segm_aligned
            # segmentation for BitMask: np.array [H, W] of np.bool
            obj["segmentation"] = image_mask >= 0.5


def build_loader(cfg):
    return build_detection_test_loader(cfg, cfg.DATASETS.TEST[0], mapper=Mapper(cfg))


def test(
        cfg: CfgNode,
        model: nn.Module,
):
    data_loader = build_loader(cfg)
    if cfg.DENSEPOSE_EVALUATION.DISTRIBUTED_INFERENCE or comm.is_main_process():
        segms, dists = inference_on_dataset(model, data_loader)
        output_dir = cfg.OUTPUT_DIR
        segms_file_path = os.path.join(output_dir, 'segms.npy')
        dists_file_path = os.path.join(output_dir, 'dists.npy')
        with PathManager.open(segms_file_path, 'wb') as f:
            np.save(f, torch.from_numpy(np.concatenate(segms)))
        with PathManager.open(dists_file_path, 'wb') as f:
            np.save(f, torch.from_numpy(np.concatenate(dists)))


def inference_on_dataset(model, data_loader):
    num_devices = comm.get_world_size()
    logger = logging.getLogger(__name__)
    logger.info("Start inference on {} batches".format(len(data_loader)))
    total = len(data_loader)

    num_warmup = min(5, total - 1)
    start_time = time.perf_counter()
    total_data_time = 0
    total_compute_time = 0
    total_eval_time = 0

    dists = []
    segms = []
    utils = Utils()

    with ExitStack() as stack:
        if isinstance(model, nn.Module):
            stack.enter_context(inference_context(model))
        stack.enter_context(torch.no_grad())

        start_data_time = time.perf_counter()
        for idx, inputs in enumerate(data_loader):
            total_data_time += time.perf_counter() - start_data_time
            detected_instances = [x['instances'] for x in inputs]
            for x in detected_instances:
                x.pred_boxes = x.gt_boxes
                x.pred_classes = x.gt_classes
            start_compute_time = time.perf_counter()
            outputs = model.inference(inputs, detected_instances, do_postprocess=False)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            total_compute_time += time.perf_counter() - start_compute_time

            start_eval_time = time.perf_counter()
            for x in outputs:
                x.proposal_boxes = x.pred_boxes
            # from chart loss
            accumulator = ChartBasedAnnotationsAccumulator()
            packed_annotations = extract_packed_annotations_from_matches(outputs, accumulator)

            if packed_annotations is None:
                continue
            # get densepose prediction
            output_type = type(outputs[0].pred_densepose)
            outputs_dict = {
                'coarse_segm': torch.concat([output.pred_densepose.coarse_segm for output in outputs], dim=0),
                'fine_segm': torch.concat([output.pred_densepose.fine_segm for output in outputs], dim=0),
                'u': torch.concat([output.pred_densepose.u for output in outputs], dim=0),
                'v': torch.concat([output.pred_densepose.v for output in outputs], dim=0),
            }
            densepose_outputs = output_type(**outputs_dict)
            h, w = densepose_outputs.u.shape[2:]
            interpolator = BilinearInterpolationHelper.from_matches(
                packed_annotations,
                (h, w),
            )

            j_valid_fg = interpolator.j_valid * (
                packed_annotations.fine_segm_labels_gt > 0
            )
            if not torch.any(j_valid_fg):
                continue

            # get est segm and uv coordinates
            s_dt = interpolator.extract_at_points(
                densepose_outputs.coarse_segm,
                slice_fine_segm=slice(None),
                w_ylo_xlo=interpolator.w_ylo_xlo[:, None],  # pyre-ignore[16]
                w_ylo_xhi=interpolator.w_ylo_xhi[:, None],  # pyre-ignore[16]
                w_yhi_xlo=interpolator.w_yhi_xlo[:, None],  # pyre-ignore[16]
                w_yhi_xhi=interpolator.w_yhi_xhi[:, None],
            )
            s_dt = torch.argmin(s_dt, dim=1) > 0
            j_valid = interpolator.j_valid
            i_dt = interpolator.extract_at_points(
                densepose_outputs.fine_segm,
                slice_fine_segm=slice(None),
                w_ylo_xlo=interpolator.w_ylo_xlo[:, None],  # pyre-ignore[16]
                w_ylo_xhi=interpolator.w_ylo_xhi[:, None],  # pyre-ignore[16]
                w_yhi_xlo=interpolator.w_yhi_xlo[:, None],  # pyre-ignore[16]
                w_yhi_xhi=interpolator.w_yhi_xhi[:, None],
            )[j_valid, :]
            i_dt = torch.argmax(i_dt, dim=1).int()
            i_dt[~s_dt] = 0
            i_dt = i_dt.long()
            u_dt = interpolator.extract_at_points(densepose_outputs.u, slice_fine_segm=i_dt)[j_valid]
            v_dt = interpolator.extract_at_points(densepose_outputs.v, slice_fine_segm=i_dt)[j_valid]

            # u_dt = u_dt[torch.arange(u_dt.shape[0]), torch.argmax(i_dt)]
            # v_dt = v_dt[torch.arange(v_dt.shape[0]), torch.argmax(i_dt)]

            u_dt = (u_dt * 255).clamp(0, 255).byte() / 255.0
            v_dt = (v_dt * 255).clamp(0, 255).byte() / 255.0

            i_gt = packed_annotations.fine_segm_labels_gt[j_valid]
            u_gt = packed_annotations.u_gt[j_valid]
            v_gt = packed_annotations.v_gt[j_valid]

            cVertsGT, ClosestVertsGTTransformed = utils.findAllClosestVertsGT(i_gt, u_gt, v_gt)
            cVerts = utils.findAllClosestVertsUV(u_dt, v_dt, i_dt)
            dist = utils.getDistancesUV(ClosestVertsGTTransformed, cVerts)

            segms.append((i_dt == i_gt).int().cpu())
            dists.append(dist)

            total_eval_time += time.perf_counter() - start_eval_time

            iters_after_start = idx + 1 - num_warmup * int(idx >= num_warmup)
            data_seconds_per_iter = total_data_time / iters_after_start
            compute_seconds_per_iter = total_compute_time / iters_after_start
            eval_seconds_per_iter = total_eval_time / iters_after_start
            total_seconds_per_iter = (time.perf_counter() - start_time) / iters_after_start
            if idx >= num_warmup * 2 or compute_seconds_per_iter > 5:
                eta = datetime.timedelta(seconds=int(total_seconds_per_iter * (total - idx - 1)))
                print(
                    f"Inference done {idx + 1}/{total}. "
                    f"Dataloading: {data_seconds_per_iter:.4f} s/iter. "
                    f"Inference: {compute_seconds_per_iter:.4f} s/iter. "
                    f"Eval: {eval_seconds_per_iter:.4f} s/iter. "
                    f"Total: {total_seconds_per_iter:.4f} s/iter. "
                    f"ETA={eta}"
                )
            start_data_time = time.perf_counter()
        total_time = time.perf_counter() - start_time
        total_time_str = str(datetime.timedelta(seconds=total_time))

        logger.info(
            "Total inference time: {} ({:.6f} s / iter per device, on {} devices)".format(
                total_time_str, total_time / (total - num_warmup), num_devices
            )
        )
        total_compute_time_str = str(datetime.timedelta(seconds=int(total_compute_time)))
        logger.info(
            "Total inference pure compute time: {} ({:.6f} s / iter per device, on {} devices)".format(
                total_compute_time_str, total_compute_time / (total - num_warmup), num_devices
            )
        )
    return segms, dists


class Utils:
    def __init__(self) -> None:
        self.loadGEval()

    def loadGEval(self):
            smpl_subdiv_fpath = PathManager.get_local_path(
                "/home/wenjiaxiao/detectron2/datasets/coco/annotations/SMPL_subdiv.mat"
            )
            pdist_transform_fpath = PathManager.get_local_path(
                "/home/wenjiaxiao/detectron2/datasets/coco/annotations/SMPL_SUBDIV_TRANSFORM.mat"
            )
            pdist_matrix_fpath = PathManager.get_local_path(
                "https://dl.fbaipublicfiles.com/densepose/data/Pdist_matrix.pkl", timeout_sec=120
            )
            SMPL_subdiv = loadmat(smpl_subdiv_fpath)
            self.PDIST_transform = loadmat(pdist_transform_fpath)
            self.PDIST_transform = self.PDIST_transform["index"].squeeze()
            UV = np.array([SMPL_subdiv["U_subdiv"], SMPL_subdiv["V_subdiv"]]).squeeze()
            ClosestVertInds = np.arange(UV.shape[1]) + 1
            self.Part_UVs = []
            self.Part_ClosestVertInds = []
            for i in np.arange(24):
                self.Part_UVs.append(UV[:, SMPL_subdiv["Part_ID_subdiv"].squeeze() == (i + 1)])
                self.Part_ClosestVertInds.append(
                    ClosestVertInds[SMPL_subdiv["Part_ID_subdiv"].squeeze() == (i + 1)]
                )

            with open(pdist_matrix_fpath, "rb") as hFile:
                arrays = pickle.load(hFile, encoding="latin1")
            self.Pdist_matrix = arrays["Pdist_matrix"]
            self.Part_ids = np.array(SMPL_subdiv["Part_ID_subdiv"].squeeze())
            # Mean geodesic distances for parts.
            self.Mean_Distances = np.array([0, 0.351, 0.107, 0.126, 0.237, 0.173, 0.142, 0.128, 0.150])
            # Coarse Part labels.
            self.CoarseParts = np.array(
                [0, 1, 1, 2, 2, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7, 8, 8]
            )

    def findAllClosestVertsGT(self, i_gt, u_gt, v_gt):
            #
            I_gt = np.array(i_gt.cpu())
            U_gt = np.array(u_gt.cpu())
            V_gt = np.array(v_gt.cpu())

            ClosestVertsGT = np.ones(I_gt.shape) * -1
            for i in np.arange(24):
                if (i + 1) in I_gt:
                    UVs = np.array([U_gt[I_gt == (i + 1)], V_gt[I_gt == (i + 1)]])
                    Current_Part_UVs = self.Part_UVs[i]
                    Current_Part_ClosestVertInds = self.Part_ClosestVertInds[i]
                    D = ssd.cdist(Current_Part_UVs.transpose(), UVs.transpose()).squeeze()
                    ClosestVertsGT[I_gt == (i + 1)] = Current_Part_ClosestVertInds[np.argmin(D, axis=0)]
            #
            ClosestVertsGTTransformed = self.PDIST_transform[ClosestVertsGT.astype(int) - 1]
            ClosestVertsGTTransformed[ClosestVertsGT < 0] = 0
            return ClosestVertsGT, ClosestVertsGTTransformed

    def findAllClosestVertsUV(self, u_dt, v_dt, i_dt):
        U_points = np.array(u_dt.cpu())
        V_points = np.array(v_dt.cpu())
        Index_points = np.array(i_dt.cpu())
        ClosestVerts = np.ones(Index_points.shape) * -1
        for i in np.arange(24):
            #
            if (i + 1) in Index_points:
                UVs = np.array(
                    [U_points[Index_points == (i + 1)], V_points[Index_points == (i + 1)]]
                )
                Current_Part_UVs = self.Part_UVs[i]
                Current_Part_ClosestVertInds = self.Part_ClosestVertInds[i]
                D = ssd.cdist(Current_Part_UVs.transpose(), UVs.transpose()).squeeze()
                ClosestVerts[Index_points == (i + 1)] = Current_Part_ClosestVertInds[
                    np.argmin(D, axis=0)
                ]
        ClosestVertsTransformed = self.PDIST_transform[ClosestVerts.astype(int) - 1]
        ClosestVertsTransformed[ClosestVerts < 0] = 0
        return ClosestVertsTransformed

    def getDistancesUV(self, cVertsGT, cVerts):
        #
        n = 27554
        dists = []
        for d in range(len(cVertsGT)):
            if cVertsGT[d] > 0:
                if cVerts[d] > 0:
                    i = cVertsGT[d] - 1
                    j = cVerts[d] - 1
                    if j == i:
                        dists.append(0)
                    elif j > i:
                        ccc = i
                        i = j
                        j = ccc
                        i = n - i - 1
                        j = n - j - 1
                        k = (n * (n - 1) / 2) - (n - i) * ((n - i) - 1) / 2 + j - i - 1
                        k = (n * n - n) / 2 - k - 1
                        dists.append(self.Pdist_matrix[int(k)][0])
                    else:
                        i = n - i - 1
                        j = n - j - 1
                        k = (n * (n - 1) / 2) - (n - i) * ((n - i) - 1) / 2 + j - i - 1
                        k = (n * n - n) / 2 - k - 1
                        dists.append(self.Pdist_matrix[int(k)][0])
                else:
                    dists.append(np.inf)
        return np.atleast_1d(np.array(dists).squeeze())


def main(args):
    cfg = setup(args)
    # disable strict kwargs checking: allow one to specify path handle
    # hints through kwargs, like timeout in DP evaluation
    PathManager.set_strict_kwargs_checking(False)

    model = Trainer.build_model(cfg)
    DensePoseCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
        cfg.MODEL.WEIGHTS, resume=args.resume
    )
    test(cfg, model)


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    cfg = setup(args)
    timeout = (
        DEFAULT_TIMEOUT if cfg.DENSEPOSE_EVALUATION.DISTRIBUTED_INFERENCE else timedelta(hours=4)
    )
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
        timeout=timeout,
    )
