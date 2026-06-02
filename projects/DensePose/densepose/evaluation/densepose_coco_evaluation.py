# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# This is a modified version of cocoeval.py where we also have the densepose evaluation.

# pyre-unsafe

__author__ = "tsungyi"

import copy
import datetime
import logging
import numpy as np
import pickle
import time
from collections import defaultdict
from enum import Enum
from typing import Any, Dict, Tuple
import scipy.spatial.distance as ssd
import torch
import torch.nn.functional as F
from pycocotools import mask as maskUtils
from scipy.io import loadmat
from scipy.ndimage import zoom as spzoom

from detectron2.utils.file_io import PathManager

from densepose.converters.chart_output_to_chart_result import resample_uv_tensors_to_bbox
from densepose.converters.segm_to_mask import (
    resample_coarse_segm_tensor_to_bbox,
    resample_fine_and_coarse_segm_tensors_to_bbox,
)
from densepose.modeling.cse.utils import squared_euclidean_distance_matrix
from densepose.structures import DensePoseDataRelative
from densepose.structures.mesh import create_mesh

logger = logging.getLogger(__name__)


class DensePoseEvalMode(str, Enum):
    # use both masks and geodesic distances (GPS * IOU) to compute scores
    GPSM = "gpsm"
    # use only geodesic distances (GPS)  to compute scores
    GPS = "gps"
    # use only masks (IOU) to compute scores
    IOU = "iou"


class DensePoseDataMode(str, Enum):
    # use estimated IUV data (default mode)
    IUV_DT = "iuvdt"
    # use ground truth IUV data
    IUV_GT = "iuvgt"
    # use ground truth labels I and set UV to 0
    I_GT_UV_0 = "igtuv0"
    # use ground truth labels I and estimated UV coordinates
    I_GT_UV_DT = "igtuvdt"
    # use estimated labels I and set UV to 0
    I_DT_UV_0 = "idtuv0"


class DensePoseCocoEval:
    # Interface for evaluating detection on the Microsoft COCO dataset.
    #
    # The usage for CocoEval is as follows:
    #  cocoGt=..., cocoDt=...       # load dataset and results
    #  E = CocoEval(cocoGt,cocoDt); # initialize CocoEval object
    #  E.params.recThrs = ...;      # set parameters as desired
    #  E.evaluate();                # run per image evaluation
    #  E.accumulate();              # accumulate per image results
    #  E.summarize();               # display summary metrics of results
    # For example usage see evalDemo.m and http://mscoco.org/.
    #
    # The evaluation parameters are as follows (defaults in brackets):
    #  imgIds     - [all] N img ids to use for evaluation
    #  catIds     - [all] K cat ids to use for evaluation
    #  iouThrs    - [.5:.05:.95] T=10 IoU thresholds for evaluation
    #  recThrs    - [0:.01:1] R=101 recall thresholds for evaluation
    #  areaRng    - [...] A=4 object area ranges for evaluation
    #  maxDets    - [1 10 100] M=3 thresholds on max detections per image
    #  iouType    - ['segm'] set iouType to 'segm', 'bbox', 'keypoints' or 'densepose'
    #  iouType replaced the now DEPRECATED useSegm parameter.
    #  useCats    - [1] if true use category labels for evaluation
    # Note: if useCats=0 category labels are ignored as in proposal scoring.
    # Note: multiple areaRngs [Ax2] and maxDets [Mx1] can be specified.
    #
    # evaluate(): evaluates detections on every image and every category and
    # concats the results into the "evalImgs" with fields:
    #  dtIds      - [1xD] id for each of the D detections (dt)
    #  gtIds      - [1xG] id for each of the G ground truths (gt)
    #  dtMatches  - [TxD] matching gt id at each IoU or 0
    #  gtMatches  - [TxG] matching dt id at each IoU or 0
    #  dtScores   - [1xD] confidence of each dt
    #  gtIgnore   - [1xG] ignore flag for each gt
    #  dtIgnore   - [TxD] ignore flag for each dt at each IoU
    #
    # accumulate(): accumulates the per-image, per-category evaluation
    # results in "evalImgs" into the dictionary "eval" with fields:
    #  params     - parameters used for evaluation
    #  date       - date evaluation was performed
    #  counts     - [T,R,K,A,M] parameter dimensions (see above)
    #  precision  - [TxRxKxAxM] precision for every evaluation setting
    #  recall     - [TxKxAxM] max recall for every evaluation setting
    # Note: precision and recall==-1 for settings with no gt objects.
    #
    # See also coco, mask, pycocoDemo, pycocoEvalDemo
    #
    # Microsoft COCO Toolbox.      version 2.0
    # Data, paper, and tutorials available at:  http://mscoco.org/
    # Code written by Piotr Dollar and Tsung-Yi Lin, 2015.
    # Licensed under the Simplified BSD License [see coco/license.txt]
    def __init__(
        self,
        cocoGt=None,
        cocoDt=None,
        iouType: str = "densepose",
        multi_storage=None,
        embedder=None,
        dpEvalMode: DensePoseEvalMode = DensePoseEvalMode.GPS,
        dpDataMode: DensePoseDataMode = DensePoseDataMode.IUV_DT,
    ):
        """
        Initialize CocoEval using coco APIs for gt and dt
        :param cocoGt: coco object with ground truth annotations
        :param cocoDt: coco object with detection results
        :return: None
        """
        self.cocoGt = cocoGt  # ground truth COCO API
        self.cocoDt = cocoDt  # detections COCO API
        self.multi_storage = multi_storage
        self.embedder = embedder
        self._dpEvalMode = dpEvalMode
        self._dpDataMode = dpDataMode
        self.evalImgs = defaultdict(list)  # per-image per-category eval results [KxAxI]
        self.eval = {}  # accumulated evaluation results
        self._gts = defaultdict(list)  # gt for evaluation
        self._dts = defaultdict(list)  # dt for evaluation
        self.params = Params(iouType=iouType)  # parameters
        self._paramsEval = {}  # parameters for evaluation
        self.stats = []  # result summarization
        self.ious = {}  # ious between all gts and dts
        if cocoGt is not None:
            self.params.imgIds = sorted(cocoGt.getImgIds())
            self.params.catIds = sorted(cocoGt.getCatIds())
        self.ignoreThrBB = 0.7
        self.ignoreThrUV = 0.9

    def _loadGEval(self):
        smpl_subdiv_fpath = PathManager.get_local_path(
            "https://dl.fbaipublicfiles.com/densepose/data/SMPL_subdiv.mat"
        )
        pdist_transform_fpath = PathManager.get_local_path(
            "https://dl.fbaipublicfiles.com/densepose/data/SMPL_SUBDIV_TRANSFORM.mat"
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

    def _prepare(self):
        """
        Prepare ._gts and ._dts for evaluation based on params
        :return: None
        """

        def _toMask(anns, coco):
            # modify ann['segmentation'] by reference
            for ann in anns:
                # safeguard for invalid segmentation annotation;
                # annotations containing empty lists exist in the posetrack
                # dataset. This is not a correct segmentation annotation
                # in terms of COCO format; we need to deal with it somehow
                segm = ann["segmentation"]
                if type(segm) is list and len(segm) == 0:
                    ann["segmentation"] = None
                    continue
                rle = coco.annToRLE(ann)
                ann["segmentation"] = rle

        def _getIgnoreRegion(iid, coco):
            img = coco.imgs[iid]

            if "ignore_regions_x" not in img.keys():
                return None

            if len(img["ignore_regions_x"]) == 0:
                return None

            rgns_merged = [
                [v for xy in zip(region_x, region_y) for v in xy]
                for region_x, region_y in zip(img["ignore_regions_x"], img["ignore_regions_y"])
            ]
            rles = maskUtils.frPyObjects(rgns_merged, img["height"], img["width"])
            rle = maskUtils.merge(rles)
            return maskUtils.decode(rle)

        def _checkIgnore(dt, iregion):
            if iregion is None:
                return True

            bb = np.array(dt["bbox"]).astype(int)
            x1, y1, x2, y2 = bb[0], bb[1], bb[0] + bb[2], bb[1] + bb[3]
            x2 = min([x2, iregion.shape[1]])
            y2 = min([y2, iregion.shape[0]])

            if bb[2] * bb[3] == 0:
                return False

            crop_iregion = iregion[y1:y2, x1:x2]

            if crop_iregion.sum() == 0:
                return True

            if "densepose" not in dt.keys():  # filtering boxes
                return crop_iregion.sum() / bb[2] / bb[3] < self.ignoreThrBB

            # filtering UVs
            ignoremask = np.require(crop_iregion, requirements=["F"])
            mask = self._extract_mask(dt)
            uvmask = np.require(np.asarray(mask > 0), dtype=np.uint8, requirements=["F"])
            uvmask_ = maskUtils.encode(uvmask)
            ignoremask_ = maskUtils.encode(ignoremask)
            uviou = maskUtils.iou([uvmask_], [ignoremask_], [1])[0]
            return uviou < self.ignoreThrUV

        p = self.params

        if p.useCats:
            gts = self.cocoGt.loadAnns(self.cocoGt.getAnnIds(imgIds=p.imgIds, catIds=p.catIds))
            dts = self.cocoDt.loadAnns(self.cocoDt.getAnnIds(imgIds=p.imgIds, catIds=p.catIds))
        else:
            gts = self.cocoGt.loadAnns(self.cocoGt.getAnnIds(imgIds=p.imgIds))
            dts = self.cocoDt.loadAnns(self.cocoDt.getAnnIds(imgIds=p.imgIds))

        imns = self.cocoGt.loadImgs(p.imgIds)
        self.size_mapping = {}
        for im in imns:
            self.size_mapping[im["id"]] = [im["height"], im["width"]]

        # if iouType == 'uv', add point gt annotations
        if p.iouType == "densepose":
            self._loadGEval()

        # convert ground truth to mask if iouType == 'segm'
        if p.iouType == "segm":
            _toMask(gts, self.cocoGt)
            _toMask(dts, self.cocoDt)

        # set ignore flag
        for gt in gts:
            gt["ignore"] = gt["ignore"] if "ignore" in gt else 0
            gt["ignore"] = "iscrowd" in gt and gt["iscrowd"]
            if p.iouType == "keypoints":
                gt["ignore"] = (gt["num_keypoints"] == 0) or gt["ignore"]
            if p.iouType == "densepose":
                gt["ignore"] = ("dp_x" in gt) == 0
            if p.iouType == "segm":
                gt["ignore"] = gt["segmentation"] is None

        self._gts = defaultdict(list)  # gt for evaluation
        self._dts = defaultdict(list)  # dt for evaluation
        self._igrgns = defaultdict(list)

        for gt in gts:
            iid = gt["image_id"]
            if iid not in self._igrgns.keys():
                self._igrgns[iid] = _getIgnoreRegion(iid, self.cocoGt)
            if _checkIgnore(gt, self._igrgns[iid]):
                self._gts[iid, gt["category_id"]].append(gt)
        for dt in dts:
            iid = dt["image_id"]
            if (iid not in self._igrgns) or _checkIgnore(dt, self._igrgns[iid]):
                self._dts[iid, dt["category_id"]].append(dt)

        self.evalImgs = defaultdict(list)  # per-image per-category evaluation results
        self.eval = {}  # accumulated evaluation results

    def evaluate(self):
        """
        Run per image evaluation on given images and store results (a list of dict) in self.evalImgs
        :return: None
        """
        tic = time.time()
        logger.info("Running per image DensePose evaluation... {}".format(self.params.iouType))
        p = self.params
        # add backward compatibility if useSegm is specified in params
        if p.useSegm is not None:
            p.iouType = "segm" if p.useSegm == 1 else "bbox"
            logger.info("useSegm (deprecated) is not None. Running DensePose evaluation")
        p.imgIds = list(np.unique(p.imgIds))
        if p.useCats:
            p.catIds = list(np.unique(p.catIds))
        p.maxDets = sorted(p.maxDets)
        self.params = p

        self._prepare()
        # loop through images, area range, max detection number
        catIds = p.catIds if p.useCats else [-1]

        if p.iouType in ["segm", "bbox"]:
            computeIoU = self.computeIoU
        elif p.iouType == "keypoints":
            computeIoU = self.computeOks
        elif p.iouType == "densepose":
            computeIoU = self.computeOgps
            if self._dpEvalMode in {DensePoseEvalMode.GPSM, DensePoseEvalMode.IOU}:
                self.real_ious = {
                    (imgId, catId): self.computeDPIoU(imgId, catId)
                    for imgId in p.imgIds
                    for catId in catIds
                }

        self.ious = {
            (imgId, catId): computeIoU(imgId, catId) for imgId in p.imgIds for catId in catIds
        }

        evaluateImg = self.evaluateImg
        maxDet = p.maxDets[-1]
        self.evalImgs = [
            evaluateImg(imgId, catId, areaRng, maxDet)
            for catId in catIds
            for areaRng in p.areaRng
            for imgId in p.imgIds
        ]
        self._paramsEval = copy.deepcopy(self.params)
        toc = time.time()
        logger.info("DensePose evaluation DONE (t={:0.2f}s).".format(toc - tic))

    def getDensePoseMask(self, polys):
        maskGen = np.zeros([256, 256])
        stop = min(len(polys) + 1, 15)
        for i in range(1, stop):
            if polys[i - 1]:
                currentMask = maskUtils.decode(polys[i - 1])
                maskGen[currentMask > 0] = i
        return maskGen

    def _generate_rlemask_on_image(self, mask, imgId, data):
        bbox_xywh = np.array(data["bbox"])
        x, y, w, h = bbox_xywh
        im_h, im_w = self.size_mapping[imgId]
        im_mask = np.zeros((im_h, im_w), dtype=np.uint8)
        if mask is not None:
            x0 = max(int(x), 0)
            x1 = min(int(x + w), im_w, int(x) + mask.shape[1])
            y0 = max(int(y), 0)
            y1 = min(int(y + h), im_h, int(y) + mask.shape[0])
            y = int(y)
            x = int(x)
            im_mask[y0:y1, x0:x1] = mask[y0 - y : y1 - y, x0 - x : x1 - x]
        im_mask = np.require(np.asarray(im_mask > 0), dtype=np.uint8, requirements=["F"])
        rle_mask = maskUtils.encode(np.array(im_mask[:, :, np.newaxis], order="F"))[0]
        return rle_mask

    def computeDPIoU(self, imgId, catId):
        p = self.params
        if p.useCats:
            gt = self._gts[imgId, catId]
            dt = self._dts[imgId, catId]
        else:
            gt = [_ for cId in p.catIds for _ in self._gts[imgId, cId]]
            dt = [_ for cId in p.catIds for _ in self._dts[imgId, cId]]
        if len(gt) == 0 and len(dt) == 0:
            return []
        inds = np.argsort([-d["score"] for d in dt], kind="mergesort")
        dt = [dt[i] for i in inds]
        if len(dt) > p.maxDets[-1]:
            dt = dt[0 : p.maxDets[-1]]

        gtmasks = []
        for g in gt:
            if DensePoseDataRelative.S_KEY in g:
                # convert DensePose mask to a binary mask
                mask = np.minimum(self.getDensePoseMask(g[DensePoseDataRelative.S_KEY]), 1.0)
                _, _, w, h = g["bbox"]
                scale_x = float(max(w, 1)) / mask.shape[1]
                scale_y = float(max(h, 1)) / mask.shape[0]
                mask = spzoom(mask, (scale_y, scale_x), order=1, prefilter=False)
                mask = np.array(mask > 0.5, dtype=np.uint8)
                rle_mask = self._generate_rlemask_on_image(mask, imgId, g)
            elif "segmentation" in g:
                segmentation = g["segmentation"]
                if isinstance(segmentation, list) and segmentation:
                    # polygons
                    im_h, im_w = self.size_mapping[imgId]
                    rles = maskUtils.frPyObjects(segmentation, im_h, im_w)
                    rle_mask = maskUtils.merge(rles)
                elif isinstance(segmentation, dict):
                    if isinstance(segmentation["counts"], list):
                        # uncompressed RLE
                        im_h, im_w = self.size_mapping[imgId]
                        rle_mask = maskUtils.frPyObjects(segmentation, im_h, im_w)
                    else:
                        # compressed RLE
                        rle_mask = segmentation
                else:
                    rle_mask = self._generate_rlemask_on_image(None, imgId, g)
            else:
                rle_mask = self._generate_rlemask_on_image(None, imgId, g)
            gtmasks.append(rle_mask)

        dtmasks = []
        for d in dt:
            mask = self._extract_mask(d)
            mask = np.require(np.asarray(mask > 0), dtype=np.uint8, requirements=["F"])
            rle_mask = self._generate_rlemask_on_image(mask, imgId, d)
            dtmasks.append(rle_mask)

        # compute iou between each dt and gt region
        iscrowd = [int(o.get("iscrowd", 0)) for o in gt]
        iousDP = maskUtils.iou(dtmasks, gtmasks, iscrowd)
        return iousDP

    def computeIoU(self, imgId, catId):
        p = self.params
        if p.useCats:
            gt = self._gts[imgId, catId]
            dt = self._dts[imgId, catId]
        else:
            gt = [_ for cId in p.catIds for _ in self._gts[imgId, cId]]
            dt = [_ for cId in p.catIds for _ in self._dts[imgId, cId]]
        if len(gt) == 0 and len(dt) == 0:
            return []
        inds = np.argsort([-d["score"] for d in dt], kind="mergesort")
        dt = [dt[i] for i in inds]
        if len(dt) > p.maxDets[-1]:
            dt = dt[0 : p.maxDets[-1]]

        if p.iouType == "segm":
            g = [g["segmentation"] for g in gt if g["segmentation"] is not None]
            d = [d["segmentation"] for d in dt if d["segmentation"] is not None]
        elif p.iouType == "bbox":
            g = [g["bbox"] for g in gt]
            d = [d["bbox"] for d in dt]
        else:
            raise Exception("unknown iouType for iou computation")

        # compute iou between each dt and gt region
        iscrowd = [int(o.get("iscrowd", 0)) for o in gt]
        ious = maskUtils.iou(d, g, iscrowd)
        return ious

    def computeOks(self, imgId, catId):
        p = self.params
        # dimension here should be Nxm
        gts = self._gts[imgId, catId]
        dts = self._dts[imgId, catId]
        inds = np.argsort([-d["score"] for d in dts], kind="mergesort")
        dts = [dts[i] for i in inds]
        if len(dts) > p.maxDets[-1]:
            dts = dts[0 : p.maxDets[-1]]
        # if len(gts) == 0 and len(dts) == 0:
        if len(gts) == 0 or len(dts) == 0:
            return []
        ious = np.zeros((len(dts), len(gts)))
        sigmas = (
            np.array(
                [
                    0.26,
                    0.25,
                    0.25,
                    0.35,
                    0.35,
                    0.79,
                    0.79,
                    0.72,
                    0.72,
                    0.62,
                    0.62,
                    1.07,
                    1.07,
                    0.87,
                    0.87,
                    0.89,
                    0.89,
                ]
            )
            / 10.0
        )
        vars = (sigmas * 2) ** 2
        k = len(sigmas)
        # compute oks between each detection and ground truth object
        for j, gt in enumerate(gts):
            # create bounds for ignore regions(double the gt bbox)
            g = np.array(gt["keypoints"])
            xg = g[0::3]
            yg = g[1::3]
            vg = g[2::3]
            k1 = np.count_nonzero(vg > 0)
            bb = gt["bbox"]
            x0 = bb[0] - bb[2]
            x1 = bb[0] + bb[2] * 2
            y0 = bb[1] - bb[3]
            y1 = bb[1] + bb[3] * 2
            for i, dt in enumerate(dts):
                d = np.array(dt["keypoints"])
                xd = d[0::3]
                yd = d[1::3]
                if k1 > 0:
                    # measure the per-keypoint distance if keypoints visible
                    dx = xd - xg
                    dy = yd - yg
                else:
                    # measure minimum distance to keypoints in (x0,y0) & (x1,y1)
                    z = np.zeros(k)
                    dx = np.max((z, x0 - xd), axis=0) + np.max((z, xd - x1), axis=0)
                    dy = np.max((z, y0 - yd), axis=0) + np.max((z, yd - y1), axis=0)
                e = (dx**2 + dy**2) / vars / (gt["area"] + np.spacing(1)) / 2
                if k1 > 0:
                    e = e[vg > 0]
                ious[i, j] = np.sum(np.exp(-e)) / e.shape[0]
        return ious

    def _extract_mask(self, dt: Dict[str, Any]) -> np.ndarray:
        if "densepose" in dt:
            densepose_results_quantized = dt["densepose"]
            return densepose_results_quantized.labels_uv_uint8[0].numpy()
        elif "cse_mask" in dt:
            return dt["cse_mask"]
        elif "coarse_segm" in dt:
            dy = max(int(dt["bbox"][3]), 1)
            dx = max(int(dt["bbox"][2]), 1)
            return (
                F.interpolate(
                    dt["coarse_segm"].unsqueeze(0),
                    (dy, dx),
                    mode="bilinear",
                    align_corners=False,
                )
                .squeeze(0)
                .argmax(0)
                .numpy()
                .astype(np.uint8)
            )
        elif "record_id" in dt:
            assert (
                self.multi_storage is not None
            ), f"Storage record id encountered in a detection {dt}, but no storage provided!"
            record = self.multi_storage.get(dt["rank"], dt["record_id"])
            coarse_segm = record["coarse_segm"]
            dy = max(int(dt["bbox"][3]), 1)
            dx = max(int(dt["bbox"][2]), 1)
            return (
                F.interpolate(
                    coarse_segm.unsqueeze(0),
                    (dy, dx),
                    mode="bilinear",
                    align_corners=False,
                )
                .squeeze(0)
                .argmax(0)
                .numpy()
                .astype(np.uint8)
            )
        else:
            raise Exception(f"No mask data in the detection: {dt}")
        raise ValueError('The prediction dict needs to contain either "densepose" or "cse_mask"')

    def _extract_iuv(
        self, densepose_data: np.ndarray, py: np.ndarray, px: np.ndarray, gt: Dict[str, Any]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract arrays of I, U and V values at given points as numpy arrays
        given the data mode stored in self._dpDataMode
        """
        if self._dpDataMode == DensePoseDataMode.IUV_DT:
            # estimated labels and UV (default)
            ipoints = densepose_data[0, py, px]
            upoints = densepose_data[1, py, px] / 255.0  # convert from uint8 by /255.
            vpoints = densepose_data[2, py, px] / 255.0
        elif self._dpDataMode == DensePoseDataMode.IUV_GT:
            # ground truth
            ipoints = np.array(gt["dp_I"])
            upoints = np.array(gt["dp_U"])
            vpoints = np.array(gt["dp_V"])
        elif self._dpDataMode == DensePoseDataMode.I_GT_UV_0:
            # ground truth labels, UV = 0
            ipoints = np.array(gt["dp_I"])
            upoints = upoints * 0.0
            vpoints = vpoints * 0.0
        elif self._dpDataMode == DensePoseDataMode.I_GT_UV_DT:
            # ground truth labels, estimated UV
            ipoints = np.array(gt["dp_I"])
            upoints = densepose_data[1, py, px] / 255.0  # convert from uint8 by /255.
            vpoints = densepose_data[2, py, px] / 255.0
        elif self._dpDataMode == DensePoseDataMode.I_DT_UV_0:
            # estimated labels, UV = 0
            ipoints = densepose_data[0, py, px]
            upoints = upoints * 0.0
            vpoints = vpoints * 0.0
        else:
            raise ValueError(f"Unknown data mode: {self._dpDataMode}")
        return ipoints, upoints, vpoints

    def computeOgps_single_pair(self, dt, gt, py, px, pt_mask):
        if "densepose" in dt:
            ipoints, upoints, vpoints = self.extract_iuv_from_quantized(dt, gt, py, px, pt_mask)
            return self.computeOgps_single_pair_iuv(dt, gt, ipoints, upoints, vpoints)
        elif "u" in dt:
            ipoints, upoints, vpoints = self.extract_iuv_from_raw(dt, gt, py, px, pt_mask)
            return self.computeOgps_single_pair_iuv(dt, gt, ipoints, upoints, vpoints)
        elif "record_id" in dt:
            assert (
                self.multi_storage is not None
            ), f"Storage record id encountered in detection {dt}, but no storage provided!"
            record = self.multi_storage.get(dt["rank"], dt["record_id"])
            record["bbox"] = dt["bbox"]
            if "u" in record:
                ipoints, upoints, vpoints = self.extract_iuv_from_raw(record, gt, py, px, pt_mask)
                return self.computeOgps_single_pair_iuv(dt, gt, ipoints, upoints, vpoints)
            elif "embedding" in record:
                return self.computeOgps_single_pair_cse(
                    dt,
                    gt,
                    py,
                    px,
                    pt_mask,
                    record["coarse_segm"],
                    record["embedding"],
                    record["bbox"],
                )
            else:
                raise Exception(f"Unknown record format: {record}")
        elif "embedding" in dt:
            return self.computeOgps_single_pair_cse(
                dt, gt, py, px, pt_mask, dt["coarse_segm"], dt["embedding"], dt["bbox"]
            )
        raise Exception(f"Unknown detection format: {dt}")

    def extract_iuv_from_quantized(self, dt, gt, py, px, pt_mask):
        densepose_results_quantized = dt["densepose"]
        ipoints, upoints, vpoints = self._extract_iuv(
            densepose_results_quantized.labels_uv_uint8.numpy(), py, px, gt
        )
        ipoints[pt_mask == -1] = 0
        return ipoints, upoints, vpoints

    def extract_iuv_from_raw(self, dt, gt, py, px, pt_mask):
        labels_dt = resample_fine_and_coarse_segm_tensors_to_bbox(
            dt["fine_segm"].unsqueeze(0),
            dt["coarse_segm"].unsqueeze(0),
            dt["bbox"],
        )
        uv = resample_uv_tensors_to_bbox(
            dt["u"].unsqueeze(0), dt["v"].unsqueeze(0), labels_dt.squeeze(0), dt["bbox"]
        )
        labels_uv_uint8 = torch.cat((labels_dt.byte(), (uv * 255).clamp(0, 255).byte()))
        ipoints, upoints, vpoints = self._extract_iuv(labels_uv_uint8.numpy(), py, px, gt)
        ipoints[pt_mask == -1] = 0
        return ipoints, upoints, vpoints

    def computeOgps_single_pair_iuv(self, dt, gt, ipoints, upoints, vpoints):
        cVertsGT, ClosestVertsGTTransformed = self.findAllClosestVertsGT(gt)
        cVerts = self.findAllClosestVertsUV(upoints, vpoints, ipoints)
        # Get pairwise geodesic distances between gt and estimated mesh points.
        dist = self.getDistancesUV(ClosestVertsGTTransformed, cVerts)
        # Compute the Ogps measure.
        # Find the mean geodesic normalization distance for
        # each GT point, based on which part it is on.
        Current_Mean_Distances = self.Mean_Distances[
            self.CoarseParts[self.Part_ids[cVertsGT[cVertsGT > 0].astype(int) - 1]]
        ]
        return dist, Current_Mean_Distances

    def computeOgps_single_pair_cse(
        self, dt, gt, py, px, pt_mask, coarse_segm, embedding, bbox_xywh_abs
    ):
        # 0-based mesh vertex indices
        cVertsGT = torch.as_tensor(gt["dp_vertex"], dtype=torch.int64)
        # label for each pixel of the bbox, [H, W] tensor of long
        labels_dt = resample_coarse_segm_tensor_to_bbox(
            coarse_segm.unsqueeze(0), bbox_xywh_abs
        ).squeeze(0)
        x, y, w, h = bbox_xywh_abs
        # embedding for each pixel of the bbox, [D, H, W] tensor of float32
        embedding = F.interpolate(
            embedding.unsqueeze(0), (int(h), int(w)), mode="bilinear", align_corners=False
        ).squeeze(0)
        # valid locations py, px
        py_pt = torch.from_numpy(py[pt_mask > -1])
        px_pt = torch.from_numpy(px[pt_mask > -1])
        cVerts = torch.ones_like(cVertsGT) * -1
        cVerts[pt_mask > -1] = self.findClosestVertsCse(
            embedding, py_pt, px_pt, labels_dt, gt["ref_model"]
        )
        # Get pairwise geodesic distances between gt and estimated mesh points.
        dist = self.getDistancesCse(cVertsGT, cVerts, gt["ref_model"])
        # normalize distances
        if (gt["ref_model"] == "smpl_27554") and ("dp_I" in gt):
            Current_Mean_Distances = self.Mean_Distances[
                self.CoarseParts[np.array(gt["dp_I"], dtype=int)]
            ]
        else:
            Current_Mean_Distances = 0.255
        return dist, Current_Mean_Distances

    def computeOgps(self, imgId, catId):
        p = self.params
        # dimension here should be Nxm
        g = self._gts[imgId, catId]
        d = self._dts[imgId, catId]
        inds = np.argsort([-d_["score"] for d_ in d], kind="mergesort")
        d = [d[i] for i in inds]
        if len(d) > p.maxDets[-1]:
            d = d[0 : p.maxDets[-1]]
        # if len(gts) == 0 and len(dts) == 0:
        if len(g) == 0 or len(d) == 0:
            return []
        ious = np.zeros((len(d), len(g)))
        # compute opgs between each detection and ground truth object
        # sigma = self.sigma #0.255 # dist = 0.3m corresponds to ogps = 0.5
        # 1 # dist = 0.3m corresponds to ogps = 0.96
        # 1.45 # dist = 1.7m (person height) corresponds to ogps = 0.5)
        for j, gt in enumerate(g):
            if not gt["ignore"]:
                g_ = gt["bbox"]
                for i, dt in enumerate(d):
                    #
                    dy = int(dt["bbox"][3])
                    dx = int(dt["bbox"][2])
                    dp_x = np.array(gt["dp_x"]) * g_[2] / 255.0
                    dp_y = np.array(gt["dp_y"]) * g_[3] / 255.0
                    py = (dp_y + g_[1] - dt["bbox"][1]).astype(int)
                    px = (dp_x + g_[0] - dt["bbox"][0]).astype(int)
                    #
                    pts = np.zeros(len(px))
                    pts[px >= dx] = -1
                    pts[py >= dy] = -1
                    pts[px < 0] = -1
                    pts[py < 0] = -1
                    if len(pts) < 1:
                        ogps = 0.0
                    elif np.max(pts) == -1:
                        ogps = 0.0
                    else:
                        px[pts == -1] = 0
                        py[pts == -1] = 0
                        dists_between_matches, dist_norm_coeffs = self.computeOgps_single_pair(
                            dt, gt, py, px, pts
                        )
                        # Compute gps
                        ogps_values = np.exp(
                            -(dists_between_matches**2) / (2 * (dist_norm_coeffs**2))
                        )
                        #
                        ogps = np.mean(ogps_values) if len(ogps_values) > 0 else 0.0
                    ious[i, j] = ogps

        gbb = [gt["bbox"] for gt in g]
        dbb = [dt["bbox"] for dt in d]

        # compute iou between each dt and gt region
        iscrowd = [int(o.get("iscrowd", 0)) for o in g]
        ious_bb = maskUtils.iou(dbb, gbb, iscrowd)
        return ious, ious_bb

    def evaluateImg(self, imgId, catId, aRng, maxDet):
        """
        perform evaluation for single category and image
        :return: dict (single image results)
        """

        p = self.params
        if p.useCats:
            gt = self._gts[imgId, catId]
            dt = self._dts[imgId, catId]
        else:
            gt = [_ for cId in p.catIds for _ in self._gts[imgId, cId]]
            dt = [_ for cId in p.catIds for _ in self._dts[imgId, cId]]
        if len(gt) == 0 and len(dt) == 0:
            return None

        for g in gt:
            # g['_ignore'] = g['ignore']
            if g["ignore"] or (g["area"] < aRng[0] or g["area"] > aRng[1]):
                g["_ignore"] = True
            else:
                g["_ignore"] = False

        # sort dt highest score first, sort gt ignore last
        gtind = np.argsort([g["_ignore"] for g in gt], kind="mergesort")
        gt = [gt[i] for i in gtind]
        dtind = np.argsort([-d["score"] for d in dt], kind="mergesort")
        dt = [dt[i] for i in dtind[0:maxDet]]
        iscrowd = [int(o.get("iscrowd", 0)) for o in gt]
        # load computed ious
        if p.iouType == "densepose":
            # print('Checking the length', len(self.ious[imgId, catId]))
            # if len(self.ious[imgId, catId]) == 0:
            #    print(self.ious[imgId, catId])
            ious = (
                self.ious[imgId, catId][0][:, gtind]
                if len(self.ious[imgId, catId]) > 0
                else self.ious[imgId, catId]
            )
            ioubs = (
                self.ious[imgId, catId][1][:, gtind]
                if len(self.ious[imgId, catId]) > 0
                else self.ious[imgId, catId]
            )
            if self._dpEvalMode in {DensePoseEvalMode.GPSM, DensePoseEvalMode.IOU}:
                iousM = (
                    self.real_ious[imgId, catId][:, gtind]
                    if len(self.real_ious[imgId, catId]) > 0
                    else self.real_ious[imgId, catId]
                )
        else:
            ious = (
                self.ious[imgId, catId][:, gtind]
                if len(self.ious[imgId, catId]) > 0
                else self.ious[imgId, catId]
            )

        T = len(p.iouThrs)
        G = len(gt)
        D = len(dt)
        gtm = np.zeros((T, G))
        dtm = np.zeros((T, D))
        gtIg = np.array([g["_ignore"] for g in gt])
        dtIg = np.zeros((T, D))
        if np.all(gtIg) and p.iouType == "densepose":
            dtIg = np.logical_or(dtIg, True)

        if len(ious) > 0:  # and not p.iouType == 'densepose':
            for tind, t in enumerate(p.iouThrs):
                for dind, d in enumerate(dt):
                    # information about best match so far (m=-1 -> unmatched)
                    iou = min([t, 1 - 1e-10])
                    m = -1
                    for gind, _g in enumerate(gt):
                        # if this gt already matched, and not a crowd, continue
                        if gtm[tind, gind] > 0 and not iscrowd[gind]:
                            continue
                        # if dt matched to reg gt, and on ignore gt, stop
                        if m > -1 and gtIg[m] == 0 and gtIg[gind] == 1:
                            break
                        if p.iouType == "densepose":
                            if self._dpEvalMode == DensePoseEvalMode.GPSM:
                                new_iou = np.sqrt(iousM[dind, gind] * ious[dind, gind])
                            elif self._dpEvalMode == DensePoseEvalMode.IOU:
                                new_iou = iousM[dind, gind]
                            elif self._dpEvalMode == DensePoseEvalMode.GPS:
                                new_iou = ious[dind, gind]
                        else:
                            new_iou = ious[dind, gind]
                        if new_iou < iou:
                            continue
                        if new_iou == 0.0:
                            continue
                        # if match successful and best so far, store appropriately
                        iou = new_iou
                        m = gind
                    # if match made store id of match for both dt and gt
                    if m == -1:
                        continue
                    dtIg[tind, dind] = gtIg[m]
                    dtm[tind, dind] = gt[m]["id"]
                    gtm[tind, m] = d["id"]

        if p.iouType == "densepose":
            if not len(ioubs) == 0:
                for dind, d in enumerate(dt):
                    # information about best match so far (m=-1 -> unmatched)
                    if dtm[tind, dind] == 0:
                        ioub = 0.8
                        m = -1
                        for gind, _g in enumerate(gt):
                            # if this gt already matched, and not a crowd, continue
                            if gtm[tind, gind] > 0 and not iscrowd[gind]:
                                continue
                            # continue to next gt unless better match made
                            if ioubs[dind, gind] < ioub:
                                continue
                            # if match successful and best so far, store appropriately
                            ioub = ioubs[dind, gind]
                            m = gind
                            # if match made store id of match for both dt and gt
                        if m > -1:
                            dtIg[:, dind] = gtIg[m]
                            if gtIg[m]:
                                dtm[tind, dind] = gt[m]["id"]
                                gtm[tind, m] = d["id"]
        # set unmatched detections outside of area range to ignore
        a = np.array([d["area"] < aRng[0] or d["area"] > aRng[1] for d in dt]).reshape((1, len(dt)))
        dtIg = np.logical_or(dtIg, np.logical_and(dtm == 0, np.repeat(a, T, 0)))
        # store results for given image and category
        # print('Done with the function', len(self.ious[imgId, catId]))
        return {
            "image_id": imgId,
            "category_id": catId,
            "aRng": aRng,
            "maxDet": maxDet,
            "dtIds": [d["id"] for d in dt],
            "gtIds": [g["id"] for g in gt],
            "dtMatches": dtm,
            "gtMatches": gtm,
            "dtScores": [d["score"] for d in dt],
            "gtIgnore": gtIg,
            "dtIgnore": dtIg,
        }

    def accumulate(self, p=None):
        """
        Accumulate per image evaluation results and store the result in self.eval
        :param p: input params for evaluation
        :return: None
        """
        logger.info("Accumulating evaluation results...")
        tic = time.time()
        if not self.evalImgs:
            logger.info("Please run evaluate() first")
        # allows input customized parameters
        if p is None:
            p = self.params
        p.catIds = p.catIds if p.useCats == 1 else [-1]
        T = len(p.iouThrs)
        R = len(p.recThrs)
        K = len(p.catIds) if p.useCats else 1
        A = len(p.areaRng)
        M = len(p.maxDets)
        precision = -(np.ones((T, R, K, A, M)))  # -1 for the precision of absent categories
        recall = -(np.ones((T, K, A, M)))

        # create dictionary for future indexing
        logger.info("Categories: {}".format(p.catIds))
        _pe = self._paramsEval
        catIds = _pe.catIds if _pe.useCats else [-1]
        setK = set(catIds)
        setA = set(map(tuple, _pe.areaRng))
        setM = set(_pe.maxDets)
        setI = set(_pe.imgIds)
        # get inds to evaluate
        k_list = [n for n, k in enumerate(p.catIds) if k in setK]
        m_list = [m for n, m in enumerate(p.maxDets) if m in setM]
        a_list = [n for n, a in enumerate(map(lambda x: tuple(x), p.areaRng)) if a in setA]
        i_list = [n for n, i in enumerate(p.imgIds) if i in setI]
        I0 = len(_pe.imgIds)
        A0 = len(_pe.areaRng)
        # retrieve E at each category, area range, and max number of detections
        for k, k0 in enumerate(k_list):
            Nk = k0 * A0 * I0
            for a, a0 in enumerate(a_list):
                Na = a0 * I0
                for m, maxDet in enumerate(m_list):
                    E = [self.evalImgs[Nk + Na + i] for i in i_list]
                    E = [e for e in E if e is not None]
                    if len(E) == 0:
                        continue
                    dtScores = np.concatenate([e["dtScores"][0:maxDet] for e in E])

                    # different sorting method generates slightly different results.
                    # mergesort is used to be consistent as Matlab implementation.
                    inds = np.argsort(-dtScores, kind="mergesort")

                    dtm = np.concatenate([e["dtMatches"][:, 0:maxDet] for e in E], axis=1)[:, inds]
                    dtIg = np.concatenate([e["dtIgnore"][:, 0:maxDet] for e in E], axis=1)[:, inds]
                    gtIg = np.concatenate([e["gtIgnore"] for e in E])
                    npig = np.count_nonzero(gtIg == 0)
                    if npig == 0:
                        continue
                    tps = np.logical_and(dtm, np.logical_not(dtIg))
                    fps = np.logical_and(np.logical_not(dtm), np.logical_not(dtIg))
                    tp_sum = np.cumsum(tps, axis=1).astype(dtype=float)
                    fp_sum = np.cumsum(fps, axis=1).astype(dtype=float)
                    for t, (tp, fp) in enumerate(zip(tp_sum, fp_sum)):
                        tp = np.array(tp)
                        fp = np.array(fp)
                        nd = len(tp)
                        rc = tp / npig
                        pr = tp / (fp + tp + np.spacing(1))
                        q = np.zeros((R,))

                        if nd:
                            recall[t, k, a, m] = rc[-1]
                        else:
                            recall[t, k, a, m] = 0

                        # numpy is slow without cython optimization for accessing elements
                        # use python array gets significant speed improvement
                        pr = pr.tolist()
                        q = q.tolist()

                        for i in range(nd - 1, 0, -1):
                            if pr[i] > pr[i - 1]:
                                pr[i - 1] = pr[i]

                        inds = np.searchsorted(rc, p.recThrs, side="left")
                        try:
                            for ri, pi in enumerate(inds):
                                q[ri] = pr[pi]
                        except Exception:
                            pass
                        precision[t, :, k, a, m] = np.array(q)
        logger.info(
            "Final: max precision {}, min precision {}".format(np.max(precision), np.min(precision))
        )
        self.eval = {
            "params": p,
            "counts": [T, R, K, A, M],
            "date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "precision": precision,
            "recall": recall,
        }
        toc = time.time()
        logger.info("DONE (t={:0.2f}s).".format(toc - tic))

    def summarize(self):
        """
        Compute and display summary metrics for evaluation results.
        Note this function can *only* be applied on the default parameter setting
        """

        def _summarize(ap=1, iouThr=None, areaRng="all", maxDets=100):
            p = self.params
            iStr = " {:<18} {} @[ {}={:<9} | area={:>6s} | maxDets={:>3d} ] = {:0.3f}"
            titleStr = "Average Precision" if ap == 1 else "Average Recall"
            typeStr = "(AP)" if ap == 1 else "(AR)"
            measure = "IoU"
            if self.params.iouType == "keypoints":
                measure = "OKS"
            elif self.params.iouType == "densepose":
                measure = "OGPS"
            iouStr = (
                "{:0.2f}:{:0.2f}".format(p.iouThrs[0], p.iouThrs[-1])
                if iouThr is None
                else "{:0.2f}".format(iouThr)
            )

            aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == areaRng]
            mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]
            if ap == 1:
                # dimension of precision: [TxRxKxAxM]
                s = self.eval["precision"]
                # IoU
                if iouThr is not None:
                    t = np.where(np.abs(iouThr - p.iouThrs) < 0.001)[0]
                    s = s[t]
                s = s[:, :, :, aind, mind]
            else:
                # dimension of recall: [TxKxAxM]
                s = self.eval["recall"]
                if iouThr is not None:
                    t = np.where(np.abs(iouThr - p.iouThrs) < 0.001)[0]
                    s = s[t]
                s = s[:, :, aind, mind]
            if len(s[s > -1]) == 0:
                mean_s = -1
            else:
                mean_s = np.mean(s[s > -1])
            logger.info(iStr.format(titleStr, typeStr, measure, iouStr, areaRng, maxDets, mean_s))
            return mean_s

        def _summarizeDets():
            stats = np.zeros((12,))
            stats[0] = _summarize(1)
            stats[1] = _summarize(1, iouThr=0.5, maxDets=self.params.maxDets[2])
            stats[2] = _summarize(1, iouThr=0.75, maxDets=self.params.maxDets[2])
            stats[3] = _summarize(1, areaRng="small", maxDets=self.params.maxDets[2])
            stats[4] = _summarize(1, areaRng="medium", maxDets=self.params.maxDets[2])
            stats[5] = _summarize(1, areaRng="large", maxDets=self.params.maxDets[2])
            stats[6] = _summarize(0, maxDets=self.params.maxDets[0])
            stats[7] = _summarize(0, maxDets=self.params.maxDets[1])
            stats[8] = _summarize(0, maxDets=self.params.maxDets[2])
            stats[9] = _summarize(0, areaRng="small", maxDets=self.params.maxDets[2])
            stats[10] = _summarize(0, areaRng="medium", maxDets=self.params.maxDets[2])
            stats[11] = _summarize(0, areaRng="large", maxDets=self.params.maxDets[2])
            return stats

        def _summarizeKps():
            stats = np.zeros((10,))
            stats[0] = _summarize(1, maxDets=20)
            stats[1] = _summarize(1, maxDets=20, iouThr=0.5)
            stats[2] = _summarize(1, maxDets=20, iouThr=0.75)
            stats[3] = _summarize(1, maxDets=20, areaRng="medium")
            stats[4] = _summarize(1, maxDets=20, areaRng="large")
            stats[5] = _summarize(0, maxDets=20)
            stats[6] = _summarize(0, maxDets=20, iouThr=0.5)
            stats[7] = _summarize(0, maxDets=20, iouThr=0.75)
            stats[8] = _summarize(0, maxDets=20, areaRng="medium")
            stats[9] = _summarize(0, maxDets=20, areaRng="large")
            return stats

        def _summarizeUvs():
            stats = [_summarize(1, maxDets=self.params.maxDets[0])]
            min_threshold = self.params.iouThrs.min()
            if min_threshold <= 0.201:
                stats += [_summarize(1, maxDets=self.params.maxDets[0], iouThr=0.2)]
            if min_threshold <= 0.301:
                stats += [_summarize(1, maxDets=self.params.maxDets[0], iouThr=0.3)]
            if min_threshold <= 0.401:
                stats += [_summarize(1, maxDets=self.params.maxDets[0], iouThr=0.4)]
            stats += [
                _summarize(1, maxDets=self.params.maxDets[0], iouThr=0.5),
                _summarize(1, maxDets=self.params.maxDets[0], iouThr=0.75),
                _summarize(1, maxDets=self.params.maxDets[0], areaRng="medium"),
                _summarize(1, maxDets=self.params.maxDets[0], areaRng="large"),
                _summarize(0, maxDets=self.params.maxDets[0]),
                _summarize(0, maxDets=self.params.maxDets[0], iouThr=0.5),
                _summarize(0, maxDets=self.params.maxDets[0], iouThr=0.75),
                _summarize(0, maxDets=self.params.maxDets[0], areaRng="medium"),
                _summarize(0, maxDets=self.params.maxDets[0], areaRng="large"),
            ]
            return np.array(stats)

        def _summarizeUvsOld():
            stats = np.zeros((18,))
            stats[0] = _summarize(1, maxDets=self.params.maxDets[0])
            stats[1] = _summarize(1, maxDets=self.params.maxDets[0], iouThr=0.5)
            stats[2] = _summarize(1, maxDets=self.params.maxDets[0], iouThr=0.55)
            stats[3] = _summarize(1, maxDets=self.params.maxDets[0], iouThr=0.60)
            stats[4] = _summarize(1, maxDets=self.params.maxDets[0], iouThr=0.65)
            stats[5] = _summarize(1, maxDets=self.params.maxDets[0], iouThr=0.70)
            stats[6] = _summarize(1, maxDets=self.params.maxDets[0], iouThr=0.75)
            stats[7] = _summarize(1, maxDets=self.params.maxDets[0], iouThr=0.80)
            stats[8] = _summarize(1, maxDets=self.params.maxDets[0], iouThr=0.85)
            stats[9] = _summarize(1, maxDets=self.params.maxDets[0], iouThr=0.90)
            stats[10] = _summarize(1, maxDets=self.params.maxDets[0], iouThr=0.95)
            stats[11] = _summarize(1, maxDets=self.params.maxDets[0], areaRng="medium")
            stats[12] = _summarize(1, maxDets=self.params.maxDets[0], areaRng="large")
            stats[13] = _summarize(0, maxDets=self.params.maxDets[0])
            stats[14] = _summarize(0, maxDets=self.params.maxDets[0], iouThr=0.5)
            stats[15] = _summarize(0, maxDets=self.params.maxDets[0], iouThr=0.75)
            stats[16] = _summarize(0, maxDets=self.params.maxDets[0], areaRng="medium")
            stats[17] = _summarize(0, maxDets=self.params.maxDets[0], areaRng="large")
            return stats

        if not self.eval:
            raise Exception("Please run accumulate() first")
        iouType = self.params.iouType
        if iouType in ["segm", "bbox"]:
            summarize = _summarizeDets
        elif iouType in ["keypoints"]:
            summarize = _summarizeKps
        elif iouType in ["densepose"]:
            summarize = _summarizeUvs
        self.stats = summarize()

    def __str__(self):
        self.summarize()

    # ================ functions for dense pose ==============================
    def findAllClosestVertsUV(self, U_points, V_points, Index_points):
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

    def findClosestVertsCse(self, embedding, py, px, mask, mesh_name):
        mesh_vertex_embeddings = self.embedder(mesh_name)
        pixel_embeddings = embedding[:, py, px].t().to(device="cuda")
        mask_vals = mask[py, px]
        edm = squared_euclidean_distance_matrix(pixel_embeddings, mesh_vertex_embeddings)
        vertex_indices = edm.argmin(dim=1).cpu()
        vertex_indices[mask_vals <= 0] = -1
        return vertex_indices

    def findAllClosestVertsGT(self, gt):
        #
        I_gt = np.array(gt["dp_I"])
        U_gt = np.array(gt["dp_U"])
        V_gt = np.array(gt["dp_V"])
        #
        # print(I_gt)
        #
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

    def getDistancesCse(self, cVertsGT, cVerts, mesh_name):
        geodists_vertices = torch.ones_like(cVertsGT) * float("inf")
        selected = (cVertsGT >= 0) * (cVerts >= 0)
        mesh = create_mesh(mesh_name, "cpu")
        geodists_vertices[selected] = mesh.geodists[cVertsGT[selected], cVerts[selected]]
        return geodists_vertices.numpy()

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


class Params:
    """
    Params for coco evaluation api
    """

    def setDetParams(self):
        self.imgIds = []
        self.catIds = []
        # np.arange causes trouble.  the data point on arange is slightly larger than the true value
        self.iouThrs = np.linspace(0.5, 0.95, int(np.round((0.95 - 0.5) / 0.05)) + 1, endpoint=True)
        self.recThrs = np.linspace(0.0, 1.00, int(np.round((1.00 - 0.0) / 0.01)) + 1, endpoint=True)
        self.maxDets = [1, 10, 100]
        self.areaRng = [
            [0**2, 1e5**2],
            [0**2, 32**2],
            [32**2, 96**2],
            [96**2, 1e5**2],
        ]
        self.areaRngLbl = ["all", "small", "medium", "large"]
        self.useCats = 1

    def setKpParams(self):
        self.imgIds = []
        self.catIds = []
        # np.arange causes trouble.  the data point on arange is slightly larger than the true value
        self.iouThrs = np.linspace(0.5, 0.95, np.round((0.95 - 0.5) / 0.05) + 1, endpoint=True)
        self.recThrs = np.linspace(0.0, 1.00, np.round((1.00 - 0.0) / 0.01) + 1, endpoint=True)
        self.maxDets = [20]
        self.areaRng = [[0**2, 1e5**2], [32**2, 96**2], [96**2, 1e5**2]]
        self.areaRngLbl = ["all", "medium", "large"]
        self.useCats = 1

    def setUvParams(self):
        self.imgIds = []
        self.catIds = []
        self.iouThrs = np.linspace(0.5, 0.95, int(np.round((0.95 - 0.5) / 0.05)) + 1, endpoint=True)
        self.recThrs = np.linspace(0.0, 1.00, int(np.round((1.00 - 0.0) / 0.01)) + 1, endpoint=True)
        self.maxDets = [20]
        self.areaRng = [[0**2, 1e5**2], [32**2, 96**2], [96**2, 1e5**2]]
        self.areaRngLbl = ["all", "medium", "large"]
        self.useCats = 1

    def __init__(self, iouType="segm"):
        if iouType == "segm" or iouType == "bbox":
            self.setDetParams()
        elif iouType == "keypoints":
            self.setKpParams()
        elif iouType == "densepose":
            self.setUvParams()
        else:
            raise Exception("iouType not supported")
        self.iouType = iouType
        # useSegm is deprecated
        self.useSegm = None
