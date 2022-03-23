import numpy as np
import pickle
import scipy.spatial.distance as ssd
from scipy.io import loadmat
import numpy as np
from scipy.ndimage import zoom as spzoom
from pycocotools import mask as maskUtils
from torch.nn import functional as F
import torch
from detectron2.structures import Instances
from typing import Any, List

from detectron2.utils.file_io import PathManager

class DenseScoreLossHelper:
    def __init__(self):
        self._loadGEval()
    
    """
    Same as with the same name function in densepose_coco_evalution.py
    """
    def _loadGEval(self):
        smpl_subdiv_fpath = PathManager.get_local_path(
            "https://dl.fbaipublicfiles.com/densepose/data/SMPL_subdiv.mat"
        )
        pdist_transform_fpath = PathManager.get_local_path(
            "https://dl.fbaipublicfiles.com/densepose/data/SMPL_SUBDIV_TRANSFORM.mat"
        )
        pdist_matrix_fpath = PathManager.get_local_path(
            "https://dl.fbaipublicfiles.com/densepose/data/Pdist_matrix.pkl"
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

    
    def computeOgps(self, u_gt, v_gt, i_gt, u_dt, v_dt, i_dt):
        """
        preprocess the uv coordinate
        """
        u_dt = (u_dt * 255).clamp(0, 255) / 255.0
        v_dt = (v_dt * 255).clamp(0, 255) / 255.0

        cVertsGT, ClosestVertsGTTransformed = self.findAllClosestVertsGT(u_gt, v_gt, i_gt)
        cVerts = self.findAllClosestVertsUV(u_dt, v_dt, i_dt)
        # Get pairwise geodesic distances between gt and estimated mesh points.
        dist = self.getDistancesUV(ClosestVertsGTTransformed, cVerts)
        # Compute the Ogps measure.
        # Find the mean geodesic normalization distance for
        # each GT point, based on which part it is on.
        Current_Mean_Distances = self.Mean_Distances[
            self.CoarseParts[self.Part_ids[cVertsGT[cVertsGT > 0].astype(int) - 1]]
        ]
        
        """
        Comput gps
        """
        return np.exp(
            -(dist ** 2) / (2 * (Current_Mean_Distances ** 2))
        )

    
    def findAllClosestVertsGT(self, u_gt, v_gt, i_gt):
        #
        I_gt = i_gt.cpu().detach().numpy()
        U_gt = u_gt.cpu().detach().numpy()
        V_gt = v_gt.cpu().detach().numpy()
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
    

    def findAllClosestVertsUV(self, u_dt, v_dt, i_dt):
        #
        I_dt = i_dt.cpu().detach().numpy()
        U_dt = u_dt.cpu().detach().numpy()
        V_dt = v_dt.cpu().detach().numpy()


        ClosestVerts = np.ones(I_dt.shape) * -1
        for i in np.arange(24):
            #
            if (i + 1) in I_dt:
                UVs = np.array(
                    [U_dt[I_dt == (i + 1)], V_dt[I_dt == (i + 1)]]
                )
                Current_Part_UVs = self.Part_UVs[i]
                Current_Part_ClosestVertInds = self.Part_ClosestVertInds[i]
                D = ssd.cdist(Current_Part_UVs.transpose(), UVs.transpose()).squeeze()
                ClosestVerts[I_dt == (i + 1)] = Current_Part_ClosestVertInds[
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


    """
    mIOU Computing
    """
    def getmIOU(self,
        proposals_with_gt: List[Instances],
        densepose_predictor_outputs: Any,
        packed_annotations: Any,):

        if packed_annotations.coarse_segm_gt is None:
            return self.fake_value(densepose_predictor_outputs)

        image_h = packed_annotations.image_size[0].detach().cpu().numpy()
        image_w = packed_annotations.image_size[1].detach().cpu().numpy()
        
        if 0 in image_h or 0 in image_w:
            print("[ERR]")

        mask_gt = []

        """
        gt mask
        """
        coarse_segm_gt = np.minimum(packed_annotations.coarse_segm_gt.detach().cpu().numpy(), 1.0)

        N = len(coarse_segm_gt)

        bbox_xywh_gt = packed_annotations.bbox_xywh_gt
        bbox_xywh_gt = bbox_xywh_gt.detach().cpu().numpy()
        is_crowd = np.zeros(N)

        for i in np.arange(N):
            scale_x = float(max(bbox_xywh_gt[i, 2], 1)) / coarse_segm_gt.shape[2]
            scale_y = float(max(bbox_xywh_gt[i, 3], 1)) / coarse_segm_gt.shape[1]
            mask = spzoom(coarse_segm_gt[i], (scale_y, scale_x), order=1, prefilter=False)
            mask = np.array(mask > 0.5, dtype=np.uint8)
            mask_gt.append(self._generate_rlemask_on_image(mask, image_h[i], image_w[i], bbox_xywh_gt[i]))

        """
        dt mask
        """
        coarse_segm_est = densepose_predictor_outputs.coarse_segm[packed_annotations.bbox_indices]
        fine_segm_est = densepose_predictor_outputs.fine_segm[packed_annotations.bbox_indices]
        bbox_xywh_est = packed_annotations.bbox_xywh_est

        N = len(coarse_segm_est)

        mask_dt = []

        for i in np.arange(N):
            x, y, w, h = bbox_xywh_est[i]
            img_h, img_w = int(image_h[i]), int(image_w[i])
            x, y = int(x), int(y)
            w = min(int(w), img_w - x)
            h = min(int(h), img_h - y)

            # coarse segmentation
            coarse_segm_bbox = F.interpolate(
                coarse_segm_est[i].unsqueeze(0), (h, w), mode="bilinear", align_corners=False
            ).argmax(dim=1)
            # combined coarse and fine segmentation
            labels = (
                F.interpolate(fine_segm_est[i].unsqueeze(0), (h, w), mode="bilinear", align_corners=False).argmax(dim=1)
                * (coarse_segm_bbox > 0).long()
            )[0]
            
            # mask = torch.zeros((img_h, img_w), dtype=torch.bool, device=labels.device)
            # mask[y : y + h, x : x + w] = labels > 0
            mask = labels > 0

            rle_mask = self._generate_rlemask_on_image(mask.detach().cpu().numpy(), image_h[i], image_w[i], bbox_xywh_est[i])
            mask_dt.append(rle_mask)

        iousDP = maskUtils.iou(mask_dt, mask_gt, is_crowd)
        return np.max(iousDP, axis=1)


    def _generate_rlemask_on_image(self, mask, im_h, im_w, bbox_xywh):
        x, y, w, h = bbox_xywh
        im_h, im_w = int(im_h), int(im_w)
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

    def fake_value(self, densepose_predictor_outputs: Any) -> torch.Tensor:
        """
        Fake segmentation loss used when no suitable ground truth data
        was found in a batch. The loss has a value 0 and is primarily used to
        construct the computation graph, so that `DistributedDataParallel`
        has similar graphs on all GPUs and can perform reduction properly.

        Args:
            densepose_predictor_outputs: DensePose predictor outputs, an object
                of a dataclass that is assumed to have `coarse_segm`
                attribute
        Return:
            Zero value loss with proper computation graph
        """
        return densepose_predictor_outputs.coarse_segm.sum() * 0