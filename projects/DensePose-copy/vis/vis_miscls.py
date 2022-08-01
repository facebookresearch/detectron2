import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D 
import numpy as np
import pickle
import matplotlib.image as mpimg

import sys
sys.path.append(r'/home/sunjunyao/Model/DensePose/detectron/utils')
import densepose_methods as dp_utils
import random
import cv2 as cv

from mpl_toolkits.mplot3d.proj3d import proj_transform
from matplotlib.text import Annotation
from mpl_toolkits.mplot3d import axes3d
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from pycocotools.coco import COCO

import copy
import datetime
import logging
from collections import defaultdict
from enum import Enum
from typing import Any, Dict, Tuple
import scipy.spatial.distance as ssd
from pycocotools import mask as maskUtils
from scipy.io import loadmat
from scipy.ndimage import zoom as spzoom

## METHOD VIS
# SMPL MODEL VIS
# Now read the smpl model.
with open('../pretrained_model/basicmodel_m_lbs_10_207_0_v1.0.0.pkl', 'rb') as f:
    data = pickle.load(f)
    Vertices = data['v_template']  ##  Loaded vertices of size (6890, 3)
    X,Y,Z = [Vertices[:,0], Vertices[:,1],Vertices[:,2]]
    
def smpl_view_set_axis_full_body(ax,azimuth=0):
    ## Manually set axis 
    ax.view_init(0, azimuth)
    max_range = 0.55
    ax.set_xlim( - max_range,   max_range)
    ax.set_ylim( - max_range,   max_range)
    ax.set_zlim( -0.2 - max_range,   -0.2 + max_range)
    ax.axis('off')
    
# def smpl_view_set_axis_face(ax, azimuth=0):
#     ## Manually set axis 
#     ax.view_init(0, azimuth)
#     max_range = 0.1
#     ax.set_xlim( - max_range,   max_range)
#     ax.set_ylim( - max_range,   max_range)
#     ax.set_zlim( 0.45 - max_range,   0.45 + max_range)
#     ax.axis('off')

class Annotation3D(Annotation):
    '''Annotate the point xyz with text s'''

    def __init__(self, s, xyz, *args, **kwargs):
        Annotation.__init__(self,s, xy=(0,0), *args, **kwargs)
        self._verts3d = xyz        

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.xy=(xs,ys)
        Annotation.draw(self, renderer)
def annotate3D(ax, s, *args, **kwargs):
    '''add anotation text s to to Axes3d ax'''

    tag = Annotation3D(s, *args, **kwargs)
    ax.add_artist(tag)

def findAllClosestVerts(I_gt, U_gt, V_gt, U_points, V_points, Index_points):
        #
    # I_gt = np.array(gt["dp_I"])
    # U_gt = np.array(gt["dp_U"])
    # V_gt = np.array(gt["dp_V"])
        #
        # print(I_gt)
        #
    ClosestVerts = np.ones(Index_points.shape) * -1
    for i in np.arange(24):
            #
        if (i + 1) in Index_points:
            UVs = np.array(
                [U_points[Index_points == (i + 1)], V_points[Index_points == (i + 1)]]
            )
            Current_Part_UVs = Part_UVs[i]
            Current_Part_ClosestVertInds = Part_ClosestVertInds[i]
            D = ssd.cdist(Current_Part_UVs.transpose(), UVs.transpose()).squeeze()
            ClosestVerts[Index_points == (i + 1)] = Current_Part_ClosestVertInds[
                np.argmin(D, axis=0)
            ]
        #
    ClosestVertsGT = np.ones(Index_points.shape) * -1
    for i in np.arange(24):
        if (i + 1) in I_gt:
            UVs = np.array([U_gt[I_gt == (i + 1)], V_gt[I_gt == (i + 1)]])
            Current_Part_UVs = Part_UVs[i]
            Current_Part_ClosestVertInds = Part_ClosestVertInds[i]
            D = ssd.cdist(Current_Part_UVs.transpose(), UVs.transpose()).squeeze()
            ClosestVertsGT[I_gt == (i + 1)] = Current_Part_ClosestVertInds[np.argmin(D, axis=0)]
        #
    return ClosestVerts, ClosestVertsGT

def getDistances(cVertsGT, cVerts):

    ClosestVertsTransformed = PDIST_transform[cVerts.astype(int) - 1]
    ClosestVertsGTTransformed = PDIST_transform[cVertsGT.astype(int) - 1]
        #
    ClosestVertsTransformed[cVerts < 0] = 0
    ClosestVertsGTTransformed[cVertsGT < 0] = 0
        #
    cVertsGT = ClosestVertsGTTransformed
    cVerts = ClosestVertsTransformed
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
                    dists.append(Pdist_matrix[int(k)][0])
                else:
                    i = n - i - 1
                    j = n - j - 1
                    k = (n * (n - 1) / 2) - (n - i) * ((n - i) - 1) / 2 + j - i - 1
                    k = (n * n - n) / 2 - k - 1
                    dists.append(Pdist_matrix[int(k)][0])
            else:
                dists.append(3.)
    return np.atleast_1d(np.array(dists).squeeze())

smpl_subdiv_fpath = '/home/sunjunyao/Model/DensePose/DensePoseData/eval_data/SMPL_subdiv.mat'
    
pdist_transform_fpath = '/home/sunjunyao/Model/DensePose/DensePoseData/eval_data/SMPL_SUBDIV_TRANSFORM.mat'
pdist_matrix_fpath = '/home/sunjunyao/Model/DensePose/DensePoseData/eval_data/Pdist_matrix.pkl'
SMPL_subdiv = loadmat(smpl_subdiv_fpath)
PDIST_transform = loadmat(pdist_transform_fpath)
PDIST_transform = PDIST_transform["index"].squeeze()
UV = np.array([SMPL_subdiv["U_subdiv"], SMPL_subdiv["V_subdiv"]]).squeeze()
ClosestVertInds = np.arange(UV.shape[1]) + 1
Part_UVs = []
Part_ClosestVertInds = []
for i in np.arange(24):
    Part_UVs.append(UV[:, SMPL_subdiv["Part_ID_subdiv"].squeeze() == (i + 1)])
    Part_ClosestVertInds.append(
        ClosestVertInds[SMPL_subdiv["Part_ID_subdiv"].squeeze() == (i + 1)]
    )

with open(pdist_matrix_fpath, "rb") as hFile:
    arrays = pickle.load(hFile)
Pdist_matrix = arrays["Pdist_matrix"]
Part_ids = np.array(SMPL_subdiv["Part_ID_subdiv"].squeeze())


# PREDICTION-SMPL-GT POINT VIS
DP = dp_utils.DensePoseMethods()
# pkl_file = open('DensePoseData/demo_data/no.json', 'rb')
coco = COCO("DensePoseData/demo_data/no.json")
image = cv.imread('DensePoseData/demo_data/65736/COCO_val2014_000000065736.jpg')
# Demo = pickle.load(pkl_file)
ann_ids = coco.getAnnIds(65736)
anns = coco.loadAnns(ann_ids)[0]

bbr = np.round(anns['bbox'])
x1,y1,x2,y2 = bbr[0],bbr[1],bbr[0]+bbr[2],bbr[1]+bbr[3]
# x2 = min( [ x2,image.shape[1] ] ); y2 = min( [ y2,image.shape[0] ] )

collected_x = np.zeros(np.array(anns['dp_x']).shape)
collected_y = np.zeros(np.array(anns['dp_x']).shape)
collected_z = np.zeros(np.array(anns['dp_x']).shape)

# collected_x = np.zeros(Demo['x'].shape)
# collected_y = np.zeros(Demo['x'].shape)
# collected_z = np.zeros(Demo['x'].shape)

# for i, (ii,uu,vv) in enumerate(zip(Demo['I'],Demo['U'],Demo['V'])):
for i, (ii,uu,vv) in enumerate(zip(np.array(anns['dp_I']),np.array(anns['dp_U']),np.array(anns['dp_V']))):
    # Convert IUV to FBC (faceIndex and barycentric coordinates.)
    FaceIndex,bc1,bc2,bc3 = DP.IUV2FBC(ii,uu,vv)
    # Use FBC to get 3D coordinates on the surface.
    p = DP.FBC2PointOnSurface( FaceIndex, bc1,bc2,bc3,Vertices )
    #
    collected_x[i] = p[0]
    collected_y[i] = p[1]
    collected_z[i] = p[2]
#fig = plt.figure(figsize=[15,5])

# #index = random.sample(np.arange(len(Demo['y'])),2)
# x1 = np.expand_dims(np.array(Demo['x'][4]), axis=0)
# y1 = np.expand_dims(np.array(Demo['y'][4]), axis=0)
# x2 = np.expand_dims(np.array(Demo['x'][4]), axis=0)
# y2 = np.expand_dims(np.array(Demo['y'][4]), axis=0)
# cz = collected_z[4:6]
# cx = collected_x[4:6]
# cy = collected_y[4:6]
# Visualize the image and collected points.
# ax = fig.add_subplot(133)
# ax.imshow(Demo['ICrop'])
# ax.scatter(x1,y1,25, np.arange(1)  )
# ax.axis('off')

# # Visualize the full body smpl male template model and collected points
# ax = fig.add_subplot(132, projection='3d')
# ax.scatter(Z,X,Y,s=0.02,c='k')
# ax.scatter(cz, cx, cy,s=25,  c=  ['r', 'b']    )
# smpl_view_set_axis_full_body(ax)


# im_iuv = mpimg.imread('DensePoseData/demo_data/image_densepose_u.png') 
# ax = fig.add_subplot(131)
# ax.imshow(im_iuv)
# ax.scatter(x2,y2,25, np.arange(1) )
# ax.axis('off'), 
# plt.show()

## RESULT VIS
# UV MAP WITH POINT(PREDICTION AND GT) VIS

Tex_Atlas = cv.imread('DensePoseData/demo_data/texture_atlas_200.png')[:,:,::-1]
# i_gt = Demo["I"]
# u_gt = Demo["U"]
# v_gt = Demo["V"]
i_gt = np.array(anns['dp_I'])
u_gt = np.array(anns['dp_U'])
v_gt = np.array(anns['dp_V'])

part_m = i_gt - (((i_gt-1) // 6)*6+1)
part_n = (i_gt-1) // 6
# print(part_m.min(), part_m.max(), part_n.min(), part_n.max())
v_cor = 200*part_m+1 + ((199.0)/(1.-0.))*(v_gt-0.) 
u_cor = 200*part_n+1 + ((199.0)/(1.-0.))*(u_gt-0.) 
# print(v_cor, u_cor)

u_pre_im_base = cv.imread('DensePoseData/demo_data/65736/image_densepose_u.png',cv.IMREAD_UNCHANGED)/255.
v_pre_im_base = cv.imread('DensePoseData/demo_data/65736/image_densepose_v.png',cv.IMREAD_UNCHANGED)/255.
i_pre_im_base = cv.imread('DensePoseData/demo_data/65736/image_densepose_i.png',cv.IMREAD_UNCHANGED)

u_pre_im_mmcl = cv.imread('DensePoseData/demo_data/65736/image_densepose_u.0001.png',cv.IMREAD_UNCHANGED)/255.
v_pre_im_mmcl = cv.imread('DensePoseData/demo_data/65736/image_densepose_v.0001.png',cv.IMREAD_UNCHANGED)/255.
i_pre_im_mmcl = cv.imread('DensePoseData/demo_data/65736/image_densepose_i.0001.png',cv.IMREAD_UNCHANGED)
# print(u_pre_im.shape, v_pre_im.shape, i_pre_im.shape, Demo['ICrop'].shape)
# print(Demo["x"].max(), Demo["y"].max())
# print(i_pre_im.max(), i_pre_im.min())
# x = Demo["x"].astype(int)
# y = Demo["y"].astype(int)

x = (np.array(anns["dp_x"])/255. * bbr[2] +x1).astype(int)
y = (np.array(anns["dp_y"])/255. * bbr[3] +y1).astype(int)

# ax = fig.add_subplot(141)
# # ax.imshow(Demo['ICrop'])
# ax.imshow(image[:,:,::-1])
# ax.scatter(x,y,s=100, c=np.arange(len(y)), marker='.')
# ax.axis('off')


u_pred_base = u_pre_im_base[y,x]
v_pred_base = v_pre_im_base[y,x]
i_pred_base = i_pre_im_base[y,x]

u_pred_mmcl = u_pre_im_mmcl[y,x]
v_pred_mmcl = v_pre_im_mmcl[y,x]
i_pred_mmcl = i_pre_im_mmcl[y,x]

u_base_miscls = u_pred_base[i_pred_mmcl != i_gt]
v_base_miscls = v_pred_base[i_pred_mmcl != i_gt]
i_base_miscls = i_pred_base[i_pred_mmcl != i_gt]

u_mmcl_miscls = u_pred_mmcl[i_pred_mmcl != i_gt]
v_mmcl_miscls = v_pred_mmcl[i_pred_mmcl != i_gt]
i_mmcl_miscls = i_pred_mmcl[i_pred_mmcl != i_gt]

i_gt_base_miscls = i_gt[i_pred_mmcl != i_gt]
u_gt_base_miscls = u_gt[i_pred_mmcl != i_gt]
v_gt_base_miscls = v_gt[i_pred_mmcl != i_gt]

i_gt_mmcl_miscls = i_gt[i_pred_mmcl != i_gt]
u_gt_mmcl_miscls = u_gt[i_pred_mmcl != i_gt]
v_gt_mmcl_miscls = v_gt[i_pred_mmcl != i_gt]

cVerts_base, cVertsGT_base = findAllClosestVerts(i_gt_base_miscls, u_gt_base_miscls, v_gt_base_miscls, u_base_miscls, v_base_miscls, i_base_miscls)
dist_base = getDistances(cVertsGT_base, cVerts_base)
cVerts_mmcl, cVertsGT_mmcl = findAllClosestVerts(i_gt_mmcl_miscls, u_gt_mmcl_miscls, v_gt_mmcl_miscls, u_mmcl_miscls, v_mmcl_miscls, i_mmcl_miscls)
dist_mmcl = getDistances(cVertsGT_mmcl, cVerts_mmcl)
print(i_gt_base_miscls,i_base_miscls,i_mmcl_miscls)
print(np.mean(dist_base), np.mean(dist_mmcl))
print(dist_base)
print(dist_mmcl)



