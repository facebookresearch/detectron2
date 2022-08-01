import matplotlib.pyplot as plt
from matplotlib import cm
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
from scipy.io import loadmat

from pycocotools.coco import COCO

## METHOD VIS
# SMPL MODEL VIS
# Now read the smpl model.
# with open('../pretrained_model/basicmodel_m_lbs_10_207_0_v1.0.0.pkl', 'rb') as f:
#     data = pickle.load(f)
#     Vertices = data['v_template']  ##  Loaded vertices of size (6890, 3)
    # X,Y,Z = [Vertices[:,0], Vertices[:,1],Vertices[:,2]]
smpl_subdiv_fpath = '/home/sunjunyao/Model/DensePose/DensePoseData/eval_data/SMPL_subdiv.mat'
num_charts = np.array([174026,521204,231835,215765,129672,127442,68736,70068,222330,220418,60061,66011,161345,166318,142698,150036,175552,184405,86666,109669,203377,192827,260668,245133], dtype=np.float_)
Nor_charts = ((num_charts-min(num_charts))/(max(num_charts)-min(num_charts)))*(1-0.1) +0.1
# print(Nor_charts)
SMPL_subdiv = loadmat(smpl_subdiv_fpath)
Vertices = SMPL_subdiv["vertex"]
X,Y,Z = [Vertices[0], Vertices[1],Vertices[2]]
Part_ids = np.array(SMPL_subdiv["Part_ID_subdiv"].squeeze())
color = Nor_charts[Part_ids-1]


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
def getXYZNfromIUV(I, U, V):
    X = np.zeros(np.array(len(I)))
    Y = np.zeros(np.array(len(I)))
    Z = np.zeros(np.array(len(I)))
    N = np.zeros(np.array(len(I)))
    bg = []
    index_vertexes = []
    for i, (ii,uu,vv) in enumerate(zip(I,U,V)):
    # Convert IUV to FBC (faceIndex and barycentric coordinates.)
        if ii == 0:
            bg.append(i)
            continue
        FaceIndex,bc1,bc2,bc3 = DP.IUV2FBC(ii,uu,vv)
        p, indver = DP.FBC2PointOnSurface( FaceIndex, bc1,bc2,bc3,Vertices )
        if indver not in index_vertexes:
            index_vertexes.extend(indver)
            X[i] = p[0]
            Y[i] = p[1]
            Z[i] = p[2]
            N[i] = num_charts[int(ii)-1]
    return X,Y,Z,N

fig = plt.figure(figsize=[8,16])
ax = fig.gca(projection="3d")
ax.scatter(Z,X,Y, c=color, s=5, cmap=plt.get_cmap('autumn_r'), marker="o", linewidths=0, edgecolors="none")
smpl_view_set_axis_full_body(ax)
plt.show()



# DP = dp_utils.DensePoseMethods()
# coco = COCO("DensePoseData/demo_data/no.json")
# image_ids = coco.getImgIds()
# # image_ids = 188592
# num_charts = np.array([174026,521204,231835,215765,129672,127442,68736,70068,222330,220418,60061,66011,161345,166318,142698,150036,175552,184405,86666,109669,203377,192827,260668,245133])
# I = []
# U = []
# V = []
# ann_ids = coco.getAnnIds(image_ids)
# anns = coco.loadAnns(ann_ids)
# im = coco.loadImgs(image_ids)[0]
# # for ann in anns:
# #     if 'dp_masks' in ann.keys():
# #         dp_I = ann['dp_I']
# #         dp_U = ann['dp_U']
# #         dp_V = ann['dp_V']
# #         I.extend(dp_I)
# #         U.extend(dp_U)
# #         V.extend(dp_V)
# # I = np.array(I)
# # U = np.array(U)
# # V = np.array(V)
# # X,Y,Z,N = getXYZNfromIUV(I, U, V)

# for id in image_ids:
#     ann_ids = coco.getAnnIds(id)
#     anns = coco.loadAnns(ann_ids)
#     im = coco.loadImgs(id)[0]
#     for ann in anns:
#         if 'dp_masks' in ann.keys():
#             dp_I = ann['dp_I']
#             dp_U = ann['dp_U']
#             dp_V = ann['dp_V']
#             I.extend(dp_I)
#             U.extend(dp_U)
#             V.extend(dp_V)
# I = np.array(I)
# U = np.array(U)
# V = np.array(V)
# X,Y,Z,N = getXYZNfromIUV(I, U, V)
# fig = plt.figure(figsize=[8,16])
# ax = fig.gca(projection="3d")
# ax.scatter(Z,X,Y, c=N, s=1, cmap=plt.get_cmap('hot_r'), marker="o", linewidths=0, edgecolors="none")
# smpl_view_set_axis_full_body(ax)
# plt.show()
            
            





## Now let's rotate around the model and zoom into the face.

# fig = plt.figure([16,4])

# ax = fig.add_subplot(141, projection='3d')
# ax.scatter(Z,X,Y,s=0.02,c='k')
# smpl_view_set_axis_full_body(ax)
#
#ax = fig.add_subplot(142, projection='3d')
#ax.scatter(Z,X,Y,s=0.02,c='k')
#smpl_view_set_axis_full_body(ax,45)
#
#ax = fig.add_subplot(143, projection='3d')
#ax.scatter(Z,X,Y,s=0.02,c='k')
#smpl_view_set_axis_full_body(ax,90)
#
#ax = fig.add_subplot(144, projection='3d')
#ax.scatter(Z,X,Y,s=0.2,c='k')
#smpl_view_set_axis_face(ax,-40)
#
#plt.show()

# # PREDICTION-SMPL-GT POINT VIS
# DP = dp_utils.DensePoseMethods()
# # pkl_file = open('DensePoseData/demo_data/no.json', 'rb')
# coco = COCO("DensePoseData/demo_data/no.json")
# image = cv.imread('DensePoseData/demo_data/65736/COCO_val2014_000000065736.jpg')
# # Demo = pickle.load(pkl_file)
# ann_ids = coco.getAnnIds(65736)
# anns = coco.loadAnns(ann_ids)[0]

# bbr = np.round(anns['bbox'])
# x1,y1,x2,y2 = bbr[0],bbr[1],bbr[0]+bbr[2],bbr[1]+bbr[3]
# # x2 = min( [ x2,image.shape[1] ] ); y2 = min( [ y2,image.shape[0] ] )

# collected_x = np.zeros(np.array(anns['dp_x']).shape)
# collected_y = np.zeros(np.array(anns['dp_x']).shape)
# collected_z = np.zeros(np.array(anns['dp_x']).shape)

# # collected_x = np.zeros(Demo['x'].shape)
# # collected_y = np.zeros(Demo['x'].shape)
# # collected_z = np.zeros(Demo['x'].shape)

# # for i, (ii,uu,vv) in enumerate(zip(Demo['I'],Demo['U'],Demo['V'])):
# for i, (ii,uu,vv) in enumerate(zip(np.array(anns['dp_I']),np.array(anns['dp_U']),np.array(anns['dp_V']))):
#     # Convert IUV to FBC (faceIndex and barycentric coordinates.)
#     FaceIndex,bc1,bc2,bc3 = DP.IUV2FBC(ii,uu,vv)
#     # Use FBC to get 3D coordinates on the surface.
#     p = DP.FBC2PointOnSurface( FaceIndex, bc1,bc2,bc3,Vertices )
#     #
#     collected_x[i] = p[0]
#     collected_y[i] = p[1]
#     collected_z[i] = p[2]
# print("1")
# #fig = plt.figure(figsize=[15,5])

# # #index = random.sample(np.arange(len(Demo['y'])),2)
# # x1 = np.expand_dims(np.array(Demo['x'][4]), axis=0)
# # y1 = np.expand_dims(np.array(Demo['y'][4]), axis=0)
# # x2 = np.expand_dims(np.array(Demo['x'][4]), axis=0)
# # y2 = np.expand_dims(np.array(Demo['y'][4]), axis=0)
# # cz = collected_z[4:6]
# # cx = collected_x[4:6]
# # cy = collected_y[4:6]
# # Visualize the image and collected points.
# # ax = fig.add_subplot(133)
# # ax.imshow(Demo['ICrop'])
# # ax.scatter(x1,y1,25, np.arange(1)  )
# # ax.axis('off')

# # # Visualize the full body smpl male template model and collected points
# # ax = fig.add_subplot(132, projection='3d')
# # ax.scatter(Z,X,Y,s=0.02,c='k')
# # ax.scatter(cz, cx, cy,s=25,  c=  ['r', 'b']    )
# # smpl_view_set_axis_full_body(ax)


# # im_iuv = mpimg.imread('DensePoseData/demo_data/image_densepose_u.png') 
# # ax = fig.add_subplot(131)
# # ax.imshow(im_iuv)
# # ax.scatter(x2,y2,25, np.arange(1) )
# # ax.axis('off'), 
# # plt.show()

# ## RESULT VIS
# # UV MAP WITH POINT(PREDICTION AND GT) VIS

# Tex_Atlas = cv.imread('DensePoseData/demo_data/texture_atlas_200.png')[:,:,::-1]
# print("1.1")
# # i_gt = Demo["I"]
# # u_gt = Demo["U"]
# # v_gt = Demo["V"]
# i_gt = np.array(anns['dp_I'])
# u_gt = np.array(anns['dp_U'])
# v_gt = np.array(anns['dp_V'])

# part_m = i_gt - (((i_gt-1) // 6)*6+1)
# part_n = (i_gt-1) // 6
# # print(part_m.min(), part_m.max(), part_n.min(), part_n.max())
# v_cor = 200*part_m+1 + ((199.0)/(1.-0.))*(v_gt-0.) 
# u_cor = 200*part_n+1 + ((199.0)/(1.-0.))*(u_gt-0.) 
# # print(v_cor, u_cor)

# u_pre_im_base = cv.imread('DensePoseData/demo_data/65736/image_densepose_u.png',cv.IMREAD_UNCHANGED)/255.
# v_pre_im_base = cv.imread('DensePoseData/demo_data/65736/image_densepose_v.png',cv.IMREAD_UNCHANGED)/255.
# i_pre_im_base = cv.imread('DensePoseData/demo_data/65736/image_densepose_i.png',cv.IMREAD_UNCHANGED)

# u_pre_im_mmcl = cv.imread('DensePoseData/demo_data/65736/image_densepose_u_p.png',cv.IMREAD_UNCHANGED)/255.
# v_pre_im_mmcl = cv.imread('DensePoseData/demo_data/65736/image_densepose_v_p.png',cv.IMREAD_UNCHANGED)/255.
# i_pre_im_mmcl = cv.imread('DensePoseData/demo_data/65736/image_densepose_i_p.png',cv.IMREAD_UNCHANGED)
# # print(u_pre_im.shape, v_pre_im.shape, i_pre_im.shape, Demo['ICrop'].shape)
# # print(Demo["x"].max(), Demo["y"].max())
# # print(i_pre_im.max(), i_pre_im.min())
# # x = Demo["x"].astype(int)
# # y = Demo["y"].astype(int)

# x = (np.array(anns["dp_x"])/255. * bbr[2] +x1).astype(int)
# y = (np.array(anns["dp_y"])/255. * bbr[3] +y1).astype(int)

# # ax = fig.add_subplot(141)
# # # ax.imshow(Demo['ICrop'])
# # ax.imshow(image[:,:,::-1])
# # ax.scatter(x,y,s=100, c=np.arange(len(y)), marker='.')
# # ax.axis('off')


# u_pred_base = u_pre_im_base[y,x]
# v_pred_base = v_pre_im_base[y,x]
# i_pred_base = i_pre_im_base[y,x]

# u_pred_mmcl = u_pre_im_mmcl[y,x]
# v_pred_mmcl = v_pre_im_mmcl[y,x]
# i_pred_mmcl = i_pre_im_mmcl[y,x]


# def getUVofUVMap(u_pred, v_pred, i_pred):
#     i_pred_fg_index = np.argwhere(i_pred!=0)
#     i_pred_fg = i_pred[i_pred_fg_index].squeeze()
#     u_pred_fg = u_pred[i_pred_fg_index].squeeze()
#     v_pred_fg = v_pred[i_pred_fg_index].squeeze()

#     part_m_pred = (i_pred_fg - (((i_pred_fg-1) // 6)*6+1)).astype(int)
#     part_n_pred = ((i_pred_fg-1) // 6).astype(int)

#     v_cor_pred = (200 *part_m_pred) +1 + ((199.0)/(1.-0.))*(v_pred_fg-0.) 
#     u_cor_pred = (200 *part_n_pred) +1 + ((199.0)/(1.-0.))*(u_pred_fg-0.)
#     return u_cor_pred, v_cor_pred
# u_cor_pred_base, v_cor_pred_base = getUVofUVMap(u_pred_base, v_pred_base, i_pred_base)
# u_cor_pred_mmcl, v_cor_pred_mmcl = getUVofUVMap(u_pred_mmcl, v_pred_mmcl, i_pred_mmcl)
# fig = plt.figure(figsize=(24,16))
# fig.tight_layout()
# plt.subplots_adjust(wspace =0.1, hspace =0.1)
# # misclassifition
# x_base_miscls = x[i_pred_base != i_gt]
# y_base_miscls = y[i_pred_base != i_gt]
# x_mmcl_miscls = x[i_pred_mmcl != i_gt]
# y_mmcl_miscls = y[i_pred_mmcl != i_gt]

# ax = fig.add_subplot(231)
# ax.imshow(image[:,:,::-1])
# ax.scatter(x, y, s=100, c=np.arange(len(y)), marker='.')
# ax.scatter(x_base_miscls, y_base_miscls, s=100, c=np.arange(len(y_base_miscls)), marker='x')
# ax.axis('off'), 
# ax = fig.add_subplot(234)
# ax.imshow(image[:,:,::-1])
# ax.scatter(x, y, s=100, c=np.arange(len(y)), marker='.')
# ax.scatter(x_mmcl_miscls, y_mmcl_miscls, s=100, c=np.arange(len(y_mmcl_miscls)), marker='x')
# ax.axis('off'), 

# ax = fig.add_subplot(232)
# ax.imshow(Tex_Atlas)
# ax.scatter(v_cor.astype(int), u_cor.astype(int), 20, c="green")
# ax.scatter(v_cor.astype(int), u_cor.astype(int), 300, c="seagreen", alpha=0.2, linewidths=0)
# ax.scatter(v_cor_pred_base.astype(int), u_cor_pred_base.astype(int), 20, c="red")
# ax.axis('off') 
# ax = fig.add_subplot(235)
# ax.imshow(Tex_Atlas)
# ax.scatter(v_cor.astype(int), u_cor.astype(int), 20, c="green")
# ax.scatter(v_cor.astype(int), u_cor.astype(int), 300, c="seagreen", alpha=0.2, linewidths=0)
# ax.scatter(v_cor_pred_mmcl.astype(int), u_cor_pred_mmcl.astype(int), 20, c="red")
# ax.axis('off') 
# # plt.savefig("65736_2d.png", dpi=300)
# # plt.show()

# def getCollectedPred(i_pred, u_pred, v_pred):
#     i_pred_fg_index = np.argwhere(i_pred!=0)
#     i_gt_fg = i_gt[i_pred_fg_index]
#     i_pred_fg = i_pred[i_pred_fg_index]
#     collected_x_pred = np.zeros(np.array(anns['dp_x']).shape)
#     collected_y_pred = np.zeros(np.array(anns['dp_x']).shape)
#     collected_z_pred = np.zeros(np.array(anns['dp_x']).shape)
#     bg_pred = []
#     for i, (ii,uu,vv) in enumerate(zip(i_pred,u_pred,v_pred)):
#     # Convert IUV to FBC (faceIndex and barycentric coordinates.)
#         if ii == 0:
#             bg_pred.append(i)
#             continue
#         FaceIndex,bc1,bc2,bc3 = DP.IUV2FBC(ii,uu,vv)
#     # Use FBC to get 3D coordinates on the surface.
#         p = DP.FBC2PointOnSurface( FaceIndex, bc1,bc2,bc3,Vertices )
#     #
#         collected_x_pred[i] = p[0]
#         collected_y_pred[i] = p[1]
#         collected_z_pred[i] = p[2]

#     collected_x_pred = collected_x_pred[i_pred_fg_index]
#     collected_y_pred = collected_y_pred[i_pred_fg_index]
#     collected_z_pred = collected_z_pred[i_pred_fg_index]

#     collected_x_pred_miscla = collected_x_pred[i_pred_fg != i_gt_fg]
#     collected_y_pred_miscla = collected_y_pred[i_pred_fg != i_gt_fg]
#     collected_z_pred_miscla = collected_z_pred[i_pred_fg != i_gt_fg]
#     return collected_x_pred, collected_y_pred, collected_z_pred, collected_x_pred_miscla, collected_y_pred_miscla, collected_z_pred_miscla
# collected_x_base, collected_y_base, collected_z_base, collected_x_base_m, collected_y_base_m,collected_z_base_m = getCollectedPred(i_pred_base, u_pred_base, v_pred_base)
# collected_x_mmcl, collected_y_mmcl, collected_z_mmcl, collected_x_mmcl_m, collected_y_mmcl_m,collected_z_mmcl_m = getCollectedPred(i_pred_mmcl, u_pred_mmcl, v_pred_mmcl)

# # fig2 = plt.figure(figsize=(8,16))
# ax = fig.add_subplot(133, projection='3d')
# ax.scatter(Z,X,Y,s=0.02,c='k')
# ax.scatter(collected_z, collected_x, collected_y, s=100, c='green', marker='.')
# ax.scatter(collected_z_base, collected_x_base, collected_y_base, s=100, c='red', marker='.')
# ax.scatter(collected_z_base_m, collected_x_base_m, collected_y_base_m, s=120, c="red", marker='x')
# ax.scatter(collected_z_mmcl, collected_x_mmcl, collected_y_mmcl, s=100, c='blue', marker='.')
# ax.scatter(collected_z_mmcl_m, collected_x_mmcl_m, collected_y_mmcl_m, s=120, c="blue", marker='x')
# smpl_view_set_axis_full_body(ax)

# plt.savefig("65736_3D.png", dpi=300)
# plt.show()


# # # u_cor_pred_miscla = u_cor_pred[i_pred_fg != i_gt_fg]
# # # v_cor_pred_miscla = v_cor_pred[i_pred_fg != i_gt_fg]

# # # u_cor_line = np.hstack((u_cor,u_cor_pred))
# # # v_cor_line = np.hstack((v_cor,v_cor_pred))
# # # print(u_cor_line.shape, v_cor_line.shape)
# # ax = fig.add_subplot(142)
# # ax.imshow(Tex_Atlas)
# # # for i in range(len(u_cor_pred_line)):
# # #     ax.arrow(u_cor_pred_line[i], v_cor_pred_line[i], u_cor_line[i]-u_cor_pred_line[i], v_cor_line[i]-v_cor_pred_line[i], ec="slategray", fc="slategray", alpha=1, length_includes_head=True, linewidth=0.5, head_width=5)
# # ax.scatter(u_cor.astype(int), v_cor.astype(int), 30, c="green")
# # # ax.scatter(u_cor_miscls.astype(int), v_cor_miscls.astype(int), 15, c="darkgreen")
# # ax.scatter(u_cor.astype(int), v_cor.astype(int), 300, c="seagreen", alpha=0.2, linewidths=0)
# # ax.scatter(u_cor_pred.astype(int), v_cor_pred.astype(int), 30, c="red")
# # # ax.scatter(u_cor_pred_miscla.astype(int), v_cor_pred_miscla.astype(int), 15, c="darkred")
# # ax.axis('off') 

# # # SMPL MODEL WITH POINT(PREDICTION AND GT) VIS
# # # collected_x_pred = np.zeros(Demo['x'].shape)
# # # collected_y_pred = np.zeros(Demo['x'].shape)
# # # collected_z_pred = np.zeros(Demo['x'].shape)
# # collected_x_pred = np.zeros(np.array(anns['dp_x']).shape)
# # collected_y_pred = np.zeros(np.array(anns['dp_x']).shape)
# # collected_z_pred = np.zeros(np.array(anns['dp_x']).shape)
# # bg_pred = []
# # for i, (ii,uu,vv) in enumerate(zip(i_pred,u_pred,v_pred)):
# #     # Convert IUV to FBC (faceIndex and barycentric coordinates.)
# #     if ii == 0:
# #         bg_pred.append(i)
# #         continue
# #     FaceIndex,bc1,bc2,bc3 = DP.IUV2FBC(ii,uu,vv)
# #     # Use FBC to get 3D coordinates on the surface.
# #     p = DP.FBC2PointOnSurface( FaceIndex, bc1,bc2,bc3,Vertices )
# #     #
# #     collected_x_pred[i] = p[0]
# #     collected_y_pred[i] = p[1]
# #     collected_z_pred[i] = p[2]
# # # Visualize the full body smpl male template model and collected points
# # collected_x_miscla = collected_x[i_pred != i_gt]
# # collected_y_miscla = collected_y[i_pred != i_gt]
# # collected_z_miscla = collected_z[i_pred != i_gt]

# # collected_x_pred = collected_x_pred[i_pred_fg_index]
# # collected_y_pred = collected_y_pred[i_pred_fg_index]
# # collected_z_pred = collected_z_pred[i_pred_fg_index]

# # collected_x_line = np.delete(collected_x, bg_pred)
# # collected_y_line = np.delete(collected_y, bg_pred)
# # collected_z_line = np.delete(collected_z, bg_pred)
# # # collected_x_line = collected_x[i_pred_fg_index]
# # # collected_y_line = collected_y[i_pred_fg_index]
# # # collected_z_line = collected_z[i_pred_fg_index]
# # assert(len(collected_x_pred) == len(collected_x_line))

# # collected_x_pred_miscla = collected_x_pred[i_pred_fg != i_gt_fg]
# # collected_y_pred_miscla = collected_y_pred[i_pred_fg != i_gt_fg]
# # collected_z_pred_miscla = collected_z_pred[i_pred_fg != i_gt_fg]
# # collected_xyz_pred = zip(collected_z_pred, collected_x_pred, collected_y_pred)
# # collected_xyz = zip(collected_z_line, collected_x_line, collected_y_line)

# # ax = fig.add_subplot(143, projection='3d')
# # ax.scatter(Z,X,Y,s=0.02,c='k')
# # ax.scatter(collected_z, collected_x, collected_y, s=100, c='green', marker='.')
# # ax.scatter(collected_z_miscla, collected_x_miscla, collected_y_miscla, s=120, c="green", marker='x')
# # ax.scatter(collected_z_pred, collected_x_pred, collected_y_pred, s=100, c='red', marker='.')
# # ax.scatter(collected_z_pred_miscla, collected_x_pred_miscla, collected_y_pred_miscla, s=120, c="red", marker='x')
# # segments = [(collected_xyz_pred[i], collected_xyz[i]) for i in range(len(collected_x_pred))]
# # edge_col = Line3DCollection(segments, lw=2, linestyle='--')
# # ax.add_collection3d(edge_col)
# # smpl_view_set_axis_full_body(ax)

# # # misclassifition
# # x_miscls = x[i_pred != i_gt]
# # y_miscls = y[i_pred != i_gt]
# # ax = fig.add_subplot(144)
# # # ax.imshow(Demo['ICrop'])
# # ax.imshow(image[:,:,::-1])
# # ax.scatter(x, y, s=100, c=np.arange(len(y)), marker='.')
# # ax.scatter(x_miscls, y_miscls, s=100, c=np.arange(len(y_miscls)), marker='x')
# # ax.axis('off'), 

# # plt.savefig("439773.png", dpi=300)
# # plt.show()