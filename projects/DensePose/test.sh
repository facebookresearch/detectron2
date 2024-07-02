#
python apply_net.py show configs/densepose_rcnn_R_50_FPN_s1x.yaml ckpt/R-50.pkl image.jpg dp_contour,bbox \
    --output image_densepose_contour.png