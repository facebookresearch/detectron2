from detectron2.config import LazyCall as L
from detectron2.modeling.backbone import HFDINOv3ViT

from .mask_rcnn_vitdet_b_100ep import dataloader, lr_multiplier, model, optimizer, train


# Keep the ViTDet Mask R-CNN architecture unchanged, but replace the MAE ViT-B
# encoder with a frozen Hugging Face DINOv3 ViT-B/16 encoder.
model.backbone.net = L(HFDINOv3ViT)(
    model_name="facebook/dinov3-vitb16-pretrain-lvd1689m",
    out_feature="last_feat",
    pretrained=True,
    freeze=True,
)

# Initialize trainable detector components from the COCO ViTDet-B detector
# checkpoint. The DINOv3 encoder loads its own Hugging Face weights above.
train.init_checkpoint = (
    "https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/"
    "mask_rcnn_vitdet_b/f325346929/model_final_61ccd1.pkl"
)
