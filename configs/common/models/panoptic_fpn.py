from detectron2.config import LazyCall as L
from detectron2.layers import ShapeSpec
from detectron2.modeling import PanopticFPN
from detectron2.modeling.meta_arch.semantic_seg import SemSegFPNHead

from .mask_rcnn_fpn import model

model._target_ = PanopticFPN
model.sem_seg_head = L(SemSegFPNHead)(
    input_shape={
        f: L(ShapeSpec)(stride=s, channels="${....backbone.out_channels}")
        for f, s in zip(["p2", "p3", "p4", "p5"], [4, 8, 16, 32])
    },
    ignore_value=255,
    num_classes=54,  # COCO stuff + 1
    conv_dims=128,
    common_stride=4,
    loss_weight=0.5,
    norm="GN",
)
