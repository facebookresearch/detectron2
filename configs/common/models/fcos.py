from detectron2.modeling.meta_arch.fcos import FCOS, FCOSHead

from .retinanet import model

model._target_ = FCOS

del model.anchor_generator
del model.box2box_transform
del model.anchor_matcher
del model.input_format

# Use P5 instead of C5 to compute P6/P7
# (Sec 2.2 of https://arxiv.org/abs/2006.09214)
model.backbone.top_block.in_feature = "p5"
model.backbone.top_block.in_channels = 256

# New score threshold determined based on sqrt(cls_score * centerness)
model.test_score_thresh = 0.2
model.test_nms_thresh = 0.6

model.head._target_ = FCOSHead
del model.head.num_anchors
model.head.norm = "GN"
