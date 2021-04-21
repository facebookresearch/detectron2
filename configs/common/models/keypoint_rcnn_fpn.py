from detectron2.config import LazyCall as L
from detectron2.layers import ShapeSpec
from detectron2.modeling.poolers import ROIPooler
from detectron2.modeling.roi_heads import KRCNNConvDeconvUpsampleHead

from .mask_rcnn_fpn import model

[model.roi_heads.pop(x) for x in ["mask_in_features", "mask_pooler", "mask_head"]]

model.roi_heads.update(
    num_classes=1,
    keypoint_in_features=["p2", "p3", "p4", "p5"],
    keypoint_pooler=L(ROIPooler)(
        output_size=14,
        scales=(1.0 / 4, 1.0 / 8, 1.0 / 16, 1.0 / 32),
        sampling_ratio=0,
        pooler_type="ROIAlignV2",
    ),
    keypoint_head=L(KRCNNConvDeconvUpsampleHead)(
        input_shape=ShapeSpec(channels=256, width=14, height=14),
        num_keypoints=17,
        conv_dims=[512] * 8,
        loss_normalizer="visible",
    ),
)

# Detectron1 uses 2000 proposals per-batch, but this option is per-image in detectron2.
# 1000 proposals per-image is found to hurt box AP.
# Therefore we increase it to 1500 per-image.
model.proposal_generator.post_nms_topk = (1500, 1000)

# Keypoint AP degrades (though box AP improves) when using plain L1 loss
model.roi_heads.box_predictor.smooth_l1_beta = 0.5
