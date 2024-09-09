from detectron2.config import LazyCall as L
from detectron2.layers import ShapeSpec
from detectron2.modeling.box_regression import Box2BoxTransform
from detectron2.modeling.matcher import Matcher
from detectron2.modeling.roi_heads import FastRCNNOutputLayers, FastRCNNConvFCHead, CascadeROIHeads
from detectron2.layers.batch_norm import NaiveSyncBatchNorm

from .mask_rcnn_mvitv2_t_3x import model, dataloader, optimizer, lr_multiplier, train


# arguments that don't exist for Cascade R-CNN
[model.roi_heads.pop(k) for k in ["box_head", "box_predictor", "proposal_matcher"]]

model.roi_heads.update(
    _target_=CascadeROIHeads,
    box_heads=[
        L(FastRCNNConvFCHead)(
            input_shape=ShapeSpec(channels=256, height=7, width=7),
            conv_dims=[256, 256, 256, 256],
            fc_dims=[1024],
            conv_norm=lambda c: NaiveSyncBatchNorm(c, stats_mode="N"),
        )
        for _ in range(3)
    ],
    box_predictors=[
        L(FastRCNNOutputLayers)(
            input_shape=ShapeSpec(channels=1024),
            test_score_thresh=0.05,
            box2box_transform=L(Box2BoxTransform)(weights=(w1, w1, w2, w2)),
            cls_agnostic_bbox_reg=True,
            num_classes="${...num_classes}",
        )
        for (w1, w2) in [(10, 5), (20, 10), (30, 15)]
    ],
    proposal_matchers=[
        L(Matcher)(thresholds=[th], labels=[0, 1], allow_low_quality_matches=False)
        for th in [0.5, 0.6, 0.7]
    ],
)

# Using NaiveSyncBatchNorm becase heads may have empty input. That is not supported by
# torch.nn.SyncBatchNorm. We can remove this after
# https://github.com/pytorch/pytorch/issues/36530 is fixed.
model.roi_heads.mask_head.conv_norm = lambda c: NaiveSyncBatchNorm(c, stats_mode="N")

# 2conv in RPN:
# https://github.com/tensorflow/tpu/blob/b24729de804fdb751b06467d3dce0637fa652060/models/official/detection/modeling/architecture/heads.py#L95-L97  # noqa: E501, B950
model.proposal_generator.head.conv_dims = [-1, -1]
