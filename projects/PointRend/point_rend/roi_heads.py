# Copyright (c) Facebook, Inc. and its affiliates.
import logging

from detectron2.modeling import ROI_HEADS_REGISTRY, StandardROIHeads


@ROI_HEADS_REGISTRY.register()
class PointRendROIHeads(StandardROIHeads):
    """
    Identical to StandardROIHeads, except for some weights conversion code to
    handle old models.
    """

    _version = 2

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        version = local_metadata.get("version", None)
        if version is None or version < 2:
            logger = logging.getLogger(__name__)
            logger.warning(
                "Weight format of PointRend models have changed! "
                "Please upgrade your models. Applying automatic conversion now ..."
            )
            for k in list(state_dict.keys()):
                newk = k
                if k.startswith(prefix + "mask_point_head"):
                    newk = k.replace(prefix + "mask_point_head", prefix + "mask_head.point_head")
                if k.startswith(prefix + "mask_coarse_head"):
                    newk = k.replace(prefix + "mask_coarse_head", prefix + "mask_head.coarse_head")
                if newk != k:
                    state_dict[newk] = state_dict[k]
                    del state_dict[k]

    @classmethod
    def _init_mask_head(cls, cfg, input_shape):
        if cfg.MODEL.MASK_ON and cfg.MODEL.ROI_MASK_HEAD.NAME != "PointRendMaskHead":
            logger = logging.getLogger(__name__)
            logger.warning(
                "Config of PointRend models have changed! "
                "Please upgrade your models. Applying automatic conversion now ..."
            )
            assert cfg.MODEL.ROI_MASK_HEAD.NAME == "CoarseMaskHead"
            cfg.defrost()
            cfg.MODEL.ROI_MASK_HEAD.NAME = "PointRendMaskHead"
            cfg.MODEL.ROI_MASK_HEAD.POOLER_TYPE = ""
            cfg.freeze()
        return super()._init_mask_head(cfg, input_shape)
