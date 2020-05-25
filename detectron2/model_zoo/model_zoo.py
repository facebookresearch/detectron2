# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import os
import pkg_resources
import torch

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.modeling import build_model


class _ModelZooUrls(object):
    """
    Mapping from names to officially released Detectron2 pre-trained models.
    """

    S3_PREFIX = "https://dl.fbaipublicfiles.com/detectron2/"

    # format: {config_path.yaml} -> model_id/model_final_{commit}.pkl
    CONFIG_PATH_TO_URL_SUFFIX = {
        # COCO Detection with Faster R-CNN
        "COCO-Detection/faster_rcnn_R_50_C4_1x.yaml": "137257644/model_final_721ade.pkl",
        "COCO-Detection/faster_rcnn_R_50_DC5_1x.yaml": "137847829/model_final_51d356.pkl",
        "COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml": "137257794/model_final_b275ba.pkl",
        "COCO-Detection/faster_rcnn_R_50_C4_3x.yaml": "137849393/model_final_f97cb7.pkl",
        "COCO-Detection/faster_rcnn_R_50_DC5_3x.yaml": "137849425/model_final_68d202.pkl",
        "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml": "137849458/model_final_280758.pkl",
        "COCO-Detection/faster_rcnn_R_101_C4_3x.yaml": "138204752/model_final_298dad.pkl",
        "COCO-Detection/faster_rcnn_R_101_DC5_3x.yaml": "138204841/model_final_3e0943.pkl",
        "COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml": "137851257/model_final_f6e8b1.pkl",
        "COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml": "139173657/model_final_68b088.pkl",
        # COCO Detection with RetinaNet
        "COCO-Detection/retinanet_R_50_FPN_1x.yaml": "190397773/model_final_bfca0b.pkl",
        "COCO-Detection/retinanet_R_50_FPN_3x.yaml": "190397829/model_final_5bd44e.pkl",
        "COCO-Detection/retinanet_R_101_FPN_3x.yaml": "190397697/model_final_971ab9.pkl",
        # COCO Detection with RPN and Fast R-CNN
        "COCO-Detection/rpn_R_50_C4_1x.yaml": "137258005/model_final_450694.pkl",
        "COCO-Detection/rpn_R_50_FPN_1x.yaml": "137258492/model_final_02ce48.pkl",
        "COCO-Detection/fast_rcnn_R_50_FPN_1x.yaml": "137635226/model_final_e5f7ce.pkl",
        # COCO Instance Segmentation Baselines with Mask R-CNN
        "COCO-InstanceSegmentation/mask_rcnn_R_50_C4_1x.yaml": "137259246/model_final_9243eb.pkl",
        "COCO-InstanceSegmentation/mask_rcnn_R_50_DC5_1x.yaml": "137260150/model_final_4f86c3.pkl",
        "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml": "137260431/model_final_a54504.pkl",
        "COCO-InstanceSegmentation/mask_rcnn_R_50_C4_3x.yaml": "137849525/model_final_4ce675.pkl",
        "COCO-InstanceSegmentation/mask_rcnn_R_50_DC5_3x.yaml": "137849551/model_final_84107b.pkl",
        "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml": "137849600/model_final_f10217.pkl",
        "COCO-InstanceSegmentation/mask_rcnn_R_101_C4_3x.yaml": "138363239/model_final_a2914c.pkl",
        "COCO-InstanceSegmentation/mask_rcnn_R_101_DC5_3x.yaml": "138363294/model_final_0464b7.pkl",
        "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml": "138205316/model_final_a3ec72.pkl",
        "COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml": "139653917/model_final_2d9806.pkl",  # noqa
        # COCO Person Keypoint Detection Baselines with Keypoint R-CNN
        "COCO-Keypoints/keypoint_rcnn_R_50_FPN_1x.yaml": "137261548/model_final_04e291.pkl",
        "COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml": "137849621/model_final_a6e10b.pkl",
        "COCO-Keypoints/keypoint_rcnn_R_101_FPN_3x.yaml": "138363331/model_final_997cc7.pkl",
        "COCO-Keypoints/keypoint_rcnn_X_101_32x8d_FPN_3x.yaml": "139686956/model_final_5ad38f.pkl",
        # COCO Panoptic Segmentation Baselines with Panoptic FPN
        "COCO-PanopticSegmentation/panoptic_fpn_R_50_1x.yaml": "139514544/model_final_dbfeb4.pkl",
        "COCO-PanopticSegmentation/panoptic_fpn_R_50_3x.yaml": "139514569/model_final_c10459.pkl",
        "COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml": "139514519/model_final_cafdb1.pkl",
        # LVIS Instance Segmentation Baselines with Mask R-CNN
        "LVIS-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml": "144219072/model_final_571f7c.pkl",
        "LVIS-InstanceSegmentation/mask_rcnn_R_101_FPN_1x.yaml": "144219035/model_final_824ab5.pkl",
        "LVIS-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_1x.yaml": "144219108/model_final_5e3439.pkl",  # noqa
        # Cityscapes & Pascal VOC Baselines
        "Cityscapes/mask_rcnn_R_50_FPN.yaml": "142423278/model_final_af9cf5.pkl",
        "PascalVOC-Detection/faster_rcnn_R_50_C4.yaml": "142202221/model_final_b1acc2.pkl",
        # Other Settings
        "Misc/mask_rcnn_R_50_FPN_1x_dconv_c3-c5.yaml": "138602867/model_final_65c703.pkl",
        "Misc/mask_rcnn_R_50_FPN_3x_dconv_c3-c5.yaml": "144998336/model_final_821d0b.pkl",
        "Misc/cascade_mask_rcnn_R_50_FPN_1x.yaml": "138602847/model_final_e9d89b.pkl",
        "Misc/cascade_mask_rcnn_R_50_FPN_3x.yaml": "144998488/model_final_480dd8.pkl",
        "Misc/mask_rcnn_R_50_FPN_3x_syncbn.yaml": "169527823/model_final_3b3c51.pkl",
        "Misc/mask_rcnn_R_50_FPN_3x_gn.yaml": "138602888/model_final_dc5d9e.pkl",
        "Misc/scratch_mask_rcnn_R_50_FPN_3x_gn.yaml": "138602908/model_final_01ca85.pkl",
        "Misc/panoptic_fpn_R_101_dconv_cascade_gn_3x.yaml": "139797668/model_final_be35db.pkl",
        "Misc/cascade_mask_rcnn_X_152_32x8d_FPN_IN5k_gn_dconv.yaml": "18131413/model_0039999_e76410.pkl",  # noqa
        # D1 Comparisons
        "Detectron1-Comparisons/faster_rcnn_R_50_FPN_noaug_1x.yaml": "137781054/model_final_7ab50c.pkl",  # noqa
        "Detectron1-Comparisons/mask_rcnn_R_50_FPN_noaug_1x.yaml": "137781281/model_final_62ca52.pkl",  # noqa
        "Detectron1-Comparisons/keypoint_rcnn_R_50_FPN_1x.yaml": "137781195/model_final_cce136.pkl",
    }


def get_checkpoint_url(config_path):
    """
    Returns the URL to the model trained using the given config

    Args:
        config_path (str): config file name relative to detectron2's "configs/"
            directory, e.g., "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml"

    Returns:
        str: a URL to the model
    """
    name = config_path.replace(".yaml", "")
    if config_path in _ModelZooUrls.CONFIG_PATH_TO_URL_SUFFIX:
        suffix = _ModelZooUrls.CONFIG_PATH_TO_URL_SUFFIX[config_path]
        return _ModelZooUrls.S3_PREFIX + name + "/" + suffix
    raise RuntimeError("{} not available in Model Zoo!".format(name))


def get_config_file(config_path):
    """
    Returns path to a builtin config file.

    Args:
        config_path (str): config file name relative to detectron2's "configs/"
            directory, e.g., "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml"

    Returns:
        str: the real path to the config file.
    """
    cfg_file = pkg_resources.resource_filename(
        "detectron2.model_zoo", os.path.join("configs", config_path)
    )
    if not os.path.exists(cfg_file):
        raise RuntimeError("{} not available in Model Zoo!".format(config_path))
    return cfg_file


def get(config_path, trained: bool = False):
    """
    Get a model specified by relative path under Detectron2's official ``configs/`` directory.

    Args:
        config_path (str): config file name relative to detectron2's "configs/"
            directory, e.g., "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml"
        trained (bool): If True, will initialize the model with the trained model zoo weights.
            If False, the checkpoint specified in the config file's ``MODEL.WEIGHTS`` is used
            instead; this will typically (though not always) initialize a subset of weights using
            an ImageNet pre-trained model, while randomly initializing the other weights.

    Example:

    .. code-block:: python

        from detectron2 import model_zoo
        model = model_zoo.get("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml", trained=True)
    """
    cfg_file = get_config_file(config_path)

    cfg = get_cfg()
    cfg.merge_from_file(cfg_file)
    if trained:
        cfg.MODEL.WEIGHTS = get_checkpoint_url(config_path)
    if not torch.cuda.is_available():
        cfg.MODEL.DEVICE = "cpu"

    model = build_model(cfg)
    DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)
    return model
