# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import logging
from fvcore.common.file_io import PathHandler, PathManager


class ThirdPartyModelCatalog(object):
    """
    Store mappings from names to third-party models.
    """

    S3_C2_DETECTRON_PREFIX = "https://dl.fbaipublicfiles.com/detectron"

    # MSRA models have STRIDE_IN_1X1=True. False otherwise.
    # NOTE: all BN models here have fused BN into an affine layer.
    # As a result, you should only load them to a model with "FrozenBN".
    # Loading them to a model with regular BN or SyncBN is wrong.
    # Even when loaded to FrozenBN, it is still different from affine by an epsilon,
    # which should be negligible for training.
    # NOTE: all models here uses PIXEL_STD=[1,1,1]
    C2_IMAGENET_MODELS = {
        "MSRA/R-50": "ImageNetPretrained/MSRA/R-50.pkl",
        "MSRA/R-101": "ImageNetPretrained/MSRA/R-101.pkl",
        "FAIR/R-50-GN": "ImageNetPretrained/47261647/R-50-GN.pkl",
        "FAIR/R-101-GN": "ImageNetPretrained/47592356/R-101-GN.pkl",
        "FAIR/X-101-32x8d": "ImageNetPretrained/20171220/X-101-32x8d.pkl",
        "FAIR/X-101-64x4d": "ImageNetPretrained/FBResNeXt/X-101-64x4d.pkl",
        "FAIR/X-152-32x8d-IN5k": "ImageNetPretrained/25093814/X-152-32x8d-IN5k.pkl",
    }

    C2_DETECTRON_PATH_FORMAT = (
        "{prefix}/{url}/output/train/{dataset}/{type}/model_final.pkl"
    )  # noqa B950

    C2_DATASET_COCO = "coco_2014_train%3Acoco_2014_valminusminival"
    C2_DATASET_COCO_KEYPOINTS = "keypoints_coco_2014_train%3Akeypoints_coco_2014_valminusminival"

    # format: {model_name} -> part of the url
    C2_DETECTRON_MODELS = {
        "35857197/e2e_faster_rcnn_R-50-C4_1x": "35857197/12_2017_baselines/e2e_faster_rcnn_R-50-C4_1x.yaml.01_33_49.iAX0mXvW",  # noqa B950
        "35857345/e2e_faster_rcnn_R-50-FPN_1x": "35857345/12_2017_baselines/e2e_faster_rcnn_R-50-FPN_1x.yaml.01_36_30.cUF7QR7I",  # noqa B950
        "35857890/e2e_faster_rcnn_R-101-FPN_1x": "35857890/12_2017_baselines/e2e_faster_rcnn_R-101-FPN_1x.yaml.01_38_50.sNxI7sX7",  # noqa B950
        "36761737/e2e_faster_rcnn_X-101-32x8d-FPN_1x": "36761737/12_2017_baselines/e2e_faster_rcnn_X-101-32x8d-FPN_1x.yaml.06_31_39.5MIHi1fZ",  # noqa B950
        "35858791/e2e_mask_rcnn_R-50-C4_1x": "35858791/12_2017_baselines/e2e_mask_rcnn_R-50-C4_1x.yaml.01_45_57.ZgkA7hPB",  # noqa B950
        "35858933/e2e_mask_rcnn_R-50-FPN_1x": "35858933/12_2017_baselines/e2e_mask_rcnn_R-50-FPN_1x.yaml.01_48_14.DzEQe4wC",  # noqa B950
        "35861795/e2e_mask_rcnn_R-101-FPN_1x": "35861795/12_2017_baselines/e2e_mask_rcnn_R-101-FPN_1x.yaml.02_31_37.KqyEK4tT",  # noqa B950
        "36761843/e2e_mask_rcnn_X-101-32x8d-FPN_1x": "36761843/12_2017_baselines/e2e_mask_rcnn_X-101-32x8d-FPN_1x.yaml.06_35_59.RZotkLKI",  # noqa B950
        "48616381/e2e_mask_rcnn_R-50-FPN_2x_gn": "GN/48616381/04_2018_gn_baselines/e2e_mask_rcnn_R-50-FPN_2x_gn_0416.13_23_38.bTlTI97Q",  # noqa B950
        "37697547/e2e_keypoint_rcnn_R-50-FPN_1x": "37697547/12_2017_baselines/e2e_keypoint_rcnn_R-50-FPN_1x.yaml.08_42_54.kdzV35ao",  # noqa B950
        "35998355/rpn_R-50-C4_1x": "35998355/12_2017_baselines/rpn_R-50-C4_1x.yaml.08_00_43.njH5oD9L",  # noqa B950
        "35998814/rpn_R-50-FPN_1x": "35998814/12_2017_baselines/rpn_R-50-FPN_1x.yaml.08_06_03.Axg0r179",  # noqa B950
        "36225147/fast_R-50-FPN_1x": "36225147/12_2017_baselines/fast_rcnn_R-50-FPN_1x.yaml.08_39_09.L3obSdQ2",  # noqa B950
    }

    @staticmethod
    def get(name):
        if name.startswith("Caffe2Detectron/COCO"):
            return ThirdPartyModelCatalog._get_c2_detectron_baseline(name)
        if name.startswith("ImageNetPretrained/"):
            return ThirdPartyModelCatalog._get_c2_imagenet_pretrained(name)
        raise RuntimeError("model not present in the catalog: {}".format(name))

    @staticmethod
    def _get_c2_imagenet_pretrained(name):
        prefix = ThirdPartyModelCatalog.S3_C2_DETECTRON_PREFIX
        name = name[len("ImageNetPretrained/") :]
        name = ThirdPartyModelCatalog.C2_IMAGENET_MODELS[name]
        url = "/".join([prefix, name])
        return url

    @staticmethod
    def _get_c2_detectron_baseline(name):
        name = name[len("Caffe2Detectron/COCO/") :]
        url = ThirdPartyModelCatalog.C2_DETECTRON_MODELS[name]
        if "keypoint_rcnn" in name:
            dataset = ThirdPartyModelCatalog.C2_DATASET_COCO_KEYPOINTS
        else:
            dataset = ThirdPartyModelCatalog.C2_DATASET_COCO

        if "35998355/rpn_R-50-C4_1x" in name:
            # this one model is somehow different from others ..
            type = "rpn"
        else:
            type = "generalized_rcnn"

        # Detectron C2 models are stored in the structure defined in `C2_DETECTRON_PATH_FORMAT`.
        url = ThirdPartyModelCatalog.C2_DETECTRON_PATH_FORMAT.format(
            prefix=ThirdPartyModelCatalog.S3_C2_DETECTRON_PREFIX,
            url=url,
            type=type,
            dataset=dataset,
        )
        return url


class ThirdPartyModelCatalogHandler(PathHandler):
    """
    Resolve URL like catalog://.
    """

    PREFIX = "catalog://"

    def _get_supported_prefixes(self):
        return [self.PREFIX]

    def _get_local_path(self, path):
        logger = logging.getLogger(__name__)
        catalog_path = ThirdPartyModelCatalog.get(path[len(self.PREFIX) :])
        logger.info("Catalog entry {} points to {}".format(path, catalog_path))
        return PathManager.get_local_path(catalog_path)

    def _open(self, path, mode="r", **kwargs):
        return PathManager.open(self._get_local_path(path), mode, **kwargs)


class Detectron2ModelCatalog(object):
    """
    Store mappings from names to officially released Detectron2 pre-trained models.
    """

    S3_DETECTRON2_PREFIX = "https://dl.fbaipublicfiles.com/detectron2/"

    # format: {model_name} -> model_id/model_final_{commit}.pkl
    DETECTRON2_MODELS = {
        "COCO-Detection/faster_rcnn_R_50_C4_1x": "137257644/model_final_721ade.pkl",
        "COCO-Detection/faster_rcnn_R_50_DC5_1x": "137847829/model_final_51d356.pkl",
        "COCO-Detection/faster_rcnn_R_50_FPN_1x": "137257794/model_final_b275ba.pkl",
        "COCO-Detection/faster_rcnn_R_50_C4_3x": "137849393/model_final_f97cb7.pkl",
        "COCO-Detection/faster_rcnn_R_50_DC5_3x": "137849425/model_final_68d202.pkl",
        "COCO-Detection/faster_rcnn_R_50_FPN_3x": "137849458/model_final_280758.pkl",
        "COCO-Detection/faster_rcnn_R_101_C4_3x": "138204752/model_final_298dad.pkl",
        "COCO-Detection/faster_rcnn_R_101_DC5_3x": "138204841/model_final_3e0943.pkl",
        "COCO-Detection/faster_rcnn_R_101_FPN_3x": "137851257/model_final_f6e8b1.pkl",
        "COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x": "139173657/model_final_68b088.pkl",
    }

    @classmethod
    def get(cls, name):
        if name.startswith("ImageNetPretrained/"):
            return cls._get_imagenet_pretrained_backbone(name)
        if name.startswith("COCO-Detection/"):
            return cls._get_detectron2_pretrained_model(name)
        raise RuntimeError("model not present in the catalog: {}".format(name))

    @classmethod
    def _get_imagenet_pretrained_backbone(cls, name):
        url = cls.S3_DETECTRON2_PREFIX + name
        return url

    @classmethod
    def _get_detectron2_pretrained_model(cls, name):
        if name.endswith(".pkl"):
            # Backward compatibility: this would already be a URL.
            url = cls.S3_DETECTRON2_PREFIX + name
        else:
            url = cls.S3_DETECTRON2_PREFIX + name + "/" + cls.DETECTRON2_MODELS[name]
        return url


class Detectron2Handler(PathHandler):
    """
    Resolve anything that's in Detectron2 model zoo.
    """

    PREFIX = "detectron2://"

    def _get_supported_prefixes(self):
        return [self.PREFIX]

    def _get_local_path(self, path):
        logger = logging.getLogger(__name__)
        catalog_path = Detectron2ModelCatalog.get(path[len(self.PREFIX) :])
        logger.info("Catalog entry {} points to {}".format(path, catalog_path))
        return PathManager.get_local_path(catalog_path)

    def _open(self, path, mode="r", **kwargs):
        return PathManager.open(self._get_local_path(path), mode, **kwargs)


PathManager.register_handler(ThirdPartyModelCatalogHandler())
PathManager.register_handler(Detectron2Handler())
