# -*- coding = utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.
# pyre-ignore-all-errors

from detectron2.config import CfgNode as CN


def add_dataset_category_config(cfg: CN):
    """
    Add config for additional category-related dataset options
     - category whitelisting
     - category mapping
    """
    _C = cfg
    _C.DATASETS.CATEGORY_MAPS = CN(new_allowed=True)
    _C.DATASETS.WHITELISTED_CATEGORIES = CN(new_allowed=True)
    # class to mesh mapping
    _C.DATASETS.CLASS_TO_MESH_NAME_MAPPING = CN(new_allowed=True)


def add_evaluation_config(cfg: CN):
    _C = cfg
    _C.DENSEPOSE_EVALUATION = CN()
    # evaluator type, possible values:
    #  - "iou": evaluator for models that produce iou data
    #  - "cse": evaluator for models that produce cse data
    _C.DENSEPOSE_EVALUATION.TYPE = "iou"
    # storage for DensePose results, possible values:
    #  - "none": no explicit storage, all the results are stored in the
    #            dictionary with predictions, memory intensive;
    #            historically the default storage type
    #  - "ram": RAM storage, uses per-process RAM storage, which is
    #           reduced to a single process storage on later stages,
    #           less memory intensive
    #  - "file": file storage, uses per-process file-based storage,
    #            the least memory intensive, but may create bottlenecks
    #            on file system accesses
    _C.DENSEPOSE_EVALUATION.STORAGE = "none"
    # minimum threshold for IOU values: the lower its values is,
    # the more matches are produced (and the higher the AP score)
    _C.DENSEPOSE_EVALUATION.MIN_IOU_THRESHOLD = 0.5


def add_bootstrap_config(cfg: CN):
    """"""
    _C = cfg
    _C.BOOTSTRAP_DATASETS = []
    _C.BOOTSTRAP_MODEL = CN()
    _C.BOOTSTRAP_MODEL.WEIGHTS = ""
    _C.BOOTSTRAP_MODEL.DEVICE = "cuda"


def get_bootstrap_dataset_config() -> CN:
    _C = CN()
    _C.DATASET = ""
    # ratio used to mix data loaders
    _C.RATIO = 0.1
    # image loader
    _C.IMAGE_LOADER = CN(new_allowed=True)
    _C.IMAGE_LOADER.TYPE = ""
    _C.IMAGE_LOADER.BATCH_SIZE = 4
    _C.IMAGE_LOADER.NUM_WORKERS = 4
    # inference
    _C.INFERENCE = CN()
    # batch size for model inputs
    _C.INFERENCE.INPUT_BATCH_SIZE = 4
    # batch size to group model outputs
    _C.INFERENCE.OUTPUT_BATCH_SIZE = 2
    # sampled data
    _C.DATA_SAMPLER = CN(new_allowed=True)
    _C.DATA_SAMPLER.TYPE = ""
    # filter
    _C.FILTER = CN(new_allowed=True)
    _C.FILTER.TYPE = ""
    return _C


def load_bootstrap_config(cfg: CN):
    """
    Bootstrap datasets are given as a list of `dict` that are not automatically
    converted into CfgNode. This method processes all bootstrap dataset entries
    and ensures that they are in CfgNode format and comply with the specification
    """
    if not cfg.BOOTSTRAP_DATASETS:
        return

    bootstrap_datasets_cfgnodes = []
    for dataset_cfg in cfg.BOOTSTRAP_DATASETS:
        _C = get_bootstrap_dataset_config().clone()
        _C.merge_from_other_cfg(CN(dataset_cfg))
        bootstrap_datasets_cfgnodes.append(_C)
    cfg.BOOTSTRAP_DATASETS = bootstrap_datasets_cfgnodes


def add_densepose_head_cse_config(cfg: CN):
    """
    Add configuration options for Continuous Surface Embeddings (CSE)
    """
    _C = cfg
    _C.MODEL.ROI_DENSEPOSE_HEAD.CSE = CN()
    # Dimensionality D of the embedding space
    _C.MODEL.ROI_DENSEPOSE_HEAD.CSE.EMBED_SIZE = 16
    # Embedder specifications for various mesh IDs
    _C.MODEL.ROI_DENSEPOSE_HEAD.CSE.EMBEDDERS = CN(new_allowed=True)
    # normalization coefficient for embedding distances
    _C.MODEL.ROI_DENSEPOSE_HEAD.CSE.EMBEDDING_DIST_GAUSS_SIGMA = 0.01
    # normalization coefficient for geodesic distances
    _C.MODEL.ROI_DENSEPOSE_HEAD.CSE.GEODESIC_DIST_GAUSS_SIGMA = 0.01
    # embedding loss weight
    _C.MODEL.ROI_DENSEPOSE_HEAD.CSE.EMBED_LOSS_WEIGHT = 0.6
    # embedding loss name, currently the following options are supported:
    # - EmbeddingLoss: cross-entropy on vertex labels
    # - SoftEmbeddingLoss: cross-entropy on vertex label combined with
    #    Gaussian penalty on distance between vertices
    _C.MODEL.ROI_DENSEPOSE_HEAD.CSE.EMBED_LOSS_NAME = "EmbeddingLoss"
    # optimizer hyperparameters
    _C.MODEL.ROI_DENSEPOSE_HEAD.CSE.FEATURES_LR_FACTOR = 1.0
    _C.MODEL.ROI_DENSEPOSE_HEAD.CSE.EMBEDDING_LR_FACTOR = 1.0


def add_densepose_head_config(cfg: CN):
    """
    Add config for densepose head.
    """
    _C = cfg

    _C.MODEL.DENSEPOSE_ON = True

    _C.MODEL.ROI_DENSEPOSE_HEAD = CN()
    _C.MODEL.ROI_DENSEPOSE_HEAD.NAME = ""
    _C.MODEL.ROI_DENSEPOSE_HEAD.NUM_STACKED_CONVS = 8
    # Number of parts used for point labels
    _C.MODEL.ROI_DENSEPOSE_HEAD.NUM_PATCHES = 24
    _C.MODEL.ROI_DENSEPOSE_HEAD.DECONV_KERNEL = 4
    _C.MODEL.ROI_DENSEPOSE_HEAD.CONV_HEAD_DIM = 512
    _C.MODEL.ROI_DENSEPOSE_HEAD.CONV_HEAD_KERNEL = 3
    _C.MODEL.ROI_DENSEPOSE_HEAD.UP_SCALE = 2
    _C.MODEL.ROI_DENSEPOSE_HEAD.HEATMAP_SIZE = 112
    _C.MODEL.ROI_DENSEPOSE_HEAD.POOLER_TYPE = "ROIAlignV2"
    _C.MODEL.ROI_DENSEPOSE_HEAD.POOLER_RESOLUTION = 28
    _C.MODEL.ROI_DENSEPOSE_HEAD.POOLER_SAMPLING_RATIO = 2
    _C.MODEL.ROI_DENSEPOSE_HEAD.NUM_COARSE_SEGM_CHANNELS = 2  # 15 or 2
    # Overlap threshold for an RoI to be considered foreground (if >= FG_IOU_THRESHOLD)
    _C.MODEL.ROI_DENSEPOSE_HEAD.FG_IOU_THRESHOLD = 0.7
    # Loss weights for annotation masks.(14 Parts)
    _C.MODEL.ROI_DENSEPOSE_HEAD.INDEX_WEIGHTS = 5.0
    # Loss weights for surface parts. (24 Parts)
    _C.MODEL.ROI_DENSEPOSE_HEAD.PART_WEIGHTS = 1.0
    # Loss weights for UV regression.
    _C.MODEL.ROI_DENSEPOSE_HEAD.POINT_REGRESSION_WEIGHTS = 0.01
    # Coarse segmentation is trained using instance segmentation task data
    _C.MODEL.ROI_DENSEPOSE_HEAD.COARSE_SEGM_TRAINED_BY_MASKS = False
    # For Decoder
    _C.MODEL.ROI_DENSEPOSE_HEAD.DECODER_ON = True
    _C.MODEL.ROI_DENSEPOSE_HEAD.DECODER_NUM_CLASSES = 256
    _C.MODEL.ROI_DENSEPOSE_HEAD.DECODER_CONV_DIMS = 256
    _C.MODEL.ROI_DENSEPOSE_HEAD.DECODER_NORM = ""
    _C.MODEL.ROI_DENSEPOSE_HEAD.DECODER_COMMON_STRIDE = 4
    # For DeepLab head
    _C.MODEL.ROI_DENSEPOSE_HEAD.DEEPLAB = CN()
    _C.MODEL.ROI_DENSEPOSE_HEAD.DEEPLAB.NORM = "GN"
    _C.MODEL.ROI_DENSEPOSE_HEAD.DEEPLAB.NONLOCAL_ON = 0
    # Predictor class name, must be registered in DENSEPOSE_PREDICTOR_REGISTRY
    # Some registered predictors:
    #   "DensePoseChartPredictor": predicts segmentation and UV coordinates for predefined charts
    #   "DensePoseChartWithConfidencePredictor": predicts segmentation, UV coordinates
    #       and associated confidences for predefined charts (default)
    _C.MODEL.ROI_DENSEPOSE_HEAD.PREDICTOR_NAME = "DensePoseChartWithConfidencePredictor"
    # Loss class name, must be registered in DENSEPOSE_LOSS_REGISTRY
    # Some registered losses:
    #   "DensePoseChartLoss": loss for chart-based models that estimate
    #      segmentation and UV coordinates
    #   "DensePoseChartWithConfidenceLoss": loss for chart-based models that estimate
    #      segmentation, UV coordinates and the corresponding confidences (default)
    _C.MODEL.ROI_DENSEPOSE_HEAD.LOSS_NAME = "DensePoseChartWithConfidenceLoss"
    # Confidences
    # Enable learning UV confidences (variances) along with the actual values
    _C.MODEL.ROI_DENSEPOSE_HEAD.UV_CONFIDENCE = CN({"ENABLED": False})
    # UV confidence lower bound
    _C.MODEL.ROI_DENSEPOSE_HEAD.UV_CONFIDENCE.EPSILON = 0.01
    # Enable learning segmentation confidences (variances) along with the actual values
    _C.MODEL.ROI_DENSEPOSE_HEAD.SEGM_CONFIDENCE = CN({"ENABLED": False})
    # Segmentation confidence lower bound
    _C.MODEL.ROI_DENSEPOSE_HEAD.SEGM_CONFIDENCE.EPSILON = 0.01
    # Statistical model type for confidence learning, possible values:
    # - "iid_iso": statistically independent identically distributed residuals
    #    with isotropic covariance
    # - "indep_aniso": statistically independent residuals with anisotropic
    #    covariances
    _C.MODEL.ROI_DENSEPOSE_HEAD.UV_CONFIDENCE.TYPE = "iid_iso"
    # List of angles for rotation in data augmentation during training
    _C.INPUT.ROTATION_ANGLES = [0]
    _C.TEST.AUG.ROTATION_ANGLES = ()  # Rotation TTA

    add_densepose_head_cse_config(cfg)


def add_hrnet_config(cfg: CN):
    """
    Add config for HRNet backbone.
    """
    _C = cfg

    # For HigherHRNet w32
    _C.MODEL.HRNET = CN()
    _C.MODEL.HRNET.STEM_INPLANES = 64
    _C.MODEL.HRNET.STAGE2 = CN()
    _C.MODEL.HRNET.STAGE2.NUM_MODULES = 1
    _C.MODEL.HRNET.STAGE2.NUM_BRANCHES = 2
    _C.MODEL.HRNET.STAGE2.BLOCK = "BASIC"
    _C.MODEL.HRNET.STAGE2.NUM_BLOCKS = [4, 4]
    _C.MODEL.HRNET.STAGE2.NUM_CHANNELS = [32, 64]
    _C.MODEL.HRNET.STAGE2.FUSE_METHOD = "SUM"
    _C.MODEL.HRNET.STAGE3 = CN()
    _C.MODEL.HRNET.STAGE3.NUM_MODULES = 4
    _C.MODEL.HRNET.STAGE3.NUM_BRANCHES = 3
    _C.MODEL.HRNET.STAGE3.BLOCK = "BASIC"
    _C.MODEL.HRNET.STAGE3.NUM_BLOCKS = [4, 4, 4]
    _C.MODEL.HRNET.STAGE3.NUM_CHANNELS = [32, 64, 128]
    _C.MODEL.HRNET.STAGE3.FUSE_METHOD = "SUM"
    _C.MODEL.HRNET.STAGE4 = CN()
    _C.MODEL.HRNET.STAGE4.NUM_MODULES = 3
    _C.MODEL.HRNET.STAGE4.NUM_BRANCHES = 4
    _C.MODEL.HRNET.STAGE4.BLOCK = "BASIC"
    _C.MODEL.HRNET.STAGE4.NUM_BLOCKS = [4, 4, 4, 4]
    _C.MODEL.HRNET.STAGE4.NUM_CHANNELS = [32, 64, 128, 256]
    _C.MODEL.HRNET.STAGE4.FUSE_METHOD = "SUM"

    _C.MODEL.HRNET.HRFPN = CN()
    _C.MODEL.HRNET.HRFPN.OUT_CHANNELS = 256


def add_densepose_config(cfg: CN):
    add_densepose_head_config(cfg)
    add_hrnet_config(cfg)
    add_bootstrap_config(cfg)
    add_dataset_category_config(cfg)
    add_evaluation_config(cfg)
