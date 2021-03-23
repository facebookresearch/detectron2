# Copyright (c) Facebook, Inc. and its affiliates.

import itertools
import logging
import numpy as np
from collections import UserDict, defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Collection, Dict, Iterable, List, Optional, Sequence, Tuple
import torch
from torch.utils.data.dataset import Dataset

from detectron2.config import CfgNode
from detectron2.data.build import build_detection_test_loader as d2_build_detection_test_loader
from detectron2.data.build import build_detection_train_loader as d2_build_detection_train_loader
from detectron2.data.build import (
    load_proposals_into_dataset,
    print_instances_class_histogram,
    trivial_batch_collator,
    worker_init_reset_seed,
)
from detectron2.data.catalog import DatasetCatalog, Metadata, MetadataCatalog
from detectron2.data.samplers import TrainingSampler
from detectron2.utils.comm import get_world_size

from densepose.config import get_bootstrap_dataset_config

from .combined_loader import CombinedDataLoader, Loader
from .dataset_mapper import DatasetMapper
from .datasets.coco import DENSEPOSE_CSE_KEYS_WITHOUT_MASK, DENSEPOSE_IUV_KEYS_WITHOUT_MASK
from .datasets.dataset_type import DatasetType
from .inference_based_loader import InferenceBasedLoader, ScoreBasedFilter
from .samplers import (
    DensePoseConfidenceBasedSampler,
    DensePoseUniformSampler,
    MaskFromDensePoseSampler,
    PredictionToGroundTruthSampler,
)
from .transform import ImageResizeTransform
from .video import (
    FirstKFramesSelector,
    FrameSelectionStrategy,
    LastKFramesSelector,
    RandomKFramesSelector,
    VideoKeyframeDataset,
    video_list_from_file,
)

__all__ = ["build_detection_train_loader", "build_detection_test_loader"]


Instance = Dict[str, Any]
InstancePredicate = Callable[[Instance], bool]


def _compute_num_images_per_worker(cfg: CfgNode):
    num_workers = get_world_size()
    images_per_batch = cfg.SOLVER.IMS_PER_BATCH
    assert (
        images_per_batch % num_workers == 0
    ), "SOLVER.IMS_PER_BATCH ({}) must be divisible by the number of workers ({}).".format(
        images_per_batch, num_workers
    )
    assert (
        images_per_batch >= num_workers
    ), "SOLVER.IMS_PER_BATCH ({}) must be larger than the number of workers ({}).".format(
        images_per_batch, num_workers
    )
    images_per_worker = images_per_batch // num_workers
    return images_per_worker


def _map_category_id_to_contiguous_id(dataset_name: str, dataset_dicts: Iterable[Instance]):
    meta = MetadataCatalog.get(dataset_name)
    for dataset_dict in dataset_dicts:
        for ann in dataset_dict["annotations"]:
            ann["category_id"] = meta.thing_dataset_id_to_contiguous_id[ann["category_id"]]


@dataclass
class _DatasetCategory:
    """
    Class representing category data in a dataset:
     - id: category ID, as specified in the dataset annotations file
     - name: category name, as specified in the dataset annotations file
     - mapped_id: category ID after applying category maps (DATASETS.CATEGORY_MAPS config option)
     - mapped_name: category name after applying category maps
     - dataset_name: dataset in which the category is defined

    For example, when training models in a class-agnostic manner, one could take LVIS 1.0
    dataset and map the animal categories to the same category as human data from COCO:
     id = 225
     name = "cat"
     mapped_id = 1
     mapped_name = "person"
     dataset_name = "lvis_v1_animals_dp_train"
    """

    id: int
    name: str
    mapped_id: int
    mapped_name: str
    dataset_name: str


_MergedCategoriesT = Dict[int, List[_DatasetCategory]]


def _add_category_id_to_contiguous_id_maps_to_metadata(merged_categories: _MergedCategoriesT):
    merged_categories_per_dataset = {}
    for contiguous_cat_id, cat_id in enumerate(sorted(merged_categories.keys())):
        for cat in merged_categories[cat_id]:
            if cat.dataset_name not in merged_categories_per_dataset:
                merged_categories_per_dataset[cat.dataset_name] = defaultdict(list)
            merged_categories_per_dataset[cat.dataset_name][cat_id].append(
                (
                    contiguous_cat_id,
                    cat,
                )
            )

    logger = logging.getLogger(__name__)
    for dataset_name, merged_categories in merged_categories_per_dataset.items():
        meta = MetadataCatalog.get(dataset_name)
        if not hasattr(meta, "thing_classes"):
            meta.thing_classes = []
            meta.thing_dataset_id_to_contiguous_id = {}
            meta.thing_dataset_id_to_merged_id = {}
        else:
            meta.thing_classes.clear()
            meta.thing_dataset_id_to_contiguous_id.clear()
            meta.thing_dataset_id_to_merged_id.clear()
        logger.info(f"Dataset {dataset_name}: category ID to contiguous ID mapping:")
        for _cat_id, categories in sorted(merged_categories.items()):
            added_to_thing_classes = False
            for contiguous_cat_id, cat in categories:
                if not added_to_thing_classes:
                    meta.thing_classes.append(cat.mapped_name)
                    added_to_thing_classes = True
                meta.thing_dataset_id_to_contiguous_id[cat.id] = contiguous_cat_id
                meta.thing_dataset_id_to_merged_id[cat.id] = cat.mapped_id
                logger.info(f"{cat.id} ({cat.name}) -> {contiguous_cat_id}")


def _maybe_create_general_keep_instance_predicate(cfg: CfgNode) -> Optional[InstancePredicate]:
    def has_annotations(instance: Instance) -> bool:
        return "annotations" in instance

    def has_only_crowd_anotations(instance: Instance) -> bool:
        for ann in instance["annotations"]:
            if ann.get("is_crowd", 0) == 0:
                return False
        return True

    def general_keep_instance_predicate(instance: Instance) -> bool:
        return has_annotations(instance) and not has_only_crowd_anotations(instance)

    if not cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS:
        return None
    return general_keep_instance_predicate


def _maybe_create_keypoints_keep_instance_predicate(cfg: CfgNode) -> Optional[InstancePredicate]:

    min_num_keypoints = cfg.MODEL.ROI_KEYPOINT_HEAD.MIN_KEYPOINTS_PER_IMAGE

    def has_sufficient_num_keypoints(instance: Instance) -> bool:
        num_kpts = sum(
            (np.array(ann["keypoints"][2::3]) > 0).sum()
            for ann in instance["annotations"]
            if "keypoints" in ann
        )
        return num_kpts >= min_num_keypoints

    if cfg.MODEL.KEYPOINT_ON and (min_num_keypoints > 0):
        return has_sufficient_num_keypoints
    return None


def _maybe_create_mask_keep_instance_predicate(cfg: CfgNode) -> Optional[InstancePredicate]:
    if not cfg.MODEL.MASK_ON:
        return None

    def has_mask_annotations(instance: Instance) -> bool:
        return any("segmentation" in ann for ann in instance["annotations"])

    return has_mask_annotations


def _maybe_create_densepose_keep_instance_predicate(cfg: CfgNode) -> Optional[InstancePredicate]:
    if not cfg.MODEL.DENSEPOSE_ON:
        return None

    use_masks = cfg.MODEL.ROI_DENSEPOSE_HEAD.COARSE_SEGM_TRAINED_BY_MASKS

    def has_densepose_annotations(instance: Instance) -> bool:
        for ann in instance["annotations"]:
            if all(key in ann for key in DENSEPOSE_IUV_KEYS_WITHOUT_MASK) or all(
                key in ann for key in DENSEPOSE_CSE_KEYS_WITHOUT_MASK
            ):
                return True
            if use_masks and "segmentation" in ann:
                return True
        return False

    return has_densepose_annotations


def _maybe_create_specific_keep_instance_predicate(cfg: CfgNode) -> Optional[InstancePredicate]:
    specific_predicate_creators = [
        _maybe_create_keypoints_keep_instance_predicate,
        _maybe_create_mask_keep_instance_predicate,
        _maybe_create_densepose_keep_instance_predicate,
    ]
    predicates = [creator(cfg) for creator in specific_predicate_creators]
    predicates = [p for p in predicates if p is not None]
    if not predicates:
        return None

    def combined_predicate(instance: Instance) -> bool:
        return any(p(instance) for p in predicates)

    return combined_predicate


def _get_train_keep_instance_predicate(cfg: CfgNode):
    general_keep_predicate = _maybe_create_general_keep_instance_predicate(cfg)
    combined_specific_keep_predicate = _maybe_create_specific_keep_instance_predicate(cfg)

    def combined_general_specific_keep_predicate(instance: Instance) -> bool:
        return general_keep_predicate(instance) and combined_specific_keep_predicate(instance)

    if (general_keep_predicate is None) and (combined_specific_keep_predicate is None):
        return None
    if general_keep_predicate is None:
        return combined_specific_keep_predicate
    if combined_specific_keep_predicate is None:
        return general_keep_predicate
    return combined_general_specific_keep_predicate


def _get_test_keep_instance_predicate(cfg: CfgNode):
    general_keep_predicate = _maybe_create_general_keep_instance_predicate(cfg)
    return general_keep_predicate


def _maybe_filter_and_map_categories(
    dataset_name: str, dataset_dicts: List[Instance]
) -> List[Instance]:
    meta = MetadataCatalog.get(dataset_name)
    category_id_map = meta.thing_dataset_id_to_contiguous_id
    filtered_dataset_dicts = []
    for dataset_dict in dataset_dicts:
        anns = []
        for ann in dataset_dict["annotations"]:
            cat_id = ann["category_id"]
            if cat_id not in category_id_map:
                continue
            ann["category_id"] = category_id_map[cat_id]
            anns.append(ann)
        dataset_dict["annotations"] = anns
        filtered_dataset_dicts.append(dataset_dict)
    return filtered_dataset_dicts


def _add_category_whitelists_to_metadata(cfg: CfgNode):
    for dataset_name, whitelisted_cat_ids in cfg.DATASETS.WHITELISTED_CATEGORIES.items():
        meta = MetadataCatalog.get(dataset_name)
        meta.whitelisted_categories = whitelisted_cat_ids
        logger = logging.getLogger(__name__)
        logger.info(
            "Whitelisted categories for dataset {}: {}".format(
                dataset_name, meta.whitelisted_categories
            )
        )


def _add_category_maps_to_metadata(cfg: CfgNode):
    for dataset_name, category_map in cfg.DATASETS.CATEGORY_MAPS.items():
        category_map = {
            int(cat_id_src): int(cat_id_dst) for cat_id_src, cat_id_dst in category_map.items()
        }
        meta = MetadataCatalog.get(dataset_name)
        meta.category_map = category_map
        logger = logging.getLogger(__name__)
        logger.info("Category maps for dataset {}: {}".format(dataset_name, meta.category_map))


def get_class_to_mesh_name_mapping(cfg):
    return {
        int(class_id): mesh_name
        for class_id, mesh_name in cfg.DATASETS.CLASS_TO_MESH_NAME_MAPPING.items()
    }


def _maybe_add_class_to_mesh_name_map_to_metadata(dataset_names: List[str], cfg: CfgNode):
    for dataset_name in dataset_names:
        meta = MetadataCatalog.get(dataset_name)
        if not hasattr(meta, "class_to_mesh_name"):
            meta.class_to_mesh_name = get_class_to_mesh_name_mapping(cfg)


def _merge_categories(dataset_names: Collection[str]) -> _MergedCategoriesT:
    merged_categories = defaultdict(list)
    category_names = {}
    for dataset_name in dataset_names:
        meta = MetadataCatalog.get(dataset_name)
        whitelisted_categories = meta.get("whitelisted_categories")
        category_map = meta.get("category_map", {})
        cat_ids = (
            whitelisted_categories if whitelisted_categories is not None else meta.categories.keys()
        )
        for cat_id in cat_ids:
            cat_name = meta.categories[cat_id]
            cat_id_mapped = category_map.get(cat_id, cat_id)
            if cat_id_mapped == cat_id or cat_id_mapped in cat_ids:
                category_names[cat_id] = cat_name
            else:
                category_names[cat_id] = str(cat_id_mapped)
            # assign temporary mapped category name, this name can be changed
            # during the second pass, since mapped ID can correspond to a category
            # from a different dataset
            cat_name_mapped = meta.categories[cat_id_mapped]
            merged_categories[cat_id_mapped].append(
                _DatasetCategory(
                    id=cat_id,
                    name=cat_name,
                    mapped_id=cat_id_mapped,
                    mapped_name=cat_name_mapped,
                    dataset_name=dataset_name,
                )
            )
    # second pass to assign proper mapped category names
    for cat_id, categories in merged_categories.items():
        for cat in categories:
            if cat_id in category_names and cat.mapped_name != category_names[cat_id]:
                cat.mapped_name = category_names[cat_id]

    return merged_categories


def _warn_if_merged_different_categories(merged_categories: _MergedCategoriesT):
    logger = logging.getLogger(__name__)
    for cat_id in merged_categories:
        merged_categories_i = merged_categories[cat_id]
        first_cat_name = merged_categories_i[0].name
        if len(merged_categories_i) > 1 and not all(
            cat.name == first_cat_name for cat in merged_categories_i[1:]
        ):
            cat_summary_str = ", ".join(
                [f"{cat.id} ({cat.name}) from {cat.dataset_name}" for cat in merged_categories_i]
            )
            logger.warning(
                f"Merged category {cat_id} corresponds to the following categories: "
                f"{cat_summary_str}"
            )


def combine_detection_dataset_dicts(
    dataset_names: Collection[str],
    keep_instance_predicate: Optional[InstancePredicate] = None,
    proposal_files: Optional[Collection[str]] = None,
) -> List[Instance]:
    """
    Load and prepare dataset dicts for training / testing

    Args:
        dataset_names (Collection[str]): a list of dataset names
        keep_instance_predicate (Callable: Dict[str, Any] -> bool): predicate
            applied to instance dicts which defines whether to keep the instance
        proposal_files (Collection[str]): if given, a list of object proposal files
            that match each dataset in `dataset_names`.
    """
    assert len(dataset_names)
    if proposal_files is None:
        proposal_files = [None] * len(dataset_names)
    assert len(dataset_names) == len(proposal_files)
    # load datasets and metadata
    dataset_name_to_dicts = {}
    for dataset_name in dataset_names:
        dataset_name_to_dicts[dataset_name] = DatasetCatalog.get(dataset_name)
        assert len(dataset_name_to_dicts), f"Dataset '{dataset_name}' is empty!"
    # merge categories, requires category metadata to be loaded
    # cat_id -> [(orig_cat_id, cat_name, dataset_name)]
    merged_categories = _merge_categories(dataset_names)
    _warn_if_merged_different_categories(merged_categories)
    merged_category_names = [
        merged_categories[cat_id][0].mapped_name for cat_id in sorted(merged_categories)
    ]
    # map to contiguous category IDs
    _add_category_id_to_contiguous_id_maps_to_metadata(merged_categories)
    # load annotations and dataset metadata
    for dataset_name, proposal_file in zip(dataset_names, proposal_files):
        dataset_dicts = dataset_name_to_dicts[dataset_name]
        assert len(dataset_dicts), f"Dataset '{dataset_name}' is empty!"
        if proposal_file is not None:
            dataset_dicts = load_proposals_into_dataset(dataset_dicts, proposal_file)
        dataset_dicts = _maybe_filter_and_map_categories(dataset_name, dataset_dicts)
        print_instances_class_histogram(dataset_dicts, merged_category_names)
        dataset_name_to_dicts[dataset_name] = dataset_dicts

    if keep_instance_predicate is not None:
        all_datasets_dicts_plain = [
            d
            for d in itertools.chain.from_iterable(dataset_name_to_dicts.values())
            if keep_instance_predicate(d)
        ]
    else:
        all_datasets_dicts_plain = list(
            itertools.chain.from_iterable(dataset_name_to_dicts.values())
        )
    return all_datasets_dicts_plain


def build_detection_train_loader(cfg: CfgNode, mapper=None):
    """
    A data loader is created in a way similar to that of Detectron2.
    The main differences are:
     - it allows to combine datasets with different but compatible object category sets

    The data loader is created by the following steps:
    1. Use the dataset names in config to query :class:`DatasetCatalog`, and obtain a list of dicts.
    2. Start workers to work on the dicts. Each worker will:
        * Map each metadata dict into another format to be consumed by the model.
        * Batch them by simply putting dicts into a list.
    The batched ``list[mapped_dict]`` is what this dataloader will return.

    Args:
        cfg (CfgNode): the config
        mapper (callable): a callable which takes a sample (dict) from dataset and
            returns the format to be consumed by the model.
            By default it will be `DatasetMapper(cfg, True)`.

    Returns:
        an infinite iterator of training data
    """

    _add_category_whitelists_to_metadata(cfg)
    _add_category_maps_to_metadata(cfg)
    _maybe_add_class_to_mesh_name_map_to_metadata(cfg.DATASETS.TRAIN, cfg)
    dataset_dicts = combine_detection_dataset_dicts(
        cfg.DATASETS.TRAIN,
        keep_instance_predicate=_get_train_keep_instance_predicate(cfg),
        proposal_files=cfg.DATASETS.PROPOSAL_FILES_TRAIN if cfg.MODEL.LOAD_PROPOSALS else None,
    )
    if mapper is None:
        mapper = DatasetMapper(cfg, True)
    return d2_build_detection_train_loader(cfg, dataset=dataset_dicts, mapper=mapper)


def build_detection_test_loader(cfg, dataset_name, mapper=None):
    """
    Similar to `build_detection_train_loader`.
    But this function uses the given `dataset_name` argument (instead of the names in cfg),
    and uses batch size 1.

    Args:
        cfg: a detectron2 CfgNode
        dataset_name (str): a name of the dataset that's available in the DatasetCatalog
        mapper (callable): a callable which takes a sample (dict) from dataset
            and returns the format to be consumed by the model.
            By default it will be `DatasetMapper(cfg, False)`.

    Returns:
        DataLoader: a torch DataLoader, that loads the given detection
            dataset, with test-time transformation and batching.
    """
    _add_category_whitelists_to_metadata(cfg)
    _add_category_maps_to_metadata(cfg)
    _maybe_add_class_to_mesh_name_map_to_metadata([dataset_name], cfg)
    dataset_dicts = combine_detection_dataset_dicts(
        [dataset_name],
        keep_instance_predicate=_get_test_keep_instance_predicate(cfg),
        proposal_files=[
            cfg.DATASETS.PROPOSAL_FILES_TEST[list(cfg.DATASETS.TEST).index(dataset_name)]
        ]
        if cfg.MODEL.LOAD_PROPOSALS
        else None,
    )
    if mapper is None:
        mapper = DatasetMapper(cfg, False)
    return d2_build_detection_test_loader(
        dataset_dicts, mapper=mapper, num_workers=cfg.DATALOADER.NUM_WORKERS
    )


def build_frame_selector(cfg: CfgNode):
    strategy = FrameSelectionStrategy(cfg.STRATEGY)
    if strategy == FrameSelectionStrategy.RANDOM_K:
        frame_selector = RandomKFramesSelector(cfg.NUM_IMAGES)
    elif strategy == FrameSelectionStrategy.FIRST_K:
        frame_selector = FirstKFramesSelector(cfg.NUM_IMAGES)
    elif strategy == FrameSelectionStrategy.LAST_K:
        frame_selector = LastKFramesSelector(cfg.NUM_IMAGES)
    elif strategy == FrameSelectionStrategy.ALL:
        frame_selector = None
    return frame_selector


def build_transform(cfg: CfgNode, data_type: str):
    if cfg.TYPE == "resize":
        if data_type == "image":
            return ImageResizeTransform(cfg.MIN_SIZE, cfg.MAX_SIZE)
    raise ValueError(f"Unknown transform {cfg.TYPE} for data type {data_type}")


def build_combined_loader(cfg: CfgNode, loaders: Collection[Loader], ratios: Sequence[float]):
    images_per_worker = _compute_num_images_per_worker(cfg)
    return CombinedDataLoader(loaders, images_per_worker, ratios)


def build_bootstrap_dataset(dataset_name: str, cfg: CfgNode) -> Sequence[torch.Tensor]:
    """
    Build dataset that provides data to bootstrap on

    Args:
        dataset_name (str): Name of the dataset, needs to have associated metadata
            to load the data
        cfg (CfgNode): bootstrapping config
    Returns:
        Sequence[Tensor] - dataset that provides image batches, Tensors of size
            [N, C, H, W] of type float32
    """
    logger = logging.getLogger(__name__)
    meta = MetadataCatalog.get(dataset_name)
    factory = BootstrapDatasetFactoryCatalog.get(meta.dataset_type)
    dataset = None
    if factory is not None:
        dataset = factory(meta, cfg)
    if dataset is None:
        logger.warning(f"Failed to create dataset {dataset_name} of type {meta.dataset_type}")
    return dataset


def build_data_sampler(cfg: CfgNode):
    if cfg.TYPE == "densepose_uniform":
        data_sampler = PredictionToGroundTruthSampler()
        # transform densepose pred -> gt
        data_sampler.register_sampler(
            "pred_densepose",
            "gt_densepose",
            DensePoseUniformSampler(count_per_class=cfg.COUNT_PER_CLASS),
        )
        data_sampler.register_sampler("pred_densepose", "gt_masks", MaskFromDensePoseSampler())
        return data_sampler
    elif cfg.TYPE == "densepose_UV_confidence":
        data_sampler = PredictionToGroundTruthSampler()
        # transform densepose pred -> gt
        data_sampler.register_sampler(
            "pred_densepose",
            "gt_densepose",
            DensePoseConfidenceBasedSampler(
                confidence_channel="sigma_2",
                count_per_class=cfg.COUNT_PER_CLASS,
                search_proportion=0.5,
            ),
        )
        data_sampler.register_sampler("pred_densepose", "gt_masks", MaskFromDensePoseSampler())
        return data_sampler
    elif cfg.TYPE == "densepose_fine_segm_confidence":
        data_sampler = PredictionToGroundTruthSampler()
        # transform densepose pred -> gt
        data_sampler.register_sampler(
            "pred_densepose",
            "gt_densepose",
            DensePoseConfidenceBasedSampler(
                confidence_channel="fine_segm_confidence",
                count_per_class=cfg.COUNT_PER_CLASS,
                search_proportion=0.5,
            ),
        )
        data_sampler.register_sampler("pred_densepose", "gt_masks", MaskFromDensePoseSampler())
        return data_sampler
    elif cfg.TYPE == "densepose_coarse_segm_confidence":
        data_sampler = PredictionToGroundTruthSampler()
        # transform densepose pred -> gt
        data_sampler.register_sampler(
            "pred_densepose",
            "gt_densepose",
            DensePoseConfidenceBasedSampler(
                confidence_channel="coarse_segm_confidence",
                count_per_class=cfg.COUNT_PER_CLASS,
                search_proportion=0.5,
            ),
        )
        data_sampler.register_sampler("pred_densepose", "gt_masks", MaskFromDensePoseSampler())
        return data_sampler

    raise ValueError(f"Unknown data sampler type {cfg.TYPE}")


def build_data_filter(cfg: CfgNode):
    if cfg.TYPE == "detection_score":
        min_score = cfg.MIN_VALUE
        return ScoreBasedFilter(min_score=min_score)
    raise ValueError(f"Unknown data filter type {cfg.TYPE}")


def build_inference_based_loader(
    cfg: CfgNode, dataset_cfg: CfgNode, model: torch.nn.Module
) -> InferenceBasedLoader:
    """
    Constructs data loader based on inference results of a model.
    """
    dataset = build_bootstrap_dataset(dataset_cfg.DATASET, dataset_cfg.IMAGE_LOADER)
    training_sampler = TrainingSampler(len(dataset))
    data_loader = torch.utils.data.DataLoader(
        dataset,  # pyre-ignore[6]
        batch_size=dataset_cfg.IMAGE_LOADER.BATCH_SIZE,
        sampler=training_sampler,
        num_workers=dataset_cfg.IMAGE_LOADER.NUM_WORKERS,
        collate_fn=trivial_batch_collator,
        worker_init_fn=worker_init_reset_seed,
    )
    return InferenceBasedLoader(
        model,
        data_loader=data_loader,
        data_sampler=build_data_sampler(dataset_cfg.DATA_SAMPLER),
        data_filter=build_data_filter(dataset_cfg.FILTER),
        shuffle=True,
        batch_size=dataset_cfg.INFERENCE.OUTPUT_BATCH_SIZE,
        inference_batch_size=dataset_cfg.INFERENCE.INPUT_BATCH_SIZE,
    )


def has_inference_based_loaders(cfg: CfgNode) -> bool:
    """
    Returns True, if at least one inferense-based loader must
    be instantiated for training
    """
    return len(cfg.BOOTSTRAP_DATASETS) > 0


def build_inference_based_loaders(
    cfg: CfgNode, model: torch.nn.Module
) -> Tuple[List[InferenceBasedLoader], List[float]]:
    loaders = []
    ratios = []
    for dataset_spec in cfg.BOOTSTRAP_DATASETS:
        dataset_cfg = get_bootstrap_dataset_config().clone()
        dataset_cfg.merge_from_other_cfg(CfgNode(dataset_spec))
        loader = build_inference_based_loader(cfg, dataset_cfg, model)
        loaders.append(loader)
        ratios.append(dataset_cfg.RATIO)
    return loaders, ratios


def build_video_list_dataset(meta: Metadata, cfg: CfgNode):
    video_list_fpath = meta.video_list_fpath
    video_base_path = meta.video_base_path
    if cfg.TYPE == "video_keyframe":
        frame_selector = build_frame_selector(cfg.SELECT)
        transform = build_transform(cfg.TRANSFORM, data_type="image")
        video_list = video_list_from_file(video_list_fpath, video_base_path)
        keyframe_helper_fpath = cfg.KEYFRAME_HELPER if hasattr(cfg, "KEYFRAME_HELPER") else None
        return VideoKeyframeDataset(video_list, frame_selector, transform, keyframe_helper_fpath)


class _BootstrapDatasetFactoryCatalog(UserDict):
    """
    A global dictionary that stores information about bootstrapped datasets creation functions
    from metadata and config, for diverse DatasetType
    """

    def register(self, dataset_type: DatasetType, factory: Callable[[Metadata, CfgNode], Dataset]):
        """
        Args:
            dataset_type (DatasetType): a DatasetType e.g. DatasetType.VIDEO_LIST
            factory (Callable[Metadata, CfgNode]): a callable which takes Metadata and cfg
            arguments and returns a dataset object.
        """
        assert dataset_type not in self, "Dataset '{}' is already registered!".format(dataset_type)
        self[dataset_type] = factory


BootstrapDatasetFactoryCatalog = _BootstrapDatasetFactoryCatalog()
BootstrapDatasetFactoryCatalog.register(DatasetType.VIDEO_LIST, build_video_list_dataset)
