# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import itertools
import logging
import numpy as np
from typing import Any, Callable, Collection, Dict, Iterable, List, Optional, Sequence
import torch

from detectron2.config import CfgNode
from detectron2.data.build import (
    build_batch_data_loader,
    load_proposals_into_dataset,
    print_instances_class_histogram,
    trivial_batch_collator,
)
from detectron2.data.catalog import DatasetCatalog, MetadataCatalog
from detectron2.data.common import DatasetFromList, MapDataset
from detectron2.data.samplers import InferenceSampler, RepeatFactorTrainingSampler, TrainingSampler
from detectron2.utils.comm import get_world_size

from .combined_loader import CombinedDataLoader, Loader
from .dataset_mapper import DatasetMapper
from .datasets.coco import DENSEPOSE_KEYS_WITHOUT_MASK as DENSEPOSE_COCO_KEYS_WITHOUT_MASK
from .datasets.coco import DENSEPOSE_MASK_KEY as DENSEPOSE_COCO_MASK_KEY
from .transform import ImageResizeTransform
from .video import (
    FirstKFramesSelector,
    FrameSelectionStrategy,
    LastKFramesSelector,
    RandomKFramesSelector,
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


def _add_category_id_to_contiguous_id_maps_to_metadata(dataset_names: Iterable[str]):
    # merge categories for all datasets
    merged_categories = {}
    for dataset_name in dataset_names:
        meta = MetadataCatalog.get(dataset_name)
        for cat_id, cat_name in meta.categories.items():
            if cat_id not in merged_categories:
                merged_categories[cat_id] = (cat_name, dataset_name)
                continue
            cat_name_other, dataset_name_other = merged_categories[cat_id]
            if cat_name_other != cat_name:
                raise ValueError(
                    f"Incompatible categories for category ID {cat_id}: "
                    f'dataset {dataset_name} value "{cat_name}", '
                    f'dataset {dataset_name_other} value "{cat_name_other}"'
                )

    merged_cat_id_to_cont_id = {}
    for i, cat_id in enumerate(sorted(merged_categories.keys())):
        merged_cat_id_to_cont_id[cat_id] = i

    # add category maps to metadata
    for dataset_name in dataset_names:
        meta = MetadataCatalog.get(dataset_name)
        categories = meta.get("categories")
        meta.thing_classes = [categories[cat_id] for cat_id in sorted(categories.keys())]
        meta.thing_dataset_id_to_contiguous_id = {
            cat_id: merged_cat_id_to_cont_id[cat_id] for cat_id in sorted(categories.keys())
        }
        meta.thing_contiguous_id_to_dataset_id = {
            merged_cat_id_to_cont_id[cat_id]: cat_id for cat_id in sorted(categories.keys())
        }


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
            if all(key in ann for key in DENSEPOSE_COCO_KEYS_WITHOUT_MASK) and (
                (DENSEPOSE_COCO_MASK_KEY in ann) or ("segmentation" in ann)
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
    whitelisted_categories = meta.get("whitelisted_categories")
    category_map = meta.get("category_map", {})
    if whitelisted_categories is None and not category_map:
        return dataset_dicts
    filtered_dataset_dicts = []
    for dataset_dict in dataset_dicts:
        anns = []
        for ann in dataset_dict["annotations"]:
            cat_id = ann["category_id"]
            if whitelisted_categories is not None and cat_id not in whitelisted_categories:
                continue
            ann["category_id"] = category_map.get(cat_id, cat_id)
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
    # load annotations and dataset metadata
    dataset_map = {}
    for dataset_name in dataset_names:
        dataset_dicts = DatasetCatalog.get(dataset_name)
        dataset_map[dataset_name] = dataset_dicts
    # initialize category maps
    _add_category_id_to_contiguous_id_maps_to_metadata(dataset_names)
    # apply category maps
    all_datasets_dicts = []
    for dataset_name, proposal_file in zip(dataset_names, proposal_files):
        dataset_dicts = dataset_map[dataset_name]
        assert len(dataset_dicts), f"Dataset '{dataset_name}' is empty!"
        if proposal_file is not None:
            dataset_dicts = load_proposals_into_dataset(dataset_dicts, proposal_file)
        dataset_dicts = _maybe_filter_and_map_categories(dataset_name, dataset_dicts)
        _map_category_id_to_contiguous_id(dataset_name, dataset_dicts)
        print_instances_class_histogram(
            dataset_dicts, MetadataCatalog.get(dataset_name).thing_classes
        )
        all_datasets_dicts.append(dataset_dicts)

    if keep_instance_predicate is not None:
        all_datasets_dicts_plain = [
            d
            for d in itertools.chain.from_iterable(all_datasets_dicts)
            if keep_instance_predicate(d)
        ]
    else:
        all_datasets_dicts_plain = list(itertools.chain.from_iterable(all_datasets_dicts))
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
    dataset_dicts = combine_detection_dataset_dicts(
        cfg.DATASETS.TRAIN,
        keep_instance_predicate=_get_train_keep_instance_predicate(cfg),
        proposal_files=cfg.DATASETS.PROPOSAL_FILES_TRAIN if cfg.MODEL.LOAD_PROPOSALS else None,
    )
    dataset = DatasetFromList(dataset_dicts, copy=False)

    if mapper is None:
        mapper = DatasetMapper(cfg, True)
    dataset = MapDataset(dataset, mapper)

    sampler_name = cfg.DATALOADER.SAMPLER_TRAIN
    logger = logging.getLogger(__name__)
    logger.info("Using training sampler {}".format(sampler_name))
    if sampler_name == "TrainingSampler":
        sampler = TrainingSampler(len(dataset))
    elif sampler_name == "RepeatFactorTrainingSampler":
        repeat_factors = RepeatFactorTrainingSampler.repeat_factors_from_category_frequency(
            dataset_dicts, cfg.DATALOADER.REPEAT_THRESHOLD
        )
        sampler = RepeatFactorTrainingSampler(repeat_factors)
    else:
        raise ValueError("Unknown training sampler: {}".format(sampler_name))

    return build_batch_data_loader(
        dataset,
        sampler,
        cfg.SOLVER.IMS_PER_BATCH,
        aspect_ratio_grouping=cfg.DATALOADER.ASPECT_RATIO_GROUPING,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
    )


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
    dataset_dicts = combine_detection_dataset_dicts(
        [dataset_name],
        keep_instance_predicate=_get_test_keep_instance_predicate(cfg),
        proposal_files=[
            cfg.DATASETS.PROPOSAL_FILES_TEST[list(cfg.DATASETS.TEST).index(dataset_name)]
        ]
        if cfg.MODEL.LOAD_PROPOSALS
        else None,
    )

    dataset = DatasetFromList(dataset_dicts)
    if mapper is None:
        mapper = DatasetMapper(cfg, False)
    dataset = MapDataset(dataset, mapper)

    sampler = InferenceSampler(len(dataset))
    # Always use 1 image per worker during inference since this is the
    # standard when reporting inference time in papers.
    batch_sampler = torch.utils.data.sampler.BatchSampler(sampler, 1, drop_last=False)

    data_loader = torch.utils.data.DataLoader(
        dataset,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        batch_sampler=batch_sampler,
        collate_fn=trivial_batch_collator,
    )
    return data_loader


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
