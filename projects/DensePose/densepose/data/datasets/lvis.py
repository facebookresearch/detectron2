# Copyright (c) Facebook, Inc. and its affiliates.

# pyre-unsafe
import logging
import os
from typing import Any, Dict, Iterable, List, Optional
from fvcore.common.timer import Timer

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets.lvis import get_lvis_instances_meta
from detectron2.structures import BoxMode
from detectron2.utils.file_io import PathManager

from ..utils import maybe_prepend_base_path
from .coco import (
    DENSEPOSE_ALL_POSSIBLE_KEYS,
    DENSEPOSE_METADATA_URL_PREFIX,
    CocoDatasetInfo,
    get_metadata,
)

DATASETS = [
    CocoDatasetInfo(
        name="densepose_lvis_v1_ds1_train_v1",
        images_root="coco_",
        annotations_fpath="lvis/densepose_lvis_v1_ds1_train_v1.json",
    ),
    CocoDatasetInfo(
        name="densepose_lvis_v1_ds1_val_v1",
        images_root="coco_",
        annotations_fpath="lvis/densepose_lvis_v1_ds1_val_v1.json",
    ),
    CocoDatasetInfo(
        name="densepose_lvis_v1_ds2_train_v1",
        images_root="coco_",
        annotations_fpath="lvis/densepose_lvis_v1_ds2_train_v1.json",
    ),
    CocoDatasetInfo(
        name="densepose_lvis_v1_ds2_val_v1",
        images_root="coco_",
        annotations_fpath="lvis/densepose_lvis_v1_ds2_val_v1.json",
    ),
    CocoDatasetInfo(
        name="densepose_lvis_v1_ds1_val_animals_100",
        images_root="coco_",
        annotations_fpath="lvis/densepose_lvis_v1_val_animals_100_v2.json",
    ),
]


def _load_lvis_annotations(json_file: str):
    """
    Load COCO annotations from a JSON file

    Args:
        json_file: str
            Path to the file to load annotations from
    Returns:
        Instance of `pycocotools.coco.COCO` that provides access to annotations
        data
    """
    from lvis import LVIS

    json_file = PathManager.get_local_path(json_file)
    logger = logging.getLogger(__name__)
    timer = Timer()
    lvis_api = LVIS(json_file)
    if timer.seconds() > 1:
        logger.info("Loading {} takes {:.2f} seconds.".format(json_file, timer.seconds()))
    return lvis_api


def _add_categories_metadata(dataset_name: str) -> None:
    metadict = get_lvis_instances_meta(dataset_name)
    categories = metadict["thing_classes"]
    metadata = MetadataCatalog.get(dataset_name)
    metadata.categories = {i + 1: categories[i] for i in range(len(categories))}
    logger = logging.getLogger(__name__)
    logger.info(f"Dataset {dataset_name} has {len(categories)} categories")


def _verify_annotations_have_unique_ids(json_file: str, anns: List[List[Dict[str, Any]]]) -> None:
    ann_ids = [ann["id"] for anns_per_image in anns for ann in anns_per_image]
    assert len(set(ann_ids)) == len(ann_ids), "Annotation ids in '{}' are not unique!".format(
        json_file
    )


def _maybe_add_bbox(obj: Dict[str, Any], ann_dict: Dict[str, Any]) -> None:
    if "bbox" not in ann_dict:
        return
    obj["bbox"] = ann_dict["bbox"]
    obj["bbox_mode"] = BoxMode.XYWH_ABS


def _maybe_add_segm(obj: Dict[str, Any], ann_dict: Dict[str, Any]) -> None:
    if "segmentation" not in ann_dict:
        return
    segm = ann_dict["segmentation"]
    if not isinstance(segm, dict):
        # filter out invalid polygons (< 3 points)
        segm = [poly for poly in segm if len(poly) % 2 == 0 and len(poly) >= 6]
        if len(segm) == 0:
            return
    obj["segmentation"] = segm


def _maybe_add_keypoints(obj: Dict[str, Any], ann_dict: Dict[str, Any]) -> None:
    if "keypoints" not in ann_dict:
        return
    keypts = ann_dict["keypoints"]  # list[int]
    for idx, v in enumerate(keypts):
        if idx % 3 != 2:
            # COCO's segmentation coordinates are floating points in [0, H or W],
            # but keypoint coordinates are integers in [0, H-1 or W-1]
            # Therefore we assume the coordinates are "pixel indices" and
            # add 0.5 to convert to floating point coordinates.
            keypts[idx] = v + 0.5
    obj["keypoints"] = keypts


def _maybe_add_densepose(obj: Dict[str, Any], ann_dict: Dict[str, Any]) -> None:
    for key in DENSEPOSE_ALL_POSSIBLE_KEYS:
        if key in ann_dict:
            obj[key] = ann_dict[key]


def _combine_images_with_annotations(
    dataset_name: str,
    image_root: str,
    img_datas: Iterable[Dict[str, Any]],
    ann_datas: Iterable[Iterable[Dict[str, Any]]],
):

    dataset_dicts = []

    def get_file_name(img_root, img_dict):
        # Determine the path including the split folder ("train2017", "val2017", "test2017") from
        # the coco_url field. Example:
        #   'coco_url': 'http://images.cocodataset.org/train2017/000000155379.jpg'
        split_folder, file_name = img_dict["coco_url"].split("/")[-2:]
        return os.path.join(img_root + split_folder, file_name)

    for img_dict, ann_dicts in zip(img_datas, ann_datas):
        record = {}
        record["file_name"] = get_file_name(image_root, img_dict)
        record["height"] = img_dict["height"]
        record["width"] = img_dict["width"]
        record["not_exhaustive_category_ids"] = img_dict.get("not_exhaustive_category_ids", [])
        record["neg_category_ids"] = img_dict.get("neg_category_ids", [])
        record["image_id"] = img_dict["id"]
        record["dataset"] = dataset_name

        objs = []
        for ann_dict in ann_dicts:
            assert ann_dict["image_id"] == record["image_id"]
            obj = {}
            _maybe_add_bbox(obj, ann_dict)
            obj["iscrowd"] = ann_dict.get("iscrowd", 0)
            obj["category_id"] = ann_dict["category_id"]
            _maybe_add_segm(obj, ann_dict)
            _maybe_add_keypoints(obj, ann_dict)
            _maybe_add_densepose(obj, ann_dict)
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)
    return dataset_dicts


def load_lvis_json(annotations_json_file: str, image_root: str, dataset_name: str):
    """
    Loads a JSON file with annotations in LVIS instances format.
    Replaces `detectron2.data.datasets.coco.load_lvis_json` to handle metadata
    in a more flexible way. Postpones category mapping to a later stage to be
    able to combine several datasets with different (but coherent) sets of
    categories.

    Args:

    annotations_json_file: str
        Path to the JSON file with annotations in COCO instances format.
    image_root: str
        directory that contains all the images
    dataset_name: str
        the name that identifies a dataset, e.g. "densepose_coco_2014_train"
    extra_annotation_keys: Optional[List[str]]
        If provided, these keys are used to extract additional data from
        the annotations.
    """
    lvis_api = _load_lvis_annotations(PathManager.get_local_path(annotations_json_file))

    _add_categories_metadata(dataset_name)

    # sort indices for reproducible results
    img_ids = sorted(lvis_api.imgs.keys())
    # imgs is a list of dicts, each looks something like:
    # {'license': 4,
    #  'url': 'http://farm6.staticflickr.com/5454/9413846304_881d5e5c3b_z.jpg',
    #  'file_name': 'COCO_val2014_000000001268.jpg',
    #  'height': 427,
    #  'width': 640,
    #  'date_captured': '2013-11-17 05:57:24',
    #  'id': 1268}
    imgs = lvis_api.load_imgs(img_ids)
    logger = logging.getLogger(__name__)
    logger.info("Loaded {} images in LVIS format from {}".format(len(imgs), annotations_json_file))
    # anns is a list[list[dict]], where each dict is an annotation
    # record for an object. The inner list enumerates the objects in an image
    # and the outer list enumerates over images.
    anns = [lvis_api.img_ann_map[img_id] for img_id in img_ids]

    _verify_annotations_have_unique_ids(annotations_json_file, anns)
    dataset_records = _combine_images_with_annotations(dataset_name, image_root, imgs, anns)
    return dataset_records


def register_dataset(dataset_data: CocoDatasetInfo, datasets_root: Optional[str] = None) -> None:
    """
    Registers provided LVIS DensePose dataset

    Args:
        dataset_data: CocoDatasetInfo
            Dataset data
        datasets_root: Optional[str]
            Datasets root folder (default: None)
    """
    annotations_fpath = maybe_prepend_base_path(datasets_root, dataset_data.annotations_fpath)
    images_root = maybe_prepend_base_path(datasets_root, dataset_data.images_root)

    def load_annotations():
        return load_lvis_json(
            annotations_json_file=annotations_fpath,
            image_root=images_root,
            dataset_name=dataset_data.name,
        )

    DatasetCatalog.register(dataset_data.name, load_annotations)
    MetadataCatalog.get(dataset_data.name).set(
        json_file=annotations_fpath,
        image_root=images_root,
        evaluator_type="lvis",
        **get_metadata(DENSEPOSE_METADATA_URL_PREFIX),
    )


def register_datasets(
    datasets_data: Iterable[CocoDatasetInfo], datasets_root: Optional[str] = None
) -> None:
    """
    Registers provided LVIS DensePose datasets

    Args:
        datasets_data: Iterable[CocoDatasetInfo]
            An iterable of dataset datas
        datasets_root: Optional[str]
            Datasets root folder (default: None)
    """
    for dataset_data in datasets_data:
        register_dataset(dataset_data, datasets_root)
