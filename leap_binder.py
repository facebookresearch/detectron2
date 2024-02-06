import json
import os
from typing import List, Union, Dict
import cv2
import numpy as np
import detectron2.data.transforms as T
from code_loader.contract.enums import LeapDataType
from detectron2.data.detection_utils import read_image

from pycocotools.coco import COCO

from torch import as_tensor
# Tensorleap imports
from code_loader import leap_binder
from code_loader.contract.datasetclasses import PreprocessResponse

from effizency.config import CONFIG
from effizency.metrics.detectron2_loss import calc_rpn_loss, calc_roi_losses, calc_detectron2_loss
from effizency.utils.gcs_utils import download_gcs
from effizency.utils.general_utils import get_bboxes, is_faulty_mask
from effizency.utils.loss_utils import polygons_to_bitmask
from effizency.utils.visualization_utils import polygons_to_mask
from effizency.visualizers import bb_gt_visualizer


# Preprocess Function
def preprocess_func() -> List[PreprocessResponse]:
    train_images = download_gcs(CONFIG['train_file'])
    with open(train_images, 'r') as f:
        train_list_of_paths = f.read().split("\n")

    val_images = download_gcs(CONFIG['val_file'])
    with open(val_images, 'r') as f:
        val_list_of_paths = f.read().split("\n")

    train_size = min(len(train_list_of_paths), CONFIG['train_size'])
    val_size = min(len(val_list_of_paths), CONFIG['val_size'])

    train_list_of_paths = [os.path.join('train', p) for p in train_list_of_paths[:train_size]]
    val_list_of_paths = [os.path.join('val', p) for p in val_list_of_paths[:val_size]]

    train = PreprocessResponse(length=len(train_list_of_paths), data={'images': train_list_of_paths})
    val = PreprocessResponse(length=len(val_list_of_paths), data={'images': val_list_of_paths})
    response = [train, val]
    return response


# Input and GT Encoders
def input_encoder(idx: int, preprocess: PreprocessResponse) -> np.ndarray:
    gcs_img_path = preprocess.data['images'][idx]
    local_img_path = download_gcs(gcs_img_path)
    img = read_image(local_img_path, format=CONFIG['image_format'])
    aug = T.ResizeShortestEdge(CONFIG['image_size'], CONFIG['max_size_test'])
    img = aug.get_transform(img).apply_image(img)
    img = img - CONFIG['pixel_means']
    img = img.astype("float32")
    img = np.asarray(img)
    return img


def raw_image_encoder(idx: int, preprocess: PreprocessResponse) -> np.ndarray:
    gcs_img_path = preprocess.data['images'][idx]
    local_img_path = download_gcs(gcs_img_path)
    img = read_image(local_img_path, format=CONFIG['image_format'])
    img = img.astype("float32")
    img = np.asarray(img)
    return img


def get_anns(idx: int, preprocessing: PreprocessResponse) -> Dict[str, Union[str, int, List[Dict]]]:
    gcs_file_path = preprocessing.data['images'][idx].replace('png', 'json')
    local_file_path = download_gcs(gcs_file_path)
    with open(local_file_path, 'r') as f:
        anns = json.load(f)
    return anns


def original_image_shape_encoder(idx: int, preprocessing: PreprocessResponse) -> np.ndarray:
    anns = get_anns(idx, preprocessing)
    image_shape = np.asarray([anns['imageHeight'], anns['imageWidth']])
    return image_shape


def bbox_gt_encoder(idx: int, preprocessing: PreprocessResponse) -> np.ndarray:
    # bboxes are in xywh format and normalized to original image size
    anns = get_anns(idx, preprocessing)
    bboxes = get_bboxes(anns)
    return bboxes.astype(np.float32)


def polygons_gt_encoder(idx: int, preprocessing: PreprocessResponse) -> np.ndarray:
    anns = get_anns(idx, preprocessing)
    instances_anns = anns['shapes']
    polygons = []
    for m in instances_anns:
        flatten_poly = np.asarray(m['points']).flatten()
        poly = np.pad(flatten_poly,
                      (0, CONFIG['max_polygon_length'] - flatten_poly.shape[0]),
                      mode='constant', constant_values=-1)
        polygons.append(poly)
    return np.asarray(polygons, dtype=np.float32)


def masks_gt_encoder(idx: int, preprocessing: PreprocessResponse) -> np.ndarray:
    anns = get_anns(idx, preprocessing)
    instances_anns = anns['shapes']
    mask = [polygons_to_mask(p['points'], anns['imageHeight'], anns['imageWidth']) for p in instances_anns]
    return np.asarray(mask)


# Metadata
def n_open_polygons(polygons: np.ndarray, image_width: int, image_height: int) -> int:
    open_poly = []
    for poly in polygons:
        # remove padding of -1
        poly = poly[poly != -1]
        mask = polygons_to_bitmask([poly], image_height, image_width)
        open_poly.append(is_faulty_mask(mask))
    return sum(open_poly)


def n_small_bboxes(bboxes: np.ndarray) -> int:
    areas = bboxes[:, 2] * bboxes[:, 3]
    n_small_bboxes = sum((areas < CONFIG['small_bb_threshold']).astype(np.int8))
    return int(n_small_bboxes)


def get_mask_areas(masks: np.ndarray) -> float:
    image_area = masks.shape[1] * masks.shape[2]
    areas = np.asarray([np.sum(mask) for mask in masks]) / image_area
    return areas


def get_metadata_dict(idx: int, preprocessing: PreprocessResponse) -> Dict[str, Union[str, int, float]]:
    # get gt
    anns = get_anns(idx, preprocessing)
    polygons = polygons_gt_encoder(idx, preprocessing)
    bboxes = bbox_gt_encoder(idx, preprocessing)
    masks = masks_gt_encoder(idx, preprocessing)

    mask_areas = get_mask_areas(masks)

    metadata = {
        'file_name': anns['imagePath'],
        'original_height': anns['imageHeight'],
        'original_width': anns['imageWidth'],
        'n_faulty_polygons': n_open_polygons(polygons, anns['imageWidth'], anns['imageHeight']),
        'n_small_bboxes': n_small_bboxes(bboxes),
        'largest_mask_area': np.max(mask_areas),
        'smallest_mask_area': np.min(mask_areas),
        'mean_mask_area': np.mean(mask_areas),
    }
    return metadata


# Bind the functions to the LeapBinder
# preprocess
leap_binder.set_preprocess(function=preprocess_func)
# set input and gt encoders
leap_binder.set_input(function=input_encoder, name='image')
leap_binder.set_input(function=raw_image_encoder, name='raw_image')
leap_binder.set_ground_truth(function=bbox_gt_encoder, name='bboxes')
leap_binder.set_ground_truth(function=polygons_gt_encoder, name='polygons')
leap_binder.set_ground_truth(function=masks_gt_encoder, name='masks')
leap_binder.set_ground_truth(function=original_image_shape_encoder, name='original_image_shape')
# set metadata
leap_binder.set_metadata(function=get_metadata_dict, name='metadata')
# set custom losses
leap_binder.add_custom_loss(function=calc_detectron2_loss, name='Detectron2 Loss')
# set custom metrics
leap_binder.add_custom_metric(function=calc_rpn_loss, name='RPN Loss Components')
leap_binder.add_custom_metric(function=calc_roi_losses, name='ROI Loss Components')
# set custom visualizers
leap_binder.set_visualizer(function=bb_gt_visualizer, name='Ground Truth Bounding Boxes',
                           visualizer_type=LeapDataType.ImageWithBBox)

if __name__ == '__main__':
    leap_binder.check()
