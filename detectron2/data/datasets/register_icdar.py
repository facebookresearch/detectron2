# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import copy

from detectron2.data import DatasetCatalog, MetadataCatalog
from .coco import load_coco_json, load_sem_seg
import os,json,cv2
import numpy as np
from detectron2.structures import BoxMode
import itertools
import random

"""
This file contains functions to register a COCO-format dataset to the DatasetCatalog.
"""

# __all__ = ["register_coco_instances", "register_coco_panoptic_separated"]


def get_icdar_dicts(img_dir,train):
    json_file = os.path.join(img_dir, "train_labels.json")
    with open(json_file) as f:
        imgs_anns = json.load(f)

    dataset_dicts = []
#     for _, v in imgs_anns.items():
#     print(imgs_anns.keys())
    imgs_anns_list=list(imgs_anns.keys())
    imgs_anns_list.remove('gt_3068');
    imgs_anns_list.remove('gt_941');
    imgs_anns_list.remove('gt_2440');
    imgs_anns_list.remove('gt_5194');
    imgs_anns_list.remove('gt_512');
    imgs_anns_list.remove('gt_626');
    imgs_anns_list.remove('gt_1433');
    imgs_anns_list.remove('gt_41');
    imgs_anns_list.remove('gt_1491');
    imgs_anns_list.remove('gt_3792');
    imgs_anns_list.remove('gt_2805');
    imgs_anns_list.remove('gt_4639');
    for each_key in imgs_anns_list:
        v=imgs_anns[each_key]
        record = {}
        
        filename = os.path.join(img_dir,'train_images', '%s.jpg'%each_key)
#         print(filename)
        height, width = cv2.imread(filename).shape[:2]
        
        record["file_name"] = filename
        record["height"] = height
        record["width"] = width
        objs = []
        for  anno in v:
            px = np.array(anno['points'])[:,0]
            py = np.array(anno['points'])[:,1]
            poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
            poly = list(itertools.chain.from_iterable(poly))

            obj = {
                "bbox": [np.min(px), np.min(py), np.max(px), np.max(py)],
                "bbox_mode": BoxMode.XYXY_ABS,
                "segmentation": [poly],
                "category_id": 0,
                "iscrowd": 0
            }
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)
    random.seed(1)
    random.shuffle(dataset_dicts)
    train_num=int(len(dataset_dicts)*0.8)
    if train:
        return dataset_dicts[:train_num]
    else:
        with open('val.txt','w+') as fr:
            for each in dataset_dicts[train_num:]:
                fr.write(each['file_name'])
                fr.write('\n')
        return dataset_dicts[train_num:]
    
def register_icdar_instances():
    """
    Register a dataset in COCO's json annotation format for
    instance detection, instance segmentation and keypoint detection.
    (i.e., Type 1 and 2 in http://cocodataset.org/#format-data.
    `instances*.json` and `person_keypoints*.json` in the dataset).

    This is an example of how to register a new dataset.
    You can do something similar to this function, to register new datasets.

    Args:
        name (str): the name that identifies a dataset, e.g. "coco_2014_train".
        metadata (dict): extra metadata associated with this dataset.  You can
            leave it as an empty dict.
        json_file (str): path to the json instance annotation file.
        image_root (str): directory which contains all the images.
    """
    from detectron2.data import DatasetCatalog, MetadataCatalog
    print('*'*100,'register_icdar')
    # for d in ["train", "val"]:
    # dataset_dicts=get_icdar_dicts('/input0')
    d='train'
    DatasetCatalog.register("icdar_" + d, lambda d=d: get_icdar_dicts('/input0',True))
    MetadataCatalog.get("icdar_" + d).set(thing_classes=["text"])
    d='val'
    DatasetCatalog.register("icdar_" + d, lambda d=d:get_icdar_dicts('/input0',False) )
    MetadataCatalog.get("icdar_" + d).set(thing_classes=["text"])
    
    json_file='/input0/train_labels.json'
    image_root='/input0/train_images'
    metadata=None
    MetadataCatalog.get('icdar_'+d).set(
        json_file=json_file, image_root=image_root, evaluator_type="icdar"
    )


