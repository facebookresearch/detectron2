import os

from dateutil import parser
from typing import List, Union
from xml.etree import ElementTree
from xml.etree.ElementTree import Element

import cv2
import numpy as np


class MaskAnnotations:
    def __init__(self, image, masks):
        self.image = image
        self.masks = masks


def point_to_attr_string(points):
    ret = ""
    length = points.shape[0]
    for i in range(length):
        point = points[i, 0, :]
        ret = f"{ret}{point[0]},{point[1]};"
    return ret[:-1]  # remove trailing semicolon


def build_polygon_nodes(instances, label_list, epsilon=0.02):
    size = len(instances.scores)
    nodes = []
    for i in range(size):
        label = label_list[instances.pred_classes[i] - 1]
        single_mask = instances.pred_masks[i].cpu().numpy()
        single_mask.dtype = np.uint8
        contours, _ = cv2.findContours(single_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        c = max(contours, key=lambda c: cv2.contourArea(c))
        peri = cv2.arcLength(c, True)
        points = cv2.approxPolyDP(c, peri * epsilon, True)

        nodes.append(ElementTree.Element("polygon", {
            "label": label,
            "occluded": "0",  # TODO
            "points": point_to_attr_string(points),
        }))
        nodes.append(ElementTree.Element("attribute"))
    return nodes


def build_image_node(polygon_nodes, image_id):
    image_node = ElementTree.Element("image")
    image_node.attrib = {
        "id": str(image_id),
    }
    image_node.extend(polygon_nodes)
    return image_node


def mask_to_polygon_xml(anns: List[MaskAnnotations], label_list, epsilon=0.02):
    annotations_node = ElementTree.Element("annotations")
    version = ElementTree.SubElement(annotations_node, "version")
    version.data("1.1")

    for ann in anns:
        image_node = ElementTree.SubElement(annotations_node, "image")
        image_node.attrib = {
            "id": ann.image,
        }
        size = len(ann.masks.scores)
        for i in range(size):
            label = label_list[ann.masks.pred_classes[i]]
            single_mask = ann.masks.pred_masks[i].numpy()
            single_mask.dtype = np.uint8
            _, contours, _ = cv2.findContours(single_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            for c in contours:
                peri = cv2.arcLength(c, True)
                points = cv2.approxPolyDP(contours[0], peri * epsilon, True)

                ElementTree.SubElement(image_node, "polygon", {
                    "name": label,
                    "points": point_to_attr_string(points),
                })


def cvat_xml_to_coco(xml: Union[str, Element], ignore_crowded=True, occluded_as_crowded=False, ignore_attributes=False,
                     cvat_base_dir=""):
    """
    convert cvat xml 1.1 into coco json
    :param xml:
    :param ignore_crowded:  remove crowded annotations
    :param occluded_as_crowded: transfer occluded attributes into crowded
    :param ignore_attributes: do not add attributes into a "attributes" key
    :param cvat_base_dir: path to cvat shared folder
    :return: coco json
    """
    import re
    from pycocotools import coco
    def polygon_points_to_segments(points_str):
        return [float(x) for x in re.split("[,;]", points_str)]

    root: Element
    if isinstance(xml, Element):
        root = xml
    else:
        root = ElementTree.fromstring(xml)
    created_text = root.find("meta/task/created").text
    created = parser.parse(created_text)

    labels = root.find("meta/task/labels")

    categories = []
    category_id_map = {}
    for i, label in enumerate(labels, 1):
        name = label.find("name").text
        categories.append({
            "id": i,
            "name": name,
            "supercategory": "",
        })
        category_id_map[name] = i

    images = []
    annotations = []
    annotation_id = 1
    image_nodes = root.findall("image")
    for image in image_nodes:
        image_id = int(image.get("id"))
        height = int(image.get("height"))
        width = int(image.get("width"))
        images.append({
            "coco_url": "",
            "date_captured": "",
            "flickr_url": "",
            "license": 0,
            "id": image_id,
            "file_name": os.path.join(cvat_base_dir or "", image.get("name")),
            "height": height,
            "width": width,

        })
        for annotation in image:
            assert annotation.tag == "polygon"
            segmentation = polygon_points_to_segments(annotation.get("points"))
            rle = coco.maskUtils.frPyObjects([segmentation], height, width)  # TODO multipart annotation
            if not occluded_as_crowded:
                iscrowd = 0
            else:
                iscrowd = 1 if annotation.get("occluded") == "1" else 0
            ann = {
                "category_id": category_id_map[annotation.get("label")],
                "id": annotation_id,
                "image_id": image_id,
                "iscrowd": iscrowd,
                "segmentation": [segmentation],
                "area": float(coco.maskUtils.area(rle)),
                "bbox": coco.maskUtils.toBbox(rle).tolist()[0],
            }
            if not ignore_attributes:
                attr_dict = {}
                for attr in annotation:
                    attr_dict[attr.get("name")] = attr.text
                    ann["attributions"] = attr_dict
            if ann["iscrowd"] and ignore_crowded:
                continue
            annotations.append(ann)
            annotation_id += 1

    # empty licenses that matches cvat export
    return {

        "licenses": [{"name": "", "id": 0, "url": ""}],
        "info": {
            "contributor": root.find("meta/task/owner/username").text,
            "date_created": created.strftime("%Y-%m-%d"),
            "description": root.find("meta/task/name").text,
            "url": "",
            "version": 3,
            "year": str(created.year)
        },
        "categories": categories,
        "images": images,
        "annotations": annotations,
    }
