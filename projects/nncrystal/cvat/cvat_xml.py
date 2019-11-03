from typing import List
from xml.etree import ElementTree

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
    return ret[:-1] # remove trailing semicolon


def build_polygon_nodes(instances, label_list, epsilon=0.02):
    size = len(instances.scores)
    nodes = []
    for i in range(size):
        label = label_list[instances.pred_classes[i]-1]
        single_mask = instances.pred_masks[i].cpu().numpy()
        single_mask.dtype = np.uint8
        contours, _ = cv2.findContours(single_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        c = max(contours, key=lambda c: cv2.contourArea(c))
        peri = cv2.arcLength(c, True)
        points = cv2.approxPolyDP(c, peri * epsilon, True)

        nodes.append(ElementTree.Element("polygon", {
            "label": label,
            "occluded": "0", # TODO
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
