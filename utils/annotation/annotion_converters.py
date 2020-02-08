import os
import math
import numpy as np
import json

from collections import Counter
from typing import Tuple
from utils.annotation.annotation_interfaces import DetectionAnnotations


class RotatedBBoxConverter:
    """
    Rotated Box directly annotated as Following
    in that case doesn't have any theta information that
    how many rotated degree from origin horizontal bounding box

    - Rotated bounding Box Annotation example
        +(P2)

                           +(P3)


    +(P1)

                        +(P4)

    - Origin horizontal bounding box example
    +(P1)                   +(P2)





    +(P4)                   +(P3)

    So, main logic approximation theta value from angle of x-axis in cartesian coordinate system
    with vector as (P1 - P2)
    """
    def rotate_horizon_bbox_with_theta(self, points: np.ndarray, theta: float) -> np.ndarray:
        """
        Args:
            points  (np.ndarray) : four points in rotated bbox. bbox dim is 3x4 or 4x3
                                   each point coordinates is homogeneous coordinates
                                   points order as following [[p1_x, p1_y, 1],
                                                              [p2_x, p2_y, 1],
                                                              [p3_x, p3_y, 1],
                                                              [p4_x, p4_y, 1]]

            theta   (float) : rotate value as radian

        Returns:
            (np.ndarray) : four points in rotated bbox. bbox dim is 3x4 or 4x3
                           each point coordinates is homogeneous coordinates
                           points order as following [[p1_x, p1_y, 1],
                                                      [p2_x, p2_y, 1],
                                                      [p3_x, p3_y, 1],
                                                      [p4_x, p4_y, 1]]
        """

        bbox_center_points = self.get_origin_point(points)
        whiten_points = self.whitening(points, bbox_center_points)
        whiten_rotated_bbox_points = self.rotate(whiten_points, -theta)
        rotated_bbox_points = self.inverse_whitening(whiten_rotated_bbox_points, bbox_center_points)

        return rotated_bbox_points

    def get_rotated_bbox(self, points: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Args:
            points  (np.ndarray) : four points in rotated bbox. bbox dim is 3x4 or 4x3
                                   each point coordinates is homogeneous coordinates
                                   points order as following [[p1_x, p1_y, 1],
                                                              [p2_x, p2_y, 1],
                                                              [p3_x, p3_y, 1],
                                                              [p4_x, p4_y, 1]]
        Returns:
            (np.ndarray, float) : ([xmin, ymin, xmax, ymax], theta)
        """
        return self.get_horizon_bbox(points), self.get_theta(points)

    def get_horizon_bbox(self, points: np.ndarray) -> np.ndarray:
        """

        Args:
            points  (np.ndarray) : four points in rotated bbox. bbox dim is 3x4 or 4x3
                                   each point coordinates is homogeneous coordinates
                                   points order as following [[p1_x, p1_y, 1],
                                                              [p2_x, p2_y, 1],
                                                              [p3_x, p3_y, 1],
                                                              [p4_x, p4_y, 1]]

        Returns:
            (np.ndarray) : [xmin, ymin, xmax, ymax]
        """
        bbox_center_points = self.get_origin_point(points)
        whiten_points = self.whitening(points, bbox_center_points)

        p1_p2_vector = whiten_points[0] - whiten_points[1]
        x_axis_vector = np.array([1, 0, 0])
        theta = math.radians(180. - math.degrees(self.get_angle_btw_vectors(p1_p2_vector,
                                                                            x_axis_vector)))

        whiten_horizon_bbox_points = self.rotate(whiten_points, -theta)
        horizon_bbox_points = self.inverse_whitening(whiten_horizon_bbox_points, bbox_center_points)

        return self.points_to_bbox(horizon_bbox_points)

    def get_theta(self, points: np.ndarray) -> float:
        """

        Args:
            points  (np.ndarray) : four points in rotated bbox. bbox dim is 3x4 or 4x3
                                   each point coordinates is homogeneous coordinates
                                   points order as following [[p1_x, p1_y, 1],
                                                              [p2_x, p2_y, 1],
                                                              [p3_x, p3_y, 1],
                                                              [p4_x, p4_y, 1]]

        Returns:
            (float) : theta values. theta value is radian
        """

        bbox_center_points = self.get_origin_point(points)
        whiten_points = self.whitening(points, bbox_center_points)

        p1_p2_vector = whiten_points[0] - whiten_points[1]
        x_axis_vector = np.array([1, 0, 0])
        return math.radians(180. - math.degrees(self.get_angle_btw_vectors(p1_p2_vector,
                                                                           x_axis_vector)))

    def whitening(self, points: np.ndarray, center_point: np.ndarray) -> np.ndarray:
        """
        change origin point in cartesian coordinate system based on given origin point

        Args:
            points  (np.ndarray) : four points in rotated bbox. bbox dim is 3x4 or 4x3
                                   each point coordinates is homogeneous coordinates
                                   points order as following [[p1_x, p1_y, 1],
                                                              [p2_x, p2_y, 1],
                                                              [p3_x, p3_y, 1],
                                                              [p4_x, p4_y, 1]]

            center_point    (np.ndarray) : [x, y] in cartesian coordinate system

        Returns
            (np.ndarray) : four whiten rotated rectangle points as 3 dim
        """
        return self.shift(points, px=-center_point[0], py=-center_point[1])

    def inverse_whitening(self, points: np.ndarray, center_point: np.ndarray) -> np.ndarray:
        return self.whitening(points=points, center_point=-center_point)

    @staticmethod
    def shift(source: np.ndarray, px: float, py: float) -> np.ndarray:
        """
        Apply Shift Matrix given source matrix.

        shift matrix has a 3x3 dimension.

        Args:
            source  (np.ndarray): at least, row or culumn dim should be 3.
            px      (float): shift value to x-axis
            py      (float): shift value to y-axis

        Returns:
            (np.ndarray): result of shift matrix
        """
        shift_matrix = np.array([[1, 0, 0],
                                 [0, 1, 0],
                                 [px, py, 1]])

        return np.dot(source, shift_matrix)

    def get_origin_point(self, points: np.ndarray):
        """

        Args:
            points  (np.ndarray) : four points in rotated bbox. bbox dim is 3x4 or 4x3
                                   each point coordinates is homogeneous coordinates
                                   points order as following [[p1_x, p1_y, 1],
                                                              [p2_x, p2_y, 1],
                                                              [p3_x, p3_y, 1],
                                                              [p4_x, p4_y, 1]]

        Returns:
            (np.ndarray) : [x, y]
        """
        rect: np.ndarray = self.points_to_bbox(points)
        center_point = self.get_center_point(rect)

        xmin = rect[0]
        ymin = rect[1]

        return np.array([xmin + center_point[0], ymin + center_point[1]])

    @staticmethod
    def points_to_bbox(points: np.ndarray) -> np.ndarray:
        """

        Args:
            points  (np.ndarray) : four points in rotated bbox. bbox dim is 3x4 or 4x3
                                   each point coordinates is homogeneous coordinates
                                   points order as following [[p1_x, p1_y, 1],
                                                              [p2_x, p2_y, 1],
                                                              [p3_x, p3_y, 1],
                                                              [p4_x, p4_y, 1]]

        Returns:
            (np.ndarray) : [xmin, ymin, xmax, ymax]
        """

        x_candidate = []
        y_candidate = []

        for point in points:
            x = point[0]
            y = point[1]

            x_candidate.append(x)
            y_candidate.append(y)

        xmin = min(x_candidate)
        ymin = min(y_candidate)
        xmax = max(x_candidate)
        ymax = max(y_candidate)

        return np.array([xmin, ymin, xmax, ymax])

    @staticmethod
    def bbox_to_points(bbox: np.ndarray) -> np.ndarray:
        """

        Args:
            bbox    (np.ndarray) : [xmin, ymin, xmax, ymax]

        Returns:
            (np.ndarray) : [[p1_x, p1_y, 1],
                            [p2_x, p2_y, 1],
                            [p3_x, p3_y, 1],
                            [p4_x, p4_y, 1]]
        """
        xmin = bbox[0]
        ymin = bbox[1]
        xmax = bbox[2]
        ymax = bbox[3]

        points = np.array([[xmin, ymin, 1],
                           [xmax, ymin, 1],
                           [xmax, ymax, 1],
                           [xmin, ymax, 1]])

        return points

    @staticmethod
    def get_center_point(rect: np.ndarray):
        """

        Args:
            rect    (np.ndarray) : [xmin, ymin, xmax, ymax]
        Returns:
            (np.ndarray) : [x, y]
        """

        xmin = rect[0]
        ymin = rect[1]
        xmax = rect[2]
        ymax = rect[3]

        return np.array([(xmax - xmin) / 2, (ymax - ymin) / 2])

    @staticmethod
    def rotate(source: np.ndarray, theta: float) -> np.ndarray:
        """
        Clock-Wise Rotate

        Args:
            source  (np.ndarray): source matrix as dim is 3
            theta   (float): angle represented radian

        Returns:
            (np.ndarray): rotated matrix as dim is 3
        """
        c, s = np.cos(theta), np.sin(theta)
        rotate_matrix = np.array(((c,   -s, 0),
                                  (s,   c,  0),
                                  (0,   0,  1)))

        return np.dot(source, rotate_matrix)

    @staticmethod
    def get_angle_btw_vectors(u: np.ndarray, v: np.ndarray):
        return np.arccos(np.dot(u, v) / (np.linalg.norm(u, ord=2) * np.linalg.norm(v, ord=2)))


class COCOConverter:

    def __init__(self, anno: DetectionAnnotations):
        self.anno = anno

        self.info = {"year": 2020,
                     "version": "1.0",
                     "description": "Ariel ship Dataset for Rotated Box Detection at EO-Detection Challenge ",
                     "contributor": "Martin Hwang",
                     "url": "https://newfront.dacon.io/competitions/official/235492/overview/description/",
                     "date_created": "2020"
                     }

        self.licenses = [{"id": 1,
                          "name": "Copyright Ministry of National Defense. All rights reserved",
                          "url": "http://www.mnd.go.kr"
                          }]

        self.type = "instances"
        self.categories = [{"id": idx, "name": category, "supercategory": "None"} for idx, category in enumerate(self._collect_class_info())]
        self.cat2id = {cat["name"]: cat_id for cat_id, cat in enumerate(self.categories)}

        self.json_data = self.parser_annotation(anno)

    def _collect_class_info(self):
        objs = sum([FILE.OBJECTS for FILE in self.anno.FILES], [])
        classes_info = Counter([obj.CLASS for obj in objs])
        classes_list = list(classes_info.keys())

        return classes_list

    def parser_annotation(self, anno: DetectionAnnotations):
        images = []
        annotations = []

        for idx, FILE in enumerate(anno.FILES):
            # Images
            license = 1
            file_name = FILE.FILEPATH.split("/")[-1]
            coco_url = ""
            height = FILE.IMAGE_HEIGHT
            width = FILE.IMAGE_WIDTH
            date_captured = ""
            flickr_url = ""
            id = idx

            for OBJ in FILE.OBJECTS:
                # Annotations
                segmentation = []
                area = 0
                iscrowd = 0

                x = OBJ.XMIN
                y = OBJ.YMIN
                w = OBJ.XMAX - OBJ.XMIN
                h = OBJ.YMAX - OBJ.YMIN
                theta = OBJ.THETA
                bbox = [x, y, w, h, theta]
                category_id = self.cat2id[OBJ.CLASS]
                image_id = id

                annotations.append({"segmentation": segmentation,
                                    "area": area,
                                    "iscrowd": iscrowd,
                                    "image_id": image_id,
                                    "bbox": bbox,
                                    "category_id": category_id,
                                    "id": 0})

            images.append({"license": license,
                           "file_name": file_name,
                           "coco_url": coco_url,
                           "height": height,
                           "width": width,
                           "date_captured": date_captured,
                           "flickr_url": flickr_url,
                           "id": id})

        json_data = {"info": self.info,
                     "images": images,
                     "licenses": self.licenses,
                     "type": self.type,
                     "annotations": annotations,
                     "categories": self.categories}

        return json_data

    def save(self, filenpath: str):
        with open(filenpath, "w") as jsonfile:
            json.dump(self.json_data, jsonfile, sort_keys=True, indent=4)


if __name__ == "__main__":
    from script.read_eo_dataset import read_eo_dataset

    np.set_printoptions(formatter={"float_kind": lambda x: "{0:0.3f}".format(x)})
    rbox_cvt = RotatedBBoxConverter()

    # Rotate Test
    horizon_bbox = np.array([[-1,   2,  0],
                             [1,    2,  0],
                             [-1,   -2, 0],
                             [1,    -2, 0]])
    theta = math.radians(45)
    result = rbox_cvt.rotate(horizon_bbox, theta)

    # [[0.707 2.121 0.000]
    #  [2.121 0.707 0.000]
    #  [-2.121 -0.707 0.000]
    #  [-0.707 -2.121 0.000]]
    print(result)

    # get_angle_btw_vectors Test
    x_axis = np.array([1, 0, 0])
    vec_45 = np.array([1, 1, 0])
    theta = rbox_cvt.get_angle_btw_vectors(vec_45, x_axis)
    # 45
    print(math.degrees(theta))

    annotations = read_eo_dataset()
    detection_anno = DetectionAnnotations(annotations)
    coco_converter = COCOConverter(detection_anno)
    coco_converter.save("eo-dataset.json")
