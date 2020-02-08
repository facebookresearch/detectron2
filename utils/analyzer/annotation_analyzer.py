import itertools
import math
import matplotlib.pyplot as plt
import numpy as np

from collections import Counter
from sklearn.linear_model import LinearRegression
from typing import List, Dict
from utils.annotation.annotation_interfaces import DetectionAnnotations
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
from pyclustering.cluster.kmeans import kmeans
from pyclustering.utils.metric import type_metric, distance_metric
from pyclustering.cluster import cluster_visualizer


class _BaseAnalyzer:
    def __init__(self, anno: DetectionAnnotations) -> None:
        """
        Base Analyzer Component

        Args:
            anno (DetectionAnnotations) : DetectionAnnotations object
        """
        self.anno = anno
        self.classes_list = self._collect_class_info()
        self.analytic_result = {}

    def _collect_class_info(self):
        objs = sum([FILE.OBJECTS for FILE in self.anno.FILES], [])
        classes_info = Counter([obj.CLASS for obj in objs])
        classes_list = list(classes_info.keys())

        return classes_list

    def fit(self):
        """
        fit function should save analysis result to `self.analytic_result`
        """
        raise NotImplementedError


class ClassDistributionAnalyzer(_BaseAnalyzer):
    def __init__(self, anno: DetectionAnnotations) -> None:
        """
        Class Distribution Analyzer

        Args:
            anno (DetectionAnnotations) : DetectionAnnotations object
        """
        super(ClassDistributionAnalyzer, self).__init__(anno)

    def fit(self) -> None:
        objs = sum([FILE.OBJECTS for FILE in self.anno.FILES], [])
        classes_info = Counter([obj.CLASS for obj in objs])

        self.analytic_result["classes_list"] = list(classes_info.keys())
        self.analytic_result["classes_frequency"] = list(classes_info.values())


class AspectRatioAnalyzer(_BaseAnalyzer):
    def __init__(self, anno: DetectionAnnotations) -> None:
        """
        Aspect Ratio Analyzer.
        * Aspect Ratio = width / height

        Args:
            anno        (DetectionAnnotations) : DetectionAnnotations object
        """
        super(AspectRatioAnalyzer, self).__init__(anno)

    def fit(self):

        for classes in self.classes_list:
            self.analytic_result.update({classes: {"data": []}})

        for FILE in self.anno.FILES:

            for OBJ in FILE.OBJECTS:
                classes = OBJ.CLASS
                xmin = OBJ.XMIN
                ymin = OBJ.YMIN
                xmax = OBJ.XMAX
                ymax = OBJ.YMAX

                width = xmax - xmin
                height = ymax - ymin
                aspect_ratio = width / height

                self.analytic_result[classes]["data"].append(aspect_ratio)

        for classes in self.classes_list:
            hist, bin_edges = np.histogram(np.array(self.analytic_result[classes]["data"]),
                                           density=True)
            self.analytic_result[classes]["hist"] = hist
            self.analytic_result[classes]["bins"] = bin_edges[:-1]


class ScaleDistributionAnalyzer(_BaseAnalyzer):
    def __init__(self, anno: DetectionAnnotations) -> None:
        """
        Scale Distribution Analyzer

        Args:
            anno (DetectionAnnotations) : DetectionAnnotations object
        """
        super(ScaleDistributionAnalyzer, self).__init__(anno)

    def fit(self):

        for classes in self.classes_list:
            self.analytic_result.update({classes: {"data": []}})

        for FILE in self.anno.FILES:

            for OBJ in FILE.OBJECTS:
                classes = OBJ.CLASS
                xmin = OBJ.XMIN
                ymin = OBJ.YMIN
                xmax = OBJ.XMAX
                ymax = OBJ.YMAX

                width = xmax - xmin
                height = ymax - ymin
                scale = width * height

                self.analytic_result[classes]["data"].append(scale)

        for classes in self.classes_list:
            hist, bin_edges = np.histogram(np.array(self.analytic_result[classes]["data"]))
            self.analytic_result[classes]["hist"] = hist
            self.analytic_result[classes]["bins"] = bin_edges[:-1]


class ThetaDistributionAnalyzer(_BaseAnalyzer):
    def __init__(self, anno: DetectionAnnotations) -> None:
        """
        Theta Distribution Analyzer

        Args:
            anno (DetectionAnnotations) : DetectionAnnotations object
        """
        super(ThetaDistributionAnalyzer, self).__init__(anno)

    def fit(self):

        for classes in self.classes_list:
            self.analytic_result.update({classes: {"data": []}})

        for FILE in self.anno.FILES:

            for OBJ in FILE.OBJECTS:
                classes = OBJ.CLASS
                degree = -math.degrees(OBJ.THETA)

                self.analytic_result[classes]["data"].append(degree)

        for classes in self.classes_list:
            hist, bin_edges = np.histogram(np.array(self.analytic_result[classes]["data"]))
            self.analytic_result[classes]["hist"] = hist
            self.analytic_result[classes]["bins"] = bin_edges[:-1].astype(np.int)


class BBoxDistributionAnalyzer:

    def __init__(self, anno: DetectionAnnotations, distance_measure: str = 'l2'):
        """
        Args:
            anno (DetectionAnnotations): DetectionAnnotations object
            distance_measure (str): similarity measure
        """

        classes_list = self._collect_classes(anno)
        self.anno = anno
        self.classes_list = classes_list
        self.bbox_data = self._collect_bbox(anno, classes_list)
        self.metric = None
        self.distance_measure = distance_measure
        self.similarity_matrix = None
        self.analysis_result = None

        if distance_measure == "l1":
            self.metric = self._manhattan_distance
        elif distance_measure == "l2":
            self.metric = self._euclidean_distance
        elif distance_measure == "inner_product":
            self.metric = self._inner_product
        elif distance_measure == "cosine_similarity":
            self.metric = self._cosine_similarity
        else:
            raise RuntimeError("Not supported {} metric".format(distance_measure))

    def fit(self):
        self.analysis_result = self._distribution_modeling(self.classes_list, self.bbox_data)
        self.similarity_matrix = self._similarity_intra_classes(self.analysis_result)

    def show_similarity_matrix(self, is_save: bool = False):
        """
        Display and save figure about similarity matrix
        Args:
            (bool) is_save : if True, save figure
                                False, not save figure
        Returns:
            (None)
        """
        if self.similarity_matrix is None:
            self.fit()

        similarity_matrix = self.similarity_matrix.astype('float')

        plt.figure()
        plt.imshow(similarity_matrix, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title("Similarity Matrix - {}".format(self.distance_measure))
        plt.colorbar()
        tick_mark = np.arange(len(self.classes_list))
        plt.xticks(tick_mark, self.classes_list, rotation=45)
        plt.yticks(tick_mark, self.classes_list)

        fmt = '.2f'
        thresh = similarity_matrix.max() / 2.
        for i, j in itertools.product(range(similarity_matrix.shape[0]),
                                      range(similarity_matrix.shape[1])):
            plt.text(j, i, format(similarity_matrix[i, j], fmt), horizontalalignment="center",
                     color="white" if similarity_matrix[i, j] > thresh else "black")

        plt.ylabel('Classes')
        plt.xlabel('Classes')
        if is_save:
            plt.savefig("BBox_Distribution_Similarity_Matrix-{}.png".format(self.distance_measure))
        plt.show()

    def report_modeling(self, is_save=False):
        """
        Display and save figure about each class box distribution and linear regression result
        Args:
            (bool) is_save : if True, save figure
                                False, not save figure
        Returns:
            (None)
        """

        float_subplotsize = np.sqrt(len(self.classes_list))
        floor_subplotsize = np.floor(float_subplotsize)

        if (float_subplotsize - floor_subplotsize) != 0:
            floor_subplotsize += 1

        subplotsize = [int(floor_subplotsize) for _ in range(2)]

        plt.figure()
        for idx, class_label in enumerate(self.classes_list):

            slope = self.analysis_result[class_label]['slope']
            bias = self.analysis_result[class_label]['bias']

            x = np.arange(0, 1, 0.1)
            y = list()
            for _x in x:
                y.append(bias + (slope * _x))

            plt.subplot(subplotsize[0], subplotsize[1], idx + 1)
            bbox_reshape = np.transpose(np.asarray(self.bbox_data[class_label]))
            plt.scatter(bbox_reshape[0], bbox_reshape[1], label=class_label)
            plt.plot(x, y, color='r')
            plt.xlim([0, 1])
            plt.ylim([0, 1])
            plt.legend()
            plt.title(class_label)

        if is_save is True:
            plt.savefig("each_Classes.png")
        plt.show()

    @staticmethod
    def _distribution_modeling(classes_list, bbox_data) -> Dict:
        """
        Args:
            classes_list (List) : classes name
            bbox_data (Dict) : bbox information as each classes
        Returns:
            (Dict) : result of Linear Regression as following
                    {
                        "classes" : {
                                        "score": (List), coefficient of determination
                                        "slope": (List),
                                        "bias": (List),
                                        "vec": (List), (x, y) vector when x is 1.0
                            }
                        ...
                    }
        """

        analysis_result = dict()
        for classes in classes_list:
            analysis_result.update({classes: {}})

        for classes in classes_list:
            data = np.transpose(np.asarray(bbox_data[classes]))
            x = data[0].reshape(-1, 1)
            y = data[1]
            model = LinearRegression().fit(x, y)
            score = model.score(x, y)
            slope = model.coef_
            bias = model.intercept_
            vec_x = np.array([[1.0]])
            vec_y = model.predict(vec_x)

            analysis_result[classes] = {"score": score,
                                        "slope": slope[0],
                                        "bias": bias,
                                        "vec": [vec_x[0][0], vec_y[0]]}

        return analysis_result

    def _similarity_intra_classes(self, analysis_result: Dict) -> np.ndarray:
        """
        Calculate similarity intra classes distribution
        Args:
            analysis_result (Dict): result of Linear Regression as following
                    {
                        "classes" : {
                                        "score": (List), coefficient of determination
                                        "slope": (List),
                                        "bias": (List),
                                        "vec": (List), (x, y) vector when x is 1.0
                            }
                        ...
                    }
        Returns:
            (np.ndarray) : similarity matrix like a confusion matrix
        """

        similarity_matrix = []

        for source_classes in self.classes_list:
            source_vec = self._unit_vector(np.asarray(analysis_result[source_classes]["vec"]))

            rows = list()
            for target_classes in self.classes_list:
                target_vec = self._unit_vector(np.asarray(analysis_result[target_classes]["vec"]))
                similarity_score = self.metric(source_vec, target_vec)
                rows.append(similarity_score)
            similarity_matrix.append(rows)

        similarity_matrix = np.asarray(similarity_matrix)

        return similarity_matrix

    @staticmethod
    def _unit_vector(vector: np.ndarray) -> np.ndarray:
        return vector / np.linalg.norm(vector, 2)

    # Distance Measure
    @staticmethod
    def _inner_product(source_vector: np.ndarray, target_vector: np.ndarray) -> np.ndarray:
        return np.inner(source_vector, target_vector)

    @staticmethod
    def _manhattan_distance(source_vector: np.ndarray, target_vector: np.ndarray) -> np.ndarray:
        return np.abs(source_vector[0] - target_vector[0]) + \
               np.abs(source_vector[1] - target_vector[1])

    @staticmethod
    def _euclidean_distance(source_vector: np.ndarray, target_vector: np.ndarray) -> np.ndarray:
        return np.sqrt(
            np.power(source_vector[0] - target_vector[0], 2) +
            np.power(source_vector[1] - target_vector[1], 2))

    def _cosine_similarity(self,
                           source_vector: np.ndarray,
                           target_vector: np.ndarray) -> np.ndarray:
        return self._inner_product(source_vector, target_vector) / \
               np.linalg.norm(source_vector, 2) * np.linalg.norm(target_vector, 2)

    @staticmethod
    def _collect_classes(anno: DetectionAnnotations) -> List:
        objs = sum([FILE.OBJECTS for FILE in anno.FILES], [])
        classes_info = Counter([obj.CLASS for obj in objs])

        return list(classes_info.keys())

    def _collect_bbox(self, anno: DetectionAnnotations, classes_list: List) -> Dict:
        """
        Args:
            anno (DetectionAnnotations) : DetectionAnnotations object
            classes_list (List) : classes names
        Returns:
            (Dict) : each classes bbox distribution as follow
            {
                "(name of class)" : [[normalized bbox width (float), normalized bbox height], ...]
                ...
            }
        """

        class_bbox = dict()
        for class_label in classes_list:
            class_bbox.update({class_label: []})

        obj_files = [FILE for FILE in anno.FILES]
        for obj_file in obj_files:
            for obj in obj_file.OBJECTS:
                class_bbox[obj.CLASS].append(self._bbox_normalize(obj_file.IMAGE_WIDTH,
                                                                  obj_file.IMAGE_HEIGHT,
                                                                  int(obj.XMIN),
                                                                  int(obj.YMIN),
                                                                  int(obj.XMAX),
                                                                  int(obj.YMAX)))
        return class_bbox

    @staticmethod
    def _bbox_normalize(image_width: int,
                        image_height: int,
                        xmin: int,
                        ymin: int,
                        xmax: int,
                        ymax: int) -> List:

        bbox_w = xmax - xmin
        bbox_h = ymax - ymin
        norm_w = bbox_w / image_width
        norm_h = bbox_h / image_height
        return [norm_w, norm_h]


class BBoxDimensionAnalyzer:

    def __init__(self,
                 anno: DetectionAnnotations,
                 num_cetroid: int = 5,
                 distance_measure: str = 'iou'):
        """
        BBox Dimension Clustering
        Args:
            anno (DetectionAnnotations) : DetectionAnnotations object
            num_cetroid : number of centroid for kmeans
            distance_measure : distance measure of kemans
        """
        classes_list = self._collect_classes(anno)
        self.anno = anno
        self.classes_list = classes_list
        self.bbox_data = self._collect_bbox(anno, classes_list)
        self.number_of_centroid = num_cetroid
        self.distance_measure = distance_measure
        self.kmeans_result = None

        # TODO should be implementation about calc similarity bbox distribution & merge

    @staticmethod
    def _collect_classes(anno: DetectionAnnotations) -> List:
        objs = sum([FILE.OBJECTS for FILE in anno.FILES], [])
        classes_info = Counter([obj.CLASS for obj in objs])

        return list(classes_info.keys())

    def _collect_bbox(self, anno: DetectionAnnotations, classes_list: List) -> Dict:
        """
        Args:
            anno (DetectionAnnotations) : DetectionAnnotations object
            classes_list (List) : classes names
        Returns:
            (Dict) : each classes bbox distribution as follow
            {
                "(name of class)" : [[normalized bbox width (float), normalized bbox height], ...]
                ...
            }
        """

        class_bbox = dict()

        for class_label in classes_list:
            class_bbox.update({class_label: []})

        obj_files = [FILE for FILE in anno.FILES]

        for obj_file in obj_files:
            for obj in obj_file.OBJECTS:
                class_bbox[obj.CLASS].append(self._bbox_normalize(obj_file.IMAGE_WIDTH,
                                                                  obj_file.IMAGE_HEIGHT,
                                                                  int(obj.XMIN),
                                                                  int(obj.YMIN),
                                                                  int(obj.XMAX),
                                                                  int(obj.YMAX)))
        return class_bbox

    @staticmethod
    def _bbox_normalize(image_width: int,
                        image_height: int,
                        xmin: int,
                        ymin: int,
                        xmax: int,
                        ymax: int) -> List:

        bbox_w = xmax - xmin
        bbox_h = ymax - ymin
        norm_w = bbox_w / image_width
        norm_h = bbox_h / image_height
        return [norm_w, norm_h]

    def fit(self) -> Dict:
        """
        Find prior boxes using Kmeans
        Args:
            (None)
        Returns
            (Dict): dimension clustering result about each classes as follow
            {
                (str) "classes": {
                                    (str) "centroid" : (List) founded centroid coordinates
                                 },
                ...
            }
        """

        self.kmeans_result = dict()
        for classes in self.classes_list:
            self.kmeans_result.update({classes: {}})

        return_value = dict()
        for classes in self.classes_list:
            return_value.update({classes: {}})

        classes_list = self.classes_list

        for classes in classes_list:
            data = self.bbox_data[classes]

            centroid_candidate = kmeans_plusplus_initializer.FARTHEST_CENTER_CANDIDATE
            centroid_initializer = kmeans_plusplus_initializer(
                data=data,
                amount_centers=self.number_of_centroid,
                amount_candidates=centroid_candidate)
            init_centroid = centroid_initializer.initialize()

            if self.distance_measure == 'iou':
                metric = distance_metric(type_metric.USER_DEFINED, func=self._iou)
            elif self.distance_measure == 'l2':
                metric = distance_metric(type_metric.EUCLIDEAN)
            elif self.distance_measure == 'l1':
                metric = distance_metric(type_metric.MANHATTAN)
            else:
                raise RuntimeError("Not supported {} metric".format(self.distance_measure))

            k_means = kmeans(data=data,
                             initial_centers=init_centroid,
                             metric=metric)
            k_means.process()
            clusters = k_means.get_clusters()
            centers = k_means.get_centers()

            return_value[classes]['centroid'] = centers
            self.kmeans_result[classes]["init_centroid"] = init_centroid
            self.kmeans_result[classes]["clusters"] = clusters
            self.kmeans_result[classes]['centers'] = centers

        return return_value

    def report(self) -> None:
        """
        Visualization Kmeans result
        """
        if self.kmeans_result is None:
            raise RuntimeError("Member variable `result` or `classes_frequency` is None"
                               "it should be run `fit` function first")

        for classes in self.classes_list:
            visualizer = cluster_visualizer(titles=["`{}`".format(classes)])

            visualizer.append_cluster(self.bbox_data[classes])
            visualizer.append_clusters(clusters=self.kmeans_result[classes]['clusters'],
                                       data=self.bbox_data[classes])
            visualizer.append_cluster(self.kmeans_result[classes]['centers'],
                                      marker='*',
                                      markersize=10)
            visualizer.show(invisible_axis=False)

    @staticmethod
    def _iou(point1, point2):
        """
        Calculate IOU
        """
        point1 = 100 * point1
        point2 = 100 * point2

        axis = 1 if len(point1.shape) > 1 else 0

        p1_area = np.prod(point1, axis=axis)
        p2_area = np.prod(point2, axis=axis)
        intersection = np.minimum(p1_area, p2_area)
        union = np.maximum(p1_area, p2_area)
        iou = intersection / union
        iou_distance = 1 - iou

        return iou_distance
