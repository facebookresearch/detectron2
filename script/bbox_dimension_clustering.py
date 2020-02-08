from read_eo_dataset import read_eo_dataset
from utils.annotation.annotation_interfaces import DetectionAnnotations
from utils.analyzer.annotation_analyzer import BBoxDimensionAnalyzer

if __name__ == "__main__":
    annotations = read_eo_dataset()
    detection_anno = DetectionAnnotations(annotations)

    bbox_dim_analyzer = BBoxDimensionAnalyzer(anno=detection_anno,
                                              num_cetroid=5,
                                              distance_measure='iou')
    prior_boxes = bbox_dim_analyzer.fit()
    bbox_dim_analyzer.report()
