from read_eo_dataset import read_eo_dataset
from utils.annotation.annotation_interfaces import DetectionAnnotations
from utils.analyzer.annotation_analyzer import BBoxDistributionAnalyzer

if __name__ == "__main__":
    annotations = read_eo_dataset()
    detection_anno = DetectionAnnotations(annotations)

    bbox_distrib_analyzer = BBoxDistributionAnalyzer(detection_anno)
    bbox_distrib_analyzer.fit()
    bbox_distrib_analyzer.report_modeling()
