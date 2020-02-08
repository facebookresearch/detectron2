from read_eo_dataset import read_eo_dataset
from utils.annotation.annotation_interfaces import DetectionAnnotations
from utils.visualizer.annotation_visualizer import RBoxVisualizer

if __name__ == "__main__":
    annotations = read_eo_dataset()
    detection_anno = DetectionAnnotations(annotations)

    rbox_visualizer = RBoxVisualizer(detection_anno)
    rbox_visualizer.show(num_display=10)
