from read_eo_dataset import read_eo_dataset
from utils.annotation.annotation_interfaces import DetectionAnnotations
from utils.analyzer.annotation_analyzer import ClassDistributionAnalyzer
from utils.visualizer.analyze_visualizer import HistorgramVisualizer

if __name__ == "__main__":
    annotations = read_eo_dataset()
    detection_anno = DetectionAnnotations(annotations)

    class_distrib_analyzer = ClassDistributionAnalyzer(detection_anno)
    class_distrib_analyzer.fit()

    print(class_distrib_analyzer.analytic_result)
    visualizer = HistorgramVisualizer()
    visualizer.show(x=class_distrib_analyzer.analytic_result["classes_list"],
                    y=class_distrib_analyzer.analytic_result["classes_frequency"],
                    xlabel="Class",
                    ylabel="Class Frequency",
                    title="Class_Historgram",
                    is_save=False)
