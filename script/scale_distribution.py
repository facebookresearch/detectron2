from read_eo_dataset import read_eo_dataset
from utils.annotation.annotation_interfaces import DetectionAnnotations
from utils.analyzer.annotation_analyzer import ScaleDistributionAnalyzer
from utils.visualizer.analyze_visualizer import HistorgramVisualizer

if __name__ == "__main__":
    annotations = read_eo_dataset()
    detection_anno = DetectionAnnotations(annotations)

    scale_distrib_analyzer = ScaleDistributionAnalyzer(detection_anno)
    scale_distrib_analyzer.fit()
    print(scale_distrib_analyzer.analytic_result)

    visualizer = HistorgramVisualizer()
    SCALE_MAX_VAL = 4000000
    for classes in scale_distrib_analyzer.classes_list:
        visualizer.show(x=scale_distrib_analyzer.analytic_result[classes]["bins"] / SCALE_MAX_VAL,
                        y=scale_distrib_analyzer.analytic_result[classes]["hist"],
                        xlabel="{}_Scale".format(classes),
                        ylabel="Frequency",
                        title="{}_Scale_Historgram".format(classes),
                        xlim=[0, 1],
                        ylim=[0, 7000],
                        is_save=False)
