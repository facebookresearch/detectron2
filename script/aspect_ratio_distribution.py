from read_eo_dataset import read_eo_dataset
from utils.annotation.annotation_interfaces import DetectionAnnotations
from utils.analyzer.annotation_analyzer import AspectRatioAnalyzer
from utils.visualizer.analyze_visualizer import HistorgramVisualizer


if __name__ == "__main__":
    annotations = read_eo_dataset()
    detection_anno = DetectionAnnotations(annotations)

    aspect_ratio_analyzer = AspectRatioAnalyzer(detection_anno)
    aspect_ratio_analyzer.fit()
    print(aspect_ratio_analyzer.analytic_result)

    visualizer = HistorgramVisualizer()
    for classes in aspect_ratio_analyzer.classes_list:
        visualizer.show(x=aspect_ratio_analyzer.analytic_result[classes]["bins"],
                        y=aspect_ratio_analyzer.analytic_result[classes]["hist"],
                        xlabel="{}_Aspect Ratio".format(classes),
                        ylabel="Frequency",
                        title="{}_Aspect_Ratio_Historgram".format(classes),
                        xlim=[-1, 1],
                        ylim=[0, 10],
                        is_save=False)
