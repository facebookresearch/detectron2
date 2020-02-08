from read_eo_dataset import read_eo_dataset
from utils.annotation.annotation_interfaces import DetectionAnnotations
from utils.analyzer.annotation_analyzer import ThetaDistributionAnalyzer
from utils.visualizer.analyze_visualizer import HistorgramVisualizer

if __name__ == "__main__":
    annotations = read_eo_dataset()
    detection_anno = DetectionAnnotations(annotations)

    theta_distrib_analyzer = ThetaDistributionAnalyzer(detection_anno)
    theta_distrib_analyzer.fit()
    print(theta_distrib_analyzer.analytic_result)

    visualizer = HistorgramVisualizer()
    for classes in theta_distrib_analyzer.classes_list:
        visualizer.show(x=theta_distrib_analyzer.analytic_result[classes]["bins"],
                        y=theta_distrib_analyzer.analytic_result[classes]["hist"],
                        xlabel="{}_Tetha".format(classes),
                        ylabel="Frequency",
                        title="{}_Tetha_Historgram".format(classes),
                        xlim=[0, 180],
                        ylim=[0, 1000],
                        is_save=False)
