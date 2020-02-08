import matplotlib.pyplot as plt
import numpy as np

from PIL import Image, ImageDraw
from utils.annotation.annotation_interfaces import DetectionAnnotations
from utils.annotation.annotion_converters import RotatedBBoxConverter


class RBoxVisualizer:

    def __init__(self, anno: DetectionAnnotations):
        self.anno = anno
        self.rb_cvt = RotatedBBoxConverter()

    def show(self, num_display=10):
        for idx, FILE in enumerate(self.anno.FILES):
            if idx > num_display:
                break

            filepath = FILE.FILEPATH

            image = Image.open(filepath)
            draw = ImageDraw.ImageDraw(image)

            for OBJ in FILE.OBJECTS:
                classes = OBJ.CLASS
                xmin = OBJ.XMIN
                ymin = OBJ.YMIN
                xmax = OBJ.XMAX
                ymax = OBJ.YMAX
                theta = OBJ.THETA

                bbox = np.array([xmin, ymin, xmax, ymax])

                horizon_bbox_points = self.rb_cvt.bbox_to_points(bbox)
                rotated_bbox_points = self.rb_cvt.rotate_horizon_bbox_with_theta(horizon_bbox_points,
                                                                                 theta)

                p1 = tuple(rotated_bbox_points[0][:-1])
                p2 = tuple(rotated_bbox_points[1][:-1])
                p3 = tuple(rotated_bbox_points[2][:-1])
                p4 = tuple(rotated_bbox_points[3][:-1])

                draw.line((p1, p2), fill=(0, 255, 0))
                draw.line((p2, p3), fill=(0, 255, 0))
                draw.line((p3, p4), fill=(0, 255, 0))
                draw.line((p4, p1), fill=(0, 255, 0))

                draw.text(p1, "P1", fill=(255, 0, 0))
                draw.text(p2, "P2", fill=(255, 0, 0))
                draw.text(p3, "P3", fill=(255, 0, 0))
                draw.text(p4, "P4", fill=(255, 0, 0))

                draw.text((p1[0] - 10, p1[1] - 10), classes, fill=(255, 255, 255))

            plt.figure()
            plt.imshow(image)
            plt.show()
