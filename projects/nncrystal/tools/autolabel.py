import argparse
import json
import multiprocessing as mp
import logging
from xml.etree import ElementTree

import numpy as np

from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.logger import setup_logger
from rx import operators
from rx.scheduler import ThreadPoolScheduler, ImmediateScheduler
import cv2

from cvat.cvat_xml import mask_to_polygon_xml, MaskAnnotations, build_image_node, build_polygon_nodes

setup_logger()

from cvat.api import CVATAPI
import rx

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--cvat_base")
    parser.add_argument("--config")
    parser.add_argument("--cvat_host", default="http://localhost:8080")
    parser.add_argument("--cvat_username")
    parser.add_argument("--cvat_password")
    parser.add_argument("--epsilon", default=0.01)
    parser.add_argument("--job_id", type=int)
    parser.add_argument("--output_xml")
    parser.add_argument("--weights")
    parser.add_argument("--label_list", type=str)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    cfg = get_cfg()
    cfg.merge_from_file(args.config)

    api = CVATAPI(args.cvat_host)
    api.login(args.cvat_username, args.cvat_password)

    job = api.get_job(args.job_id).json()
    start_frame = job["start_frame"]
    stop_frame = job["stop_frame"]+1
    task_id = job["task_id"]

    if args.weights:
        cfg.MODEL.WEIGHTS=args.weights
    if args.cpu:
        cfg.MODEL.DEVICE = "cpu"

    predictor = DefaultPredictor(cfg=cfg)

    task = api.list_task(task_id)[0]
    labels = task["labels"]
    if args.label_list is None:
        label_list = [lbl["name"] for lbl in labels]
    else:
        label_list = json.loads(args.label_list)

    original_annotations = api.export_data(task_id, format="CVAT XML 1.1 for images").content
    # strip the original annotations
    original_xml = ElementTree.fromstring(original_annotations)

    for img_node in original_xml.findall("image"):
        original_xml.remove(img_node)

    logging.info(f"Start fetching frames [{start_frame}-{stop_frame}]")

    scheduler = ThreadPoolScheduler()


    def frame_producer():
        for i in range(start_frame, stop_frame):
            frame = api.get_frame(task_id, i).content
            print(f"Got frame: {i}")
            buf = np.frombuffer(frame, dtype=np.uint8)
            yield (cv2.imdecode(buf, -1), i)


    def inference_ops(frame):
        predict = predictor(frame[0])
        instances = predict["instances"]
        return instances


    def polygon_xml_ops(instance):
        return build_polygon_nodes(instance, label_list, args.epsilon)


    def image_xml_ops(x):
        polygon_nodes = x[0]
        image_id = x[1]
        return build_image_node(polygon_nodes, image_id)


    def xml_accumulator(acc, x):
        acc.append(x)
        return acc


    def xml_builder(xmldata):
        if args.output_xml is not None:
            print(f"Writing {args.output_xml}")
            tree = ElementTree.ElementTree(xmldata)
            tree.write(args.output_xml)

        print(f"Upload annotation to job {args.job_id}")
        api.upload_annotations(args.job_id, ElementTree.tostring(xmldata, encoding="unicode"))



    source = rx.from_iterable(frame_producer(), scheduler).pipe(operators.share())
    image_id_source = source.pipe(operators.map(lambda x: x[1]))
    source.pipe(
        operators.map(inference_ops),
        operators.map(polygon_xml_ops),
        operators.zip(image_id_source),
        operators.map(image_xml_ops),
        operators.scan(xml_accumulator, original_xml),
        operators.take_last(1),
        operators.observe_on(ImmediateScheduler())
    ).subscribe(xml_builder)
