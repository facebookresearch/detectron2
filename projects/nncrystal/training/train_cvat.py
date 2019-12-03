import argparse
import json

from detectron2.data.datasets import register_coco_instances

from cvat.api import CVATAPI
from cvat.utils import resolve_images
from detectron2.data import DatasetCatalog, MetadataCatalog

from training.train import train
from detectron2.utils.logger import setup_logger

setup_logger()


def strip_annotation(coco_json, annotation_name):
    for cat in coco_json["categories"]:
        if cat["name"] == annotation_name:
            matching_id = cat["id"]
            coco_json["categories"].remove(cat)

            for ann in coco_json["annotations"]:
                if ann["category_id"] == matching_id:
                    coco_json["annotations"].remove(ann)


def get_data_set(job_id, strips):
    job = api.get_job(job_id).json()
    task_id = job["task_id"]
    data = api.export_data(task_id).json()

    if strips is not None:
        for strip in strips:
            strip_annotation(data, strip)
    return data, task_id


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--cvat_base")
    parser.add_argument("--config")
    parser.add_argument("--cvat_host", default="http://localhost:8080")
    parser.add_argument("--cvat_username")
    parser.add_argument("--cvat_password")
    parser.add_argument("--job_id", type=int, action="append")
    parser.add_argument("--val_job_id", type=int, action="append")
    parser.add_argument("--output")
    parser.add_argument("--strip", type=str, action="append")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--restart", action="store_true")
    parser.add_argument("--force_test", action="store_true")
    parser.add_argument("--epoch")
    args = parser.parse_args()

    api = CVATAPI(args.cvat_host)
    api.login(args.cvat_username, args.cvat_password)

    train_tasks = []

    # training tasks
    for job_id in args.job_id:
        data, task_id = get_data_set(job_id, args.strip)

        coco_json = f"datasets/train_cvat_{task_id}.coco.json"
        with open(coco_json, "w") as f:
            json.dump(data, f)
        train_task = f"cvat/train_{task_id}"
        register_coco_instances(train_task, {}, coco_json, args.cvat_base)
        train_tasks.append(train_task)
    override_cfg = ["DATASETS.TRAIN", tuple(train_tasks)]

    # validation tasks
    if args.val_job_id is not None:
        valid_tasks = []
        for val_job_id in args.val_job_id:
            data, task_id = get_data_set(val_job_id, args.strip)
            coco_json = f"datasets/valid_cvat_{task_id}.coco.json"
            with open(coco_json, "w") as f:
                json.dump(data, f)
            valid_task = f"cvat/valid_{task_id}"
            register_coco_instances(valid_task, {}, coco_json, args.cvat_base)
            valid_tasks.append(valid_task)
        override_cfg.extend(["DATASETS.TEST", tuple(valid_tasks)])

    if args.output is not None:
        override_cfg.extend(["OUTPUT_DIR", args.output])
        override_cfg.extend(["MODEL.WEIGHTS", f"{args.output}/model_final.pth"])

    if args.epoch is not None:
        override_cfg.extend(["SOLVER.MAX_ITER", args.epoch])
    train(
        config_file=args.config,
        override_cfg=override_cfg,
        resume=args.resume,
        restart=args.restart,
        force_test=args.force_test,
    )
