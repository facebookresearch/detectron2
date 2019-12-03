import argparse
import json

from detectron2.data.datasets import register_coco_instances
from detectron2.utils.logger import setup_logger

from cvat.api import CVATAPI
from cvat.argument import cvat_args
from cvat.cvat_xml import cvat_xml_to_coco
from training.train import train

setup_logger()


def get_data_set(job_id):
    job = api.get_job(job_id).json()
    task_id = job["task_id"]
    data = api.export_data(task_id, format="CVAT XML 1.1 for images").content
    data = cvat_xml_to_coco(data, ignore_crowded=True)

    return data, task_id


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    cvat_args(parser)
    parser.add_argument("--config")
    parser.add_argument("--job_id", type=int, action="append")
    parser.add_argument("--val_job_id", type=int, action="append")
    parser.add_argument("--output")
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
        data, task_id = get_data_set(job_id)

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
            data, task_id = get_data_set(val_job_id)
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
