import argparse

from detectron2.data.datasets import register_coco_instances
from detectron2.utils.logger import setup_logger

from training.train import train
setup_logger()
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="append", required=True)
    parser.add_argument("--config")
    parser.add_argument("--epoch")
    parser.add_argument("--output")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--restart", action="store_true")
    parser.add_argument("--base_dir")
    args = parser.parse_args()

    # training tasks
    train_tasks = []
    for i, file in enumerate(args.train):
        train_task = f"cvat/train_{i}"
        register_coco_instances(train_task, {}, file, args.base_dir)
        train_tasks.append(train_task)
    override_cfg = ["DATASETS.TRAIN", tuple(train_tasks)]

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
        force_test=False,
    )