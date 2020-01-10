import argparse
import io
import numpy as np
import os
import time
import torchvision
from PIL import Image

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import build_detection_test_loader
from detectron2.evaluation import COCOEvaluator, inference_on_dataset, print_csv_format
from detectron2.export import add_export_config, export_caffe2_model
from detectron2.model_zoo import get_checkpoint_url, get_config_file
from detectron2.modeling import build_model
from detectron2.utils.logger import setup_logger

import requests


"""
CLI tool to export a Detectron2 pytorch model to Caffe2 protobuf init+predict nets. Supports cpu
and cuda devices.

Tip: when using --from-model-zoo, config-file should be relative to detectron2/configs, eg:
    --config-file COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml

Example usage:

# (cuda) Download a pretrained Detectron2 pytorch model from the model zoo, and export it to Caffe2
$ python3.6 tools/caffe2_converter.py \
    --config-file COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml \
    --output /data1/erickim/demo_caffe2_converter/faster_rcnn_X_101_32x8d_FPN_3x_cuda \
    --from-model-zoo \
    --test-image-path https://farm1.staticflickr.com/33/38956064_47b68504cb_z.jpg \
    MODEL.DEVICE cuda

To export to cpu device, set MODEL.DEVICE to cpu.
"""


def _create_and_setup_cfg(config_path, opts):
    """Creates and initializes a Detectron2 Config.
    Args:
        config_path (str):
        opts (list[str]): List of Config options. Overrides the values in config_path.
            Example: ["MODEL.DEVICE", "cpu", "DATASETS.TEST", "('my_test_set',)"]
    Returns:
        config:
    """
    cfg = get_cfg()
    cfg = add_export_config(cfg)
    cfg.merge_from_file(config_path)
    cfg.merge_from_list(opts)
    cfg.freeze()
    return cfg


def _download_image_to_pil(image_url):
    """Downloads an image from a URL.
    Args:
        image_url (str): Target image to download.
    Returns:
        image (PIL.Image): Image in PIL form.
    """
    response = requests.get(image_url)
    return Image.open(io.BytesIO(response.content))


def is_url(path):
    """Returns True if path is a URL.
    Args:
        path (str):
            Example: "http://foo/bar.jpg"
    Returns:
        out (bool):
    """
    # simple heuristic
    path_lower = path.lower()
    return path_lower.startswith("http://") or path_lower.startswith("https://")


def _build_batch_from_image_path(test_image_path):
    """Builds a batch from an image path.
    Args:
        test_image_path (str): Can either be a local path to an image, or an image URL.
    Returns:
        first_batch (list[dict]):
    """
    if is_url(test_image_path):
        test_image_pil = _download_image_to_pil(test_image_path)
    else:
        test_image_pil = Image.open(test_image_path)

    # torchvision ToTensor() scales pixel values from [0,255] to [0,1]. Detection data loaders do
    # not perform this scaling, so undo it.
    # Return image as float32, to avoid unnecessary Cast ops in caffe2 exported model
    test_image_tensor = (
        torchvision.transforms.ToTensor()(test_image_pil).float() * 255.0
    )  # [C, H, W]
    first_batch = [
        {
            "file_name": test_image_path,
            "width": test_image_pil.size[0],
            "height": test_image_pil.size[1],
            "image_id": 0,
            "image": test_image_tensor,
        }
    ]
    return first_batch


def infer_and_record_latency(caffe2_model, first_batch, nb_trials=25):
    """Performs an inference benchmark, and logs the model output.
    Args:
        caffe2_model:
        first_batch (list[dict]):
        nb_trials (int):
    Returns:
        model_output:
        tocs (list[float]): Inference latencies for each trial. Length should be nb_trials.
    """
    logger.info("Running inference benchmark on: {}".format(first_batch[0]["file_name"]))
    # warm up
    for _ in range(10):
        caffe2_model(first_batch)
    # perform benchmark
    tocs = []
    for _ in range(nb_trials):
        tic_i = time.time()
        caffe2_model(first_batch)
        tocs.append(time.time() - tic_i)
    model_output = caffe2_model(first_batch)
    logger.info("model_output: {}".format(model_output))
    logger.info(
        "inference latency: mn={:.4f}s std={:.4f}s min={:.4f}s max={:.4f}s ({} trials)".format(
            np.mean(tocs), np.std(tocs), np.min(tocs), np.max(tocs), nb_trials
        )
    )
    return model_output, tocs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert a model to Caffe2")
    parser.add_argument("--config-file", default="", metavar="FILE", help="path to config file")
    parser.add_argument("--run-eval", action="store_true")
    parser.add_argument("--output", help="output directory for the converted caffe2 model")
    parser.add_argument(
        "--from-model-zoo",
        action="store_true",
        help="Optional. If set, this will load the config and model snapshot weights from the "
        "Detectron2 model zoo API. --config-file should refer to a config file relative to "
        "detectron2/configs, eg 'configs/COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml'",
    )
    parser.add_argument(
        "--test-image-path",
        help="Optional. Use a provided image as the sample image used to trace the model during "
        "conversion. Can be either a local image path or a URL. Overrides the DATASETS.TEST "
        "behavior",
    )
    parser.add_argument(
        "--skip-infer-benchmark",
        action="store_true",
        help="Skips the inference benchmark after model export",
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()
    logger = setup_logger()
    logger.info("Command line arguments: " + str(args))

    tic_start = time.time()

    # Create Config
    opts = list(args.opts)
    if args.from_model_zoo:
        # Use Detectron2 model zoo API to retrieve Config and model snapshot
        local_config_fullpath = get_config_file(args.config_file)
        model_snapshot_url = get_checkpoint_url(args.config_file)
        assert "MODEL.WEIGHTS" not in opts
        opts.extend(["MODEL.WEIGHTS", model_snapshot_url])
    else:
        # directly use local config file
        local_config_fullpath = args.config_file
    if not os.path.isfile(local_config_fullpath):
        raise OSError("Could not find config file: {}".format(local_config_fullpath))
    if "MODEL.WEIGHTS" not in opts:
        # For most usecases, the user should pass MODEL.WEIGHTS in the CLI opts to point to their
        # trained weights.
        # Most config files have MODEL.WEIGHTS set to the backbone weights only, eg ResNet101
        # trained on ImageNet.
        logger.info(
            "Warning: MODEL.WEIGHTS was not passed in CLI. Take care to ensure that MODEL.WEIGHTS"
            "in the config file points to your trained weights, otherwise your exported model will "
            "contain uninitialized weights!"
        )
    logger.info("Creating config from: {}, with opts={}".format(local_config_fullpath, opts))
    cfg = _create_and_setup_cfg(local_config_fullpath, opts)
    logger.info(
        "Exporting model for {} device, using weights from {}".format(
            cfg.MODEL.DEVICE, cfg.MODEL.WEIGHTS
        )
    )

    # Create sample image to run detector on during conversion ("first_batch")
    if args.test_image_path:
        # (simple) use a single image
        logger.info("Using test_image_path={}".format(args.test_image_path))
        first_batch = _build_batch_from_image_path(args.test_image_path)
    else:
        # get a sample data from the test set
        data_loader = build_detection_test_loader(cfg, cfg.DATASETS.TEST[0])
        first_batch = next(iter(data_loader))

    # create a torch model
    torch_model = build_model(cfg)
    DetectionCheckpointer(torch_model).resume_or_load(cfg.MODEL.WEIGHTS)

    # convert and save caffe2 model
    caffe2_model = export_caffe2_model(cfg, torch_model, first_batch)
    caffe2_model.save_protobuf(args.output)
    # draw the caffe2 graph
    caffe2_model.save_graph(os.path.join(args.output, "model_def.svg"), inputs=first_batch)

    if not args.skip_infer_benchmark:
        # run model on sample image
        infer_and_record_latency(caffe2_model, first_batch)

    # run evaluation with the converted model
    if args.run_eval:
        dataset = cfg.DATASETS.TEST[0]
        data_loader = build_detection_test_loader(cfg, dataset)
        # NOTE: hard-coded evaluator. change to the evaluator for your dataset
        evaluator = COCOEvaluator(dataset, cfg, True, args.output)
        metrics = inference_on_dataset(caffe2_model, data_loader, evaluator)
        print_csv_format(metrics)
    logger.info("Done ({:.4f}s)".format(time.time() - tic_start))
