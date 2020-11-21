#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
import argparse
import os
import onnx
import torch
from torch import Tensor, nn

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import build_detection_test_loader
from detectron2.evaluation import COCOEvaluator, inference_on_dataset, print_csv_format
from detectron2.export import Caffe2Tracer, add_export_config
from detectron2.export.torchscript import dump_torchscript_IR, export_torchscript_with_instances
from detectron2.modeling import GeneralizedRCNN, build_model
from detectron2.structures import Boxes
from detectron2.utils.env import TORCH_VERSION
from detectron2.utils.logger import setup_logger


def setup_cfg(args):
    cfg = get_cfg()
    # cuda context is initialized before creating dataloader, so we don't fork anymore
    cfg.DATALOADER.NUM_WORKERS = 0
    cfg = add_export_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg


def export_caffe2_tracing(cfg, torch_model, inputs):
    tracer = Caffe2Tracer(cfg, torch_model, inputs)
    if args.format == "caffe2":
        caffe2_model = tracer.export_caffe2()
        caffe2_model.save_protobuf(args.output)
        # draw the caffe2 graph
        caffe2_model.save_graph(os.path.join(args.output, "model.svg"), inputs=inputs)
        return caffe2_model
    elif args.format == "onnx":
        onnx_model = tracer.export_onnx()
        onnx.save(onnx_model, os.path.join(args.output, "model.onnx"))
    elif args.format == "torchscript":
        ts_model = tracer.export_torchscript()
        ts_model.save(os.path.join(args.output, "model.ts"))
        dump_torchscript_IR(ts_model, args.output)


# experimental. API not yet final
def export_scripting(torch_model):
    assert TORCH_VERSION >= (1, 8)
    fields = {
        "proposal_boxes": Boxes,
        "objectness_logits": Tensor,
        "pred_boxes": Boxes,
        "scores": Tensor,
        "pred_classes": Tensor,
        "pred_masks": Tensor,
        "pred_keypoints": torch.Tensor,
        "pred_keypoint_heatmaps": torch.Tensor,
    }
    # maybe can export to onnx format?
    assert args.format == "torchscript"
    ts_model = export_torchscript_with_instances(torch_model, fields)
    ts_model.save(os.path.join(args.output, "model.ts"))
    dump_torchscript_IR(ts_model, args.output)
    # TODO inference in Python now missing postprocessing glue code
    return None


# experimental. API not yet final
def export_tracing(torch_model, inputs):
    assert TORCH_VERSION >= (1, 8)
    # RetinaNet needs a slightly different wrapper.
    assert isinstance(torch_model, GeneralizedRCNN)
    image = inputs[0]["image"]

    # TODO wrapper should be automatically generated
    class WrapModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.torch_model = torch_model

        def forward(self, image):
            inputs = [{"image": image}]
            outputs = self.torch_model.inference(inputs, do_postprocess=False)[0]
            outputs = outputs.get_fields()
            from detectron2.utils.analysis import _flatten_to_tuple

            return _flatten_to_tuple(outputs)

    from detectron2.export.torchscript_patch import patch_builtin_len

    with torch.no_grad(), patch_builtin_len():
        assert args.format == "torchscript"
        ts_model = torch.jit.trace(WrapModel(), (image,))
        ts_model.save(os.path.join(args.output, "model.ts"))
        dump_torchscript_IR(ts_model, args.output)
        # NOTE onnx export fails in pytorch
        # if args.format == "onnx":
        #     torch.onnx.export(WrapModel(), (image,), os.path.join(args.output, "model.onnx"))

    # TODO inference in Python now missing postprocessing glue code
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert a model using caffe2 tracing.")
    parser.add_argument(
        "--format",
        choices=["caffe2", "onnx", "torchscript"],
        help="output format",
        default="caffe2",
    )
    parser.add_argument(
        "--export-method",
        choices=["caffe2_tracing", "tracing", "scripting"],
        help="Method to export models",
        default="caffe2_tracing",
    )
    parser.add_argument("--config-file", default="", metavar="FILE", help="path to config file")
    parser.add_argument("--run-eval", action="store_true")
    parser.add_argument("--output", help="output directory for the converted model")
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()
    logger = setup_logger()
    logger.info("Command line arguments: " + str(args))
    os.makedirs(args.output, exist_ok=True)

    cfg = setup_cfg(args)

    # create a torch model
    torch_model = build_model(cfg)
    DetectionCheckpointer(torch_model).resume_or_load(cfg.MODEL.WEIGHTS)
    torch_model.eval()

    # get a sample data
    data_loader = build_detection_test_loader(cfg, cfg.DATASETS.TEST[0])
    first_batch = next(iter(data_loader))

    # convert and save model
    if args.export_method == "caffe2_tracing":
        exported_model = export_caffe2_tracing(cfg, torch_model, first_batch)
    elif args.export_method == "scripting":
        exported_model = export_scripting(torch_model)
    elif args.export_method == "tracing":
        exported_model = export_tracing(torch_model, first_batch)

    # run evaluation with the converted model
    if args.run_eval:
        assert exported_model is not None, (
            "Python inference is not implemented for "
            f"export_method={args.export_method}, format={args.format}."
        )
        logger.info("Running evaluation ... this takes a long time if you export to CPU.")
        dataset = cfg.DATASETS.TEST[0]
        data_loader = build_detection_test_loader(cfg, dataset)
        # NOTE: hard-coded evaluator. change to the evaluator for your dataset
        evaluator = COCOEvaluator(dataset, output_dir=args.output)
        metrics = inference_on_dataset(exported_model, data_loader, evaluator)
        print_csv_format(metrics)
