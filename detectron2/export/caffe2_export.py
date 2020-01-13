# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import copy
import io
import logging
import numpy as np
from typing import List
import onnx
import torch
from caffe2.python import core
from caffe2.python.onnx.backend import Caffe2Backend
from torch.onnx import OperatorExportTypes

from detectron2.export.model_convert_utils_2 import (
    convert_model_gpu,
    get_device_option_cpu,
    get_device_option_cuda,
)

from .shared import (
    ScopedWS,
    construct_init_net_from_params,
    fuse_alias_placeholder,
    get_params_from_init_net,
    group_norm_replace_aten_with_caffe2,
    remove_reshape_for_fc,
    save_graph,
)

logger = logging.getLogger(__name__)


def _export_via_onnx(model, inputs, device="CPU"):
    """
    Convert a pytorch Detectron2 model to Caffe2, via onnx.
    Args:
        model: a caffe2-compatible version of detectron2 model, defined in caffe2_modeling.py
        inputs tuple[List[torch.Tensor]]: a list of tensors that caffe2 model takes as input.
        device (str): Which device to export to. One of: "CPU", "CUDA".
    Returns:
        predict_net (NetDef):
        init_net (NetDef):
    """
    # make sure all modules are in eval mode, onnx may change the training state
    #  of the moodule if the states are not consistent
    def _check_eval(module):
        assert not module.training

    model.apply(_check_eval)

    # Export the model to ONNX
    # Set OperatorExportTypes.ONNX_ATEN_FALLBACK to fix this onnx.optimizer.optimize() error:
    #   https://github.com/onnx/onnx/issues/2417#issuecomment-546063460
    with torch.no_grad():
        with io.BytesIO() as f:
            torch.onnx.export(
                model,
                inputs,
                f,
                operator_export_type=OperatorExportTypes.ONNX_ATEN_FALLBACK,
                # verbose=True,  # NOTE: uncomment this for debugging
                # export_params=True,
            )
            onnx_model = onnx.load_from_string(f.getvalue())

    # Apply ONNX's Optimization
    all_passes = onnx.optimizer.get_available_passes()
    passes = ["fuse_bn_into_conv"]
    assert all(p in all_passes for p in passes)
    onnx_model = onnx.optimizer.optimize(onnx_model, passes)

    # Convert ONNX model to Caffe2 protobuf
    assert device in ("CPU", "CUDA")
    init_net, predict_net = Caffe2Backend.onnx_graph_to_caffe2_net(onnx_model, device=device)

    if device == "CUDA":
        # Context: onnx does not set any of the device_option's for init_net and predict_net. Since
        # detection models, eg FRCNN, utilize several CPU-only operators, eg GenerateProposals, we
        # must explicitly set device_option's in the init and predict nets (as well as CopyXToY ops)
        # to avoid runtime errors
        predict_net_net = core.Net(predict_net)
        init_net_net = core.Net(init_net)

        predict_net_net_gpu, init_net_net_gpu = convert_model_gpu(predict_net_net, init_net_net)

        predict_net = predict_net_net_gpu.Proto()
        init_net = init_net_net_gpu.Proto()

    return predict_net, init_net


def _op_stats(net_def):
    type_count = {}
    for t in [op.type for op in net_def.op]:
        type_count[t] = type_count.get(t, 0) + 1
    type_count_list = sorted(type_count.items(), key=lambda kv: kv[0])  # alphabet
    type_count_list = sorted(type_count_list, key=lambda kv: -kv[1])  # count
    return "\n".join("{:>4}x {}".format(count, name) for name, count in type_count_list)


def export_caffe2_detection_model(model: torch.nn.Module, tensor_inputs: List[torch.Tensor]):
    """
    Export a Detectron2 model via ONNX.

    Arg:
        model: a caffe2-compatible version of detectron2 model, defined in caffe2_modeling.py
        tensor_inputs: a list of tensors that caffe2 model takes as input.
    """
    model = copy.deepcopy(model)
    assert isinstance(model, torch.nn.Module)
    assert hasattr(model, "encode_additional_info")

    # infer device from model
    # model._wrapped_model.device is set by cfg.MODEL.DEVICE
    model_device = model._wrapped_model.device.type  # one of: "cuda", "cpu"
    assert model_device in ("cuda", "cpu")

    device_option = (
        get_device_option_cuda() if (model_device == "cuda") else get_device_option_cpu()
    )

    # Export via ONNX
    logger.info("Exporting a {} model via ONNX ...".format(type(model).__name__))
    # onnx uses "CUDA","CPU", not "cuda","cpu"
    predict_net, init_net = _export_via_onnx(model, (tensor_inputs,), device=model_device.upper())
    logger.info("ONNX export Done.")

    # Apply protobuf optimization
    fuse_alias_placeholder(predict_net, init_net)
    params = get_params_from_init_net(init_net)
    predict_net, params = remove_reshape_for_fc(predict_net, params)
    group_norm_replace_aten_with_caffe2(predict_net)
    # if we don't pass device_option here, then this function will return a new init_net with NO
    # device_option's set
    init_net = construct_init_net_from_params(params, device_option=device_option)

    # Record necessary information for running the pb model in Detectron2 system.
    model.encode_additional_info(predict_net, init_net)

    logger.info("Operators used in predict_net: \n{}".format(_op_stats(predict_net)))
    logger.info("Operators used in init_net: \n{}".format(_op_stats(init_net)))

    return predict_net, init_net


def run_and_save_graph(predict_net, init_net, tensor_inputs, graph_save_path):
    """
    Run the caffe2 model on given inputs, recording the shape and draw the graph.

    predict_net/init_net: caffe2 model.
    tensor_inputs: a list of tensors that caffe2 model takes as input.
    graph_save_path: path for saving graph of exported model.
    """

    logger.info("Saving graph of ONNX exported model to {} ...".format(graph_save_path))
    save_graph(predict_net, graph_save_path, op_only=False)

    # Run the exported Caffe2 net
    logger.info("Running ONNX exported model ...")
    with ScopedWS("__ws_tmp__", True) as ws:
        ws.RunNetOnce(init_net)
        initialized_blobs = set(ws.Blobs())
        uninitialized = [inp for inp in predict_net.external_input if inp not in initialized_blobs]
        for name, blob in zip(uninitialized, tensor_inputs):
            ws.FeedBlob(name, blob)

        try:
            ws.RunNetOnce(predict_net)
        except RuntimeError as e:
            logger.warning("Encountered RuntimeError: \n{}".format(str(e)))

        ws_blobs = {b: ws.FetchBlob(b) for b in ws.Blobs()}
        blob_sizes = {b: ws_blobs[b].shape for b in ws_blobs if isinstance(ws_blobs[b], np.ndarray)}

        logger.info("Saving graph with blob shapes to {} ...".format(graph_save_path))
        save_graph(predict_net, graph_save_path, op_only=False, blob_sizes=blob_sizes)

        return ws_blobs
