# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import copy
import logging
import os
import torch
from caffe2.proto import caffe2_pb2
from torch import nn

from detectron2.config import CfgNode as CN

from .caffe2_inference import ProtobufDetectionModel
from .caffe2_modeling import META_ARCH_CAFFE2_EXPORT_TYPE_MAP, convert_batched_inputs_to_c2_format
from .shared import get_pb_arg_vali, get_pb_arg_vals, save_graph

__all__ = [
    "add_export_config",
    "export_caffe2_model",
    "Caffe2Model",
    "export_onnx_model",
    "Caffe2Tracer",
]


def add_export_config(cfg):
    """
    Args:
        cfg (CfgNode): a detectron2 config

    Returns:
        CfgNode: an updated config with new options that will be used
            by :class:`Caffe2Tracer`.
    """
    is_frozen = cfg.is_frozen()
    cfg.defrost()
    cfg.EXPORT_CAFFE2 = CN()
    cfg.EXPORT_CAFFE2.USE_HEATMAP_MAX_KEYPOINT = False
    if is_frozen:
        cfg.freeze()
    return cfg


class Caffe2Tracer:
    """
    Make a detectron2 model traceable with caffe2 style.

    An original detectron2 model may not be traceable, or
    cannot be deployed directly after being traced, due to some reasons:

    1. control flow in some ops
    2. custom ops
    3. complicated pre/post processing

    This class provides a traceable version of a detectron2 model by:

    1. Rewrite parts of the model using ops in caffe2. Note that some ops do
       not have GPU implementation.
    2. Define the inputs "after pre-processing" as inputs to the model
    3. Remove post-processing and produce raw layer outputs

    More specifically about inputs: all builtin models take two input tensors.

    1. NCHW float "data" which is an image (usually in [0, 255])
    2. Nx3 float "im_info", each row of which is (height, width, 1.0)

    After making a traceable model, the class provide methods to export such a
    model to different deployment formats.

    The class currently only supports models using builtin meta architectures.
    """

    def __init__(self, cfg, model, inputs):
        """
        Args:
            cfg (CfgNode): a detectron2 config, with extra export-related options
                added by :func:`add_export_config`.
            model (nn.Module): a model built by
                :func:`detectron2.modeling.build_model`. Weights have to be already
                loaded to this model.
            inputs: sample inputs that the given model takes for inference.
                Will be used to trace the model. Random input with no detected objects
                will not work if the model has data-dependent control flow (e.g., R-CNN).
        """
        assert isinstance(cfg, CN), cfg
        assert isinstance(model, torch.nn.Module), type(model)
        if "EXPORT_CAFFE2" not in cfg:
            cfg = add_export_config(cfg)  # will just the defaults

        self.cfg = cfg
        self.model = model
        self.inputs = inputs

    def _get_traceable(self):
        # TODO how to make it extensible to support custom models
        C2MetaArch = META_ARCH_CAFFE2_EXPORT_TYPE_MAP[self.cfg.MODEL.META_ARCHITECTURE]
        traceable_model = C2MetaArch(self.cfg, copy.deepcopy(self.model))
        traceable_inputs = traceable_model.get_caffe2_inputs(self.inputs)
        return traceable_model, traceable_inputs

    def export_caffe2(self):
        """
        Export the model to Caffe2's protobuf format.
        The returned object can be saved with ``.save_protobuf()`` method.
        The result can be loaded and executed using Caffe2 runtime.

        Returns:
            Caffe2Model
        """
        from .caffe2_export import export_caffe2_detection_model

        model, inputs = self._get_traceable()
        predict_net, init_net = export_caffe2_detection_model(model, inputs)
        return Caffe2Model(predict_net, init_net)

    def export_onnx(self):
        """
        Export the model to ONNX format.
        Note that the exported model contains custom ops only available in caffe2, therefore it
        cannot be directly executed by other runtime. Post-processing or transformation passes
        may be applied on the model to accommodate different runtimes, but we currently do not
        provide support for them.

        Returns:
            onnx.ModelProto: an onnx model.
        """
        from .caffe2_export import export_onnx_model as export_onnx_model_impl

        model, inputs = self._get_traceable()
        return export_onnx_model_impl(model, (inputs,))

    def export_torchscript(self):
        """
        Export the model to a ``torch.jit.TracedModule`` by tracing.
        The returned object can be saved to a file by ``.save()``.

        Returns:
            torch.jit.TracedModule: a torch TracedModule
        """
        model, inputs = self._get_traceable()
        logger = logging.getLogger(__name__)
        logger.info("Tracing the model with torch.jit.trace ...")
        with torch.no_grad():
            return torch.jit.trace(model, (inputs,), optimize=True)


def export_caffe2_model(cfg, model, inputs):
    """
    Export a detectron2 model to caffe2 format.

    Args:
        cfg (CfgNode): a detectron2 config, with extra export-related options
            added by :func:`add_export_config`.
        model (nn.Module): a model built by
            :func:`detectron2.modeling.build_model`.
            It will be modified by this function.
        inputs: sample inputs that the given model takes for inference.
            Will be used to trace the model.

    Returns:
        Caffe2Model
    """
    return Caffe2Tracer(cfg, model, inputs).export_caffe2()


def export_onnx_model(cfg, model, inputs):
    """
    Export a detectron2 model to ONNX format.
    Note that the exported model contains custom ops only available in caffe2, therefore it
    cannot be directly executed by other runtime. Post-processing or transformation passes
    may be applied on the model to accommodate different runtimes, but we currently do not
    provide support for them.

    Args:
        cfg (CfgNode): a detectron2 config, with extra export-related options
            added by :func:`add_export_config`.
        model (nn.Module): a model built by
            :func:`detectron2.modeling.build_model`.
            It will be modified by this function.
        inputs: sample inputs that the given model takes for inference.
            Will be used to trace the model.
    Returns:
        onnx.ModelProto: an onnx model.
    """
    return Caffe2Tracer(cfg, model, inputs).export_onnx()


class Caffe2Model(nn.Module):
    """
    A wrapper around the traced model in caffe2's pb format.
    """

    def __init__(self, predict_net, init_net):
        super().__init__()
        self.eval()  # always in eval mode
        self._predict_net = predict_net
        self._init_net = init_net
        self._predictor = None

    @property
    def predict_net(self):
        """
        Returns:
            core.Net: the underlying caffe2 predict net
        """
        return self._predict_net

    @property
    def init_net(self):
        """
        Returns:
            core.Net: the underlying caffe2 init net
        """
        return self._init_net

    __init__.__HIDE_SPHINX_DOC__ = True

    def save_protobuf(self, output_dir):
        """
        Save the model as caffe2's protobuf format.

        Args:
            output_dir (str): the output directory to save protobuf files.
        """
        logger = logging.getLogger(__name__)
        logger.info("Saving model to {} ...".format(output_dir))
        os.makedirs(output_dir, exist_ok=True)

        with open(os.path.join(output_dir, "model.pb"), "wb") as f:
            f.write(self._predict_net.SerializeToString())
        with open(os.path.join(output_dir, "model.pbtxt"), "w") as f:
            f.write(str(self._predict_net))
        with open(os.path.join(output_dir, "model_init.pb"), "wb") as f:
            f.write(self._init_net.SerializeToString())

    def save_graph(self, output_file, inputs=None):
        """
        Save the graph as SVG format.

        Args:
            output_file (str): a SVG file
            inputs: optional inputs given to the model.
                If given, the inputs will be used to run the graph to record
                shape of every tensor. The shape information will be
                saved together with the graph.
        """
        from .caffe2_export import run_and_save_graph

        if inputs is None:
            save_graph(self._predict_net, output_file, op_only=False)
        else:
            size_divisibility = get_pb_arg_vali(self._predict_net, "size_divisibility", 0)
            device = get_pb_arg_vals(self._predict_net, "device", b"cpu").decode("ascii")
            inputs = convert_batched_inputs_to_c2_format(inputs, size_divisibility, device)
            inputs = [x.cpu().numpy() for x in inputs]
            run_and_save_graph(self._predict_net, self._init_net, inputs, output_file)

    @staticmethod
    def load_protobuf(dir):
        """
        Args:
            dir (str): a directory used to save Caffe2Model with
                :meth:`save_protobuf`.
                The files "model.pb" and "model_init.pb" are needed.

        Returns:
            Caffe2Model: the caffe2 model loaded from this directory.
        """
        predict_net = caffe2_pb2.NetDef()
        with open(os.path.join(dir, "model.pb"), "rb") as f:
            predict_net.ParseFromString(f.read())

        init_net = caffe2_pb2.NetDef()
        with open(os.path.join(dir, "model_init.pb"), "rb") as f:
            init_net.ParseFromString(f.read())

        return Caffe2Model(predict_net, init_net)

    def __call__(self, inputs):
        """
        An interface that wraps around a caffe2 model and mimics detectron2's models'
        input & output format. This is used to compare the outputs of caffe2 model
        with its original torch model.

        Due to the extra conversion between torch/caffe2,
        this method is not meant for benchmark.
        """
        if self._predictor is None:
            self._predictor = ProtobufDetectionModel(self._predict_net, self._init_net)
        return self._predictor(inputs)
