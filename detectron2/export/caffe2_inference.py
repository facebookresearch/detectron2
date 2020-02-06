# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import collections
import logging
import numpy as np
import torch
from caffe2.proto import caffe2_pb2
from caffe2.python import core

from .caffe2_modeling import META_ARCH_CAFFE2_EXPORT_TYPE_MAP, convert_batched_inputs_to_c2_format
from .shared import ScopedWS, get_pb_arg_vali, get_pb_arg_vals, infer_device_type

logger = logging.getLogger(__name__)


class ProtobufModel(torch.nn.Module):
    """
    A class works just like nn.Module in terms of inference, but running
    caffe2 model under the hood. Input/Output are Dict[str, tensor] whose keys
    are in external_input/output.
    """

    def __init__(self, predict_net, init_net):
        logger.info("Initializing ProtobufModel ...")
        super().__init__()
        assert isinstance(predict_net, caffe2_pb2.NetDef)
        assert isinstance(init_net, caffe2_pb2.NetDef)
        self.ws_name = "__ws_tmp__"
        self.net = core.Net(predict_net)

        with ScopedWS(self.ws_name, is_reset=True, is_cleanup=False) as ws:
            ws.RunNetOnce(init_net)
            for blob in self.net.Proto().external_input:
                if blob not in ws.Blobs():
                    ws.CreateBlob(blob)
            ws.CreateNet(self.net)

        self._error_msgs = set()

    def forward(self, inputs_dict):
        assert all(inp in self.net.Proto().external_input for inp in inputs_dict)
        with ScopedWS(self.ws_name, is_reset=False, is_cleanup=False) as ws:
            for b, tensor in inputs_dict.items():
                ws.FeedBlob(b, tensor)
            try:
                ws.RunNet(self.net.Proto().name)
            except RuntimeError as e:
                if not str(e) in self._error_msgs:
                    self._error_msgs.add(str(e))
                    logger.warning("Encountered new RuntimeError: \n{}".format(str(e)))
                logger.warning("Catch the error and use partial results.")

            outputs_dict = collections.OrderedDict(
                [(b, ws.FetchBlob(b)) for b in self.net.Proto().external_output]
            )
            # Remove outputs of current run, this is necessary in order to
            # prevent fetching the result from previous run if the model fails
            # in the middle.
            for b in self.net.Proto().external_output:
                # Needs to create uninitialized blob to make the net runable.
                # This is "equivalent" to: ws.RemoveBlob(b) then ws.CreateBlob(b),
                # but there'no such API.
                ws.FeedBlob(b, "{}, a C++ native class of type nullptr (uninitialized).".format(b))

        return outputs_dict


class ProtobufDetectionModel(torch.nn.Module):
    """
    A class works just like a pytorch meta arch in terms of inference, but running
    caffe2 model under the hood.
    """

    def __init__(self, predict_net, init_net, *, convert_outputs=None):
        """
        Args:
            predict_net, init_net (core.Net): caffe2 nets
            convert_outptus (callable): a function that converts caffe2
                outputs to the same format of the original pytorch model.
                By default, use the one defined in the caffe2 meta_arch.
        """
        super().__init__()
        self.protobuf_model = ProtobufModel(predict_net, init_net)
        self.size_divisibility = get_pb_arg_vali(predict_net, "size_divisibility", 0)
        self.device = get_pb_arg_vals(predict_net, "device", b"cpu").decode("ascii")

        if convert_outputs is None:
            meta_arch = get_pb_arg_vals(predict_net, "meta_architecture", b"GeneralizedRCNN")
            meta_arch = META_ARCH_CAFFE2_EXPORT_TYPE_MAP[meta_arch.decode("ascii")]
            self._convert_outputs = meta_arch.get_outputs_converter(predict_net, init_net)
        else:
            self._convert_outputs = convert_outputs

    def _infer_output_devices(self, inputs_dict):
        def _get_device_type(torch_tensor):
            assert torch_tensor.device.type in ["cpu", "cuda"]
            assert torch_tensor.device.index == 0
            return torch_tensor.device.type

        predict_net = self.protobuf_model.net.Proto()
        input_device_types = {
            (name, 0): _get_device_type(tensor) for name, tensor in inputs_dict.items()
        }
        device_type_map = infer_device_type(
            predict_net, known_status=input_device_types, device_name_style="pytorch"
        )
        ssa, versions = core.get_ssa(predict_net)
        versioned_outputs = [(name, versions[name]) for name in predict_net.external_output]
        output_devices = [device_type_map[outp] for outp in versioned_outputs]
        return output_devices

    def _convert_inputs(self, batched_inputs):
        # currently all models convert inputs in the same way
        data, im_info = convert_batched_inputs_to_c2_format(
            batched_inputs, self.size_divisibility, self.device
        )
        return {"data": data, "im_info": im_info}

    def forward(self, batched_inputs):
        c2_inputs = self._convert_inputs(batched_inputs)
        c2_results = self.protobuf_model(c2_inputs)

        if any(t.device.type != "cpu" for _, t in c2_inputs.items()):
            output_devices = self._infer_output_devices(c2_inputs)
        else:
            output_devices = ["cpu" for _ in self.protobuf_model.net.Proto().external_output]

        def _cast_caffe2_blob_to_torch_tensor(blob, device):
            return torch.Tensor(blob).to(device) if isinstance(blob, np.ndarray) else None

        c2_results = {
            name: _cast_caffe2_blob_to_torch_tensor(c2_results[name], device)
            for name, device in zip(self.protobuf_model.net.Proto().external_output, output_devices)
        }

        return self._convert_outputs(batched_inputs, c2_inputs, c2_results)
