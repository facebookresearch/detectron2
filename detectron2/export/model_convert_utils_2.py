# Copyright (c) 2017-present, Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################

"""Helper functions for model conversion to pb"""
# This is a copy+paste from Detectron, along with unused functions removed:
#   Detectron/detectron/utils/model_convert_utils.py

from __future__ import absolute_import, division, print_function, unicode_literals
import copy
from functools import wraps
from caffe2.proto import caffe2_pb2
from caffe2.python import core


class OpFilter(object):
    def __init__(self, **kwargs):
        self.type = None
        self.type_in = None
        self.inputs = None
        self.outputs = None
        self.input_has = None
        self.output_has = None
        self.cond = None
        self.reverse = False

        assert all(x in self.__dict__ for x in kwargs)
        self.__dict__.update(kwargs)

    def check(self, op):
        ret = self.reverse
        if self.type and op.type != self.type:
            return ret
        if self.type_in and op.type not in self.type_in:
            return ret
        if self.inputs and set(op.input) != set(self.inputs):
            return ret
        if self.outputs and set(op.output) != set(self.outputs):
            return ret
        if self.input_has and self.input_has not in op.input:
            return ret
        if self.output_has and self.output_has not in op.output:
            return ret
        if self.cond is not None and not self.cond:
            return ret
        return not ret


def filter_op(op, **kwargs):
    """ Returns true if passed all checks """
    return OpFilter(**kwargs).check(op)


def op_filter(**filter_args):
    """ Returns None if no condition is satisfied """

    def actual_decorator(f):
        @wraps(f)
        def wrapper(op, **params):
            if not filter_op(op, **filter_args):
                return None
            return f(op, **params)

        return wrapper

    return actual_decorator


def op_func_chain(convert_func_list):
    """ Run funcs one by one until func return is not None """
    assert isinstance(convert_func_list, list)

    def _chain(op):
        for x in convert_func_list:
            ret = x(op)
            if ret is not None:
                return ret
        return None

    return _chain


def convert_op_in_ops(ops_ref, func_or_list):
    func = func_or_list
    if isinstance(func_or_list, list):
        func = op_func_chain(func_or_list)
    ops = list(ops_ref)
    converted_ops = []
    for op in ops:
        new_ops = func(op)
        if new_ops is not None and not isinstance(new_ops, list):
            new_ops = [new_ops]
        converted_ops.extend(new_ops if new_ops is not None else [op])
    del ops_ref[:]
    # ops_ref maybe of type RepeatedCompositeFieldContainer
    # which does not have append()
    ops_ref.extend(converted_ops)


def convert_op_in_proto(proto, func_or_list):
    convert_op_in_ops(proto.op, func_or_list)


def get_device_option_cpu():
    device_option = core.DeviceOption(caffe2_pb2.CPU)
    return device_option


def get_device_option_cuda(gpu_id=0):
    device_option = caffe2_pb2.DeviceOption()
    device_option.device_type = caffe2_pb2.CUDA
    device_option.device_id = gpu_id
    return device_option


def convert_model_gpu(net, init_net):
    """Given a predict/init net with no device_option's set, eg a CPU only model, this upgrades the
    predict/init nets to run on the GPU device.
    In summary, this performs:
    (1) init_net: Set device_option's for parameter-populating ops to the GPU device
    (2) predict_net: Set device_option's to GPU/CPU as appropriate, and also add CopyCPUToGPU,
        CopyGPUToCPU ops appropriate
    Copied + modified from: Detectron/tools/convert_pkl_to_pb.py
    Args:
        net (core.Net): A predict net, containing inference operators.
        init_net (core.Net): An init net, containing parameter-blob populating operators.
    Returns:
        predict_net_out (core.Net):
        init_net_out (core.Net):
    """
    ret_net = copy.deepcopy(net)
    ret_init_net = copy.deepcopy(init_net)

    cdo_cuda = get_device_option_cuda()
    cdo_cpu = get_device_option_cpu()

    # Caffe2 operators that only have a CPU implementation, eg no GPU implementation
    CPU_OPS = [
        # Detectron
        ["CollectAndDistributeFpnRpnProposals", None],
        ["GenerateProposals", None],
        ["BBoxTransform", None],
        ["BoxWithNMSLimit", None],
        ["Slice", None],
        ["FlattenToVec", None],
        ["Negative", None],
        ["Cast", None],
        ["Shape", None],
        ["FlexibleTopK", None],
        ["Gather", None],
        # Detectron2
        ["CollectRpnProposals", None],
        ["DistributeFpnProposals", None],
    ]
    # Blobs that reside on CPU memory
    CPU_BLOBS = ["im_info", "anchor"]

    @op_filter()
    def convert_op_gpu(op):
        for x in CPU_OPS:
            if filter_op(op, type=x[0], inputs=x[1]):
                op.device_option.CopyFrom(cdo_cpu)
                return None
        op.device_option.CopyFrom(cdo_cuda)
        return [op]

    @op_filter()
    def convert_init_op_gpu(op):
        if op.output[0] in CPU_BLOBS:
            op.device_option.CopyFrom(cdo_cpu)
        else:
            op.device_option.CopyFrom(cdo_cuda)
        return [op]

    convert_op_in_proto(ret_init_net.Proto(), convert_init_op_gpu)
    convert_op_in_proto(ret_net.Proto(), convert_op_gpu)

    ret = core.InjectDeviceCopiesAmongNets([ret_init_net, ret_net])

    return [ret[0][1], ret[0][0]]
