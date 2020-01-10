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
# This is a copy+paste from Detectron:
#   PinDetectron/detectron/utils/model_convert_utils.py

from __future__ import absolute_import, division, print_function, unicode_literals
import copy
import numpy as np
from functools import wraps
from caffe2.proto import caffe2_pb2
from caffe2.python import core, workspace


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

        assert all([x in self.__dict__ for x in kwargs])
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
    ops = [op for op in ops_ref]
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


def get_op_arg(op, arg_name):
    for x in op.arg:
        if x.name == arg_name:
            return x
    return None


def get_op_arg_valf(op, arg_name, default_val):
    arg = get_op_arg(op, arg_name)
    return arg.f if arg is not None else default_val


def update_mobile_engines(net):
    for op in net.op:
        if op.type == "Conv":
            op.engine = "NNPACK"
        if op.type == "ConvTranspose":
            op.engine = "BLOCK"


def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    from itertools import tee

    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


def blob_uses(net, blob):
    u = []
    for i, op in enumerate(net.op):
        if blob in op.input or blob in op.control_input:
            u.append(i)
    return u


def fuse_first_affine(net, params, removed_tensors):
    net = copy.deepcopy(net)
    params = copy.deepcopy(params)

    for ((i, current), (j, next_)) in pairwise(enumerate(net.op)):
        if next_.input[0] != current.output[0]:
            continue

        if current.type not in ("Conv", "ConvTranspose") or next_.type != "AffineChannel":
            continue
        if current.output[0] != next_.output[0] and len(blob_uses(net, current.output[0])) != 1:
            # Can't fuse if more than one user unless AffineChannel is inplace
            continue

        # else, can fuse
        conv = current
        affine = next_
        fused_conv = copy.deepcopy(conv)
        fused_conv.output[0] = affine.output[0]
        conv_weight = params[conv.input[1]]
        conv_has_bias = len(conv.input) > 2
        conv_bias = params[conv.input[2]] if conv_has_bias else 0

        A = params[affine.input[1]]
        B = params[affine.input[2]]

        # Thus, can just have the affine transform
        # X * A + B
        # where
        # A = bn_scale * 1.0 / (sqrt(running_var + eps))
        # B =  (bias - running_mean * (1.0 / sqrt(running_var + eps))
        # * bn_scale)

        # This identify should hold if we have correctly fused
        # np.testing.assert_array_equal(
        #     params[conv.output[0]] * A + B,
        #     params[bn.output[0]])

        # Now, we have that the computation made is the following:
        # ((X `conv` W) + b) * A + B
        # Then, we can simply fuse this as follows:
        # (X `conv` (W * A)) + b * A + B
        # which is simply
        # (X `conv` Q) + C
        # where

        # Q = W * A
        # C = b * A + B

        # For ConvTranspose, from the view of convolutions as a
        # Toepeliz multiplication, we have W_ = W^T, so the weights
        # are laid out as (R, S, K, K) (vs (S, R, K, K) for a Conv),
        # so the weights broadcast slightly differently. Remember, our
        # BN scale 'B' is of size (S,)

        A_ = A.reshape(-1, 1, 1, 1) if conv.type == "Conv" else A.reshape(1, -1, 1, 1)

        C = conv_bias * A + B
        Q = conv_weight * A_

        assert params[conv.input[1]].shape == Q.shape

        params[conv.input[1]] = Q
        if conv_has_bias:
            assert params[conv.input[2]].shape == C.shape
            params[conv.input[2]] = C
        else:
            # make af_bias to be bias of the conv layer
            fused_conv.input.append(affine.input[2])
            params[affine.input[2]] = B

        new_ops = net.op[:i] + [fused_conv] + net.op[j + 1 :]
        del net.op[:]
        if conv_has_bias:
            del params[affine.input[2]]
            removed_tensors.append(affine.input[2])
        removed_tensors.append(affine.input[1])
        del params[affine.input[1]]
        net.op.extend(new_ops)
        break
    return net, params, removed_tensors


def fuse_affine(net, params, ignore_failure):
    # Run until we hit a fixed point
    removed_tensors = []
    while True:
        (next_net, next_params, removed_tensors) = fuse_first_affine(net, params, removed_tensors)
        if len(next_net.op) == len(net.op):
            if any(op.type == "AffineChannel" for op in next_net.op) and not ignore_failure:
                raise Exception("Model contains AffineChannel op after fusion: %s", next_net)
            return (next_net, next_params, removed_tensors)
        net, params, removed_tensors = (next_net, next_params, removed_tensors)


def fuse_net(fuse_func, net, blobs, ignore_failure=False):
    is_core_net = isinstance(net, core.Net)
    if is_core_net:
        net = net.Proto()

    net, params, removed_tensors = fuse_func(net, blobs, ignore_failure)
    for rt in removed_tensors:
        net.external_input.remove(rt)

    if is_core_net:
        net = core.Net(net)

    return net, params


def fuse_net_affine(net, blobs):
    return fuse_net(fuse_affine, net, blobs)


def add_tensor(net, name, blob):
    """ Create an operator to store the tensor 'blob',
        run the operator to put the blob to workspace.
        uint8 is stored as an array of string with one element.
    """
    kTypeNameMapper = {
        np.dtype("float32"): "GivenTensorFill",
        np.dtype("int32"): "GivenTensorIntFill",
        np.dtype("int64"): "GivenTensorInt64Fill",
        np.dtype("uint8"): "GivenTensorStringFill",
    }

    shape = blob.shape
    values = blob
    # pass array of uint8 as a string to save storage
    # storing uint8_t has a large overhead for now
    if blob.dtype == np.dtype("uint8"):
        shape = [1]
        values = [str(blob.data)]

    op = core.CreateOperator(
        kTypeNameMapper[blob.dtype],
        [],
        [name],
        shape=shape,
        values=values,
        # arg=[
        #     putils.MakeArgument("shape", shape),
        #     putils.MakeArgument("values", values),
        # ]
    )
    net.op.extend([op])


def gen_init_net_from_blobs(blobs, blobs_to_use=None, excluded_blobs=None):
    """ Generate an initialization net based on a blob dict """
    ret = caffe2_pb2.NetDef()
    if blobs_to_use is None:
        blobs_to_use = {x for x in blobs}
    else:
        blobs_to_use = copy.deepcopy(blobs_to_use)
    if excluded_blobs is not None:
        blobs_to_use = [x for x in blobs_to_use if x not in excluded_blobs]
    for name in blobs_to_use:
        blob = blobs[name]
        if isinstance(blob, str):
            print(
                "Blob {} with type {} is not supported in generating init net,"
                " skipped.".format(name, type(blob))
            )
            continue
        add_tensor(ret, name, blob)

    return ret


def get_ws_blobs(blob_names=None):
    """ Get blobs in 'blob_names' in the default workspace,
        get all blobs if blob_names is None """
    blobs = {}
    if blob_names is None:
        blob_names = workspace.Blobs()
    blobs = {x: workspace.FetchBlob(x) for x in blob_names}

    return blobs


def get_device_option_cpu():
    device_option = core.DeviceOption(caffe2_pb2.CPU)
    return device_option


def get_device_option_cuda(gpu_id=0):
    device_option = caffe2_pb2.DeviceOption()
    device_option.device_type = caffe2_pb2.CUDA
    device_option.device_id = gpu_id
    return device_option


def create_input_blobs_for_net(net_def):
    for op in net_def.op:
        for blob_in in op.input:
            if not workspace.HasBlob(blob_in):
                workspace.CreateBlob(blob_in)


def compare_model(model1_func, model2_func, test_image, check_blobs):
    """ model_func(test_image, check_blobs)
    """
    cb1, cb2 = check_blobs, check_blobs
    if isinstance(check_blobs, dict):
        cb1 = check_blobs.keys()
        cb2 = check_blobs.values()
    print("Running the first model...")
    res1 = model1_func(test_image, check_blobs)
    print("Running the second model...")
    res2 = model2_func(test_image, check_blobs)
    for idx in range(len(cb1)):
        print("Checking {} -> {}...".format(cb1[idx], cb2[idx]))
        n1, n2 = cb1[idx], cb2[idx]
        r1 = res1[n1] if n1 in res1 else None
        r2 = res2[n2] if n2 in res2 else None
        assert r1 is not None or r2 is None, "Blob {} in model1 is None".format(n1)
        assert r2 is not None or r1 is None, "Blob {} in model2 is None".format(n2)
        assert r1.shape == r2.shape, "Blob {} and {} shape mismatched: {} vs {}".format(
            n1, n2, r1.shape, r2.shape
        )

        np.testing.assert_array_almost_equal(
            r1,
            r2,
            decimal=3,
            err_msg="{} and {} not matched. Max diff: {}".format(
                n1, n2, np.amax(np.absolute(r1 - r2))
            ),
        )

    return True


# graph_name could not contain word 'graph'
def save_graph(net, file_name, graph_name="net", op_only=True):
    from caffe2.python import net_drawer

    graph = None
    ops = net.op
    if not op_only:
        graph = net_drawer.GetPydotGraph(ops, graph_name, rankdir="TB")
    else:
        graph = net_drawer.GetPydotGraphMinimal(
            ops, graph_name, rankdir="TB", minimal_dependency=True
        )

    try:
        graph.write_png(file_name)
    except Exception as e:
        print("Error when writing graph to image {}".format(e))


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
