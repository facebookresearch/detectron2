"""This file contains PyTorch patches that are needed during Test pipeline only.

Periodically, especially when new/older PyTorch versions are supported/deprecated,
a cleanup should be done to remove unnecessary patches.
"""
import torch
import torch.onnx.symbolic_helper as sym_help
from torch._C import ListType


# TODO: Remove after PyTorch 1.11.1+ is used by detectron2's CI
def _pytorch1111_symbolic_opset9_to(g, self, *args):
    """aten::to() symbolic that must be used for testing with PyTorch < 1.11.1."""

    def is_aten_to_device_only(args):
        if len(args) == 4:
            # aten::to(Tensor, Device, bool, bool, memory_format)
            return (
                args[0].node().kind() == "prim::device"
                or args[0].type().isSubtypeOf(ListType.ofInts())
                or (
                    sym_help._is_value(args[0])
                    and args[0].node().kind() == "onnx::Constant"
                    and isinstance(args[0].node()["value"], str)
                )
            )
        elif len(args) == 5:
            # aten::to(Tensor, Device, ScalarType, bool, bool, memory_format)
            # When dtype is None, this is a aten::to(device) call
            dtype = sym_help._get_const(args[1], "i", "dtype")
            return dtype is None
        elif len(args) in (6, 7):
            # aten::to(Tensor, ScalarType, Layout, Device, bool, bool, memory_format)
            # aten::to(Tensor, ScalarType, Layout, Device, bool, bool, bool, memory_format)
            # When dtype is None, this is a aten::to(device) call
            dtype = sym_help._get_const(args[0], "i", "dtype")
            return dtype is None
        return False

    # ONNX doesn't have a concept of a device, so we ignore device-only casts
    if is_aten_to_device_only(args):
        return self

    if len(args) == 4:
        # TestONNXRuntime::test_ones_bool shows args[0] of aten::to can be onnx::Constant[Tensor]
        # In this case, the constant value is a tensor not int,
        # so sym_help._maybe_get_const(args[0], 'i') would not work.
        dtype = args[0]
        if sym_help._is_value(args[0]) and args[0].node().kind() == "onnx::Constant":
            tval = args[0].node()["value"]
            if isinstance(tval, torch.Tensor):
                if len(tval.shape) == 0:
                    tval = tval.item()
                    dtype = int(tval)
                else:
                    dtype = tval

        if sym_help._is_value(dtype) or isinstance(dtype, torch.Tensor):
            # aten::to(Tensor, Tensor, bool, bool, memory_format)
            dtype = args[0].type().scalarType()
            return g.op("Cast", self, to_i=sym_help.cast_pytorch_to_onnx[dtype])
        else:
            # aten::to(Tensor, ScalarType, bool, bool, memory_format)
            # memory_format is ignored
            return g.op("Cast", self, to_i=sym_help.scalar_type_to_onnx[dtype])
    elif len(args) == 5:
        # aten::to(Tensor, Device, ScalarType, bool, bool, memory_format)
        dtype = sym_help._get_const(args[1], "i", "dtype")
        # memory_format is ignored
        return g.op("Cast", self, to_i=sym_help.scalar_type_to_onnx[dtype])
    elif len(args) == 6:
        # aten::to(Tensor, ScalarType, Layout, Device, bool, bool, memory_format)
        dtype = sym_help._get_const(args[0], "i", "dtype")
        # Layout, device and memory_format are ignored
        return g.op("Cast", self, to_i=sym_help.scalar_type_to_onnx[dtype])
    elif len(args) == 7:
        # aten::to(Tensor, ScalarType, Layout, Device, bool, bool, bool, memory_format)
        dtype = sym_help._get_const(args[0], "i", "dtype")
        # Layout, device and memory_format are ignored
        return g.op("Cast", self, to_i=sym_help.scalar_type_to_onnx[dtype])
    else:
        return sym_help._onnx_unsupported("Unknown aten::to signature")


# TODO: Remove after PyTorch 1.11.1+ is used by detectron2's CI
def _pytorch1111_symbolic_opset9_repeat_interleave(g, self, repeats, dim=None, output_size=None):

    # from torch.onnx.symbolic_helper import ScalarType
    from torch.onnx.symbolic_opset9 import expand, unsqueeze

    input = self
    # if dim is None flatten
    # By default, use the flattened input array, and return a flat output array
    if sym_help._is_none(dim):
        input = sym_help._reshape_helper(g, self, g.op("Constant", value_t=torch.tensor([-1])))
        dim = 0
    else:
        dim = sym_help._maybe_get_scalar(dim)

    repeats_dim = sym_help._get_tensor_rank(repeats)
    repeats_sizes = sym_help._get_tensor_sizes(repeats)
    input_sizes = sym_help._get_tensor_sizes(input)
    if repeats_dim is None:
        raise RuntimeError(
            "Unsupported: ONNX export of repeat_interleave for unknown " "repeats rank."
        )
    if repeats_sizes is None:
        raise RuntimeError(
            "Unsupported: ONNX export of repeat_interleave for unknown " "repeats size."
        )
    if input_sizes is None:
        raise RuntimeError(
            "Unsupported: ONNX export of repeat_interleave for unknown " "input size."
        )

    input_sizes_temp = input_sizes.copy()
    for idx, input_size in enumerate(input_sizes):
        if input_size is None:
            input_sizes[idx], input_sizes_temp[idx] = 0, -1

    # Cases where repeats is an int or single value tensor
    if repeats_dim == 0 or (repeats_dim == 1 and repeats_sizes[0] == 1):
        if not sym_help._is_tensor(repeats):
            repeats = g.op("Constant", value_t=torch.LongTensor(repeats))
        if input_sizes[dim] == 0:
            return sym_help._onnx_opset_unsupported_detailed(
                "repeat_interleave",
                9,
                13,
                "Unsupported along dimension with unknown input size",
            )
        else:
            reps = input_sizes[dim]
            repeats = expand(g, repeats, g.op("Constant", value_t=torch.tensor([reps])), None)

    # Cases where repeats is a 1 dim Tensor
    elif repeats_dim == 1:
        if input_sizes[dim] == 0:
            return sym_help._onnx_opset_unsupported_detailed(
                "repeat_interleave",
                9,
                13,
                "Unsupported along dimension with unknown input size",
            )
        if repeats_sizes[0] is None:
            return sym_help._onnx_opset_unsupported_detailed(
                "repeat_interleave", 9, 13, "Unsupported for cases with dynamic repeats"
            )
        assert (
            repeats_sizes[0] == input_sizes[dim]
        ), "repeats must have the same size as input along dim"
        reps = repeats_sizes[0]
    else:
        raise RuntimeError("repeats must be 0-dim or 1-dim tensor")

    final_splits = list()
    r_splits = sym_help._repeat_interleave_split_helper(g, repeats, reps, 0)
    if isinstance(r_splits, torch._C.Value):
        r_splits = [r_splits]
    i_splits = sym_help._repeat_interleave_split_helper(g, input, reps, dim)
    if isinstance(i_splits, torch._C.Value):
        i_splits = [i_splits]
    input_sizes[dim], input_sizes_temp[dim] = -1, 1
    for idx, r_split in enumerate(r_splits):
        i_split = unsqueeze(g, i_splits[idx], dim + 1)
        r_concat = [
            g.op("Constant", value_t=torch.LongTensor(input_sizes_temp[: dim + 1])),
            r_split,
            g.op("Constant", value_t=torch.LongTensor(input_sizes_temp[dim + 1 :])),
        ]
        r_concat = g.op("Concat", *r_concat, axis_i=0)
        i_split = expand(g, i_split, r_concat, None)
        i_split = sym_help._reshape_helper(
            g,
            i_split,
            g.op("Constant", value_t=torch.LongTensor(input_sizes)),
            allowzero=0,
        )
        final_splits.append(i_split)
    return g.op("Concat", *final_splits, axis_i=dim)
