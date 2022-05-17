# noqa D100
from typing import Callable
import torch
from packaging import version
from torch.onnx import register_custom_op_symbolic


def min_torch_version(min_version: str) -> bool:
    """Returns True when torch's  version is at least `min_version`."""
    try:
        import torch
    except ImportError:
        return False

    installed_version = version.parse(torch.__version__.split("+")[0])
    min_version = version.parse(min_version)
    return installed_version >= min_version


def _register_custom_op_onnx_export(
    opname: str, symbolic_fn: Callable, opset_version: int, min_version: str
) -> None:
    """Register `symbolic_fn` as PyTorch's symbolic `opname`-`opset_version` for ONNX export.

    The registration is performed only when current PyTorch's version is < `min_version.`
    IMPORTANT: symbolic must be manually unregistered after the caller function returns
    """
    if min_torch_version(min_version):
        print(
            f"_register_custom_op_onnx_export({opname}, {opset_version}) will be skipped."
            f" Installed PyTorch {torch.__version__} >= {min_version}."
        )
        return
    register_custom_op_symbolic(opname, symbolic_fn, opset_version)
    print(f"_register_custom_op_onnx_export({opname}, {opset_version}) succeeded.")


def _unregister_custom_op_onnx_export(opname: str, opset_version: int, min_version: str) -> None:
    """Unregister PyTorch's symbolic `opname`-`opset_version` for ONNX export.

    The un-registration is performed only when PyTorch's version is < `min_version`
    IMPORTANT: The symbolic must have been manually registered by the caller, otherwise
               the incorrect symbolic may be unregistered instead.
    """
    if min_torch_version(min_version):
        print(
            f"_unregister_custom_op_onnx_export({opname}, {opset_version}) will be skipped."
            f" Installed PyTorch {torch.__version__} >= {min_version}."
        )
        return
    _unregister_custom_op_symbolic(opname, opset_version)
    print(f"_unregister_custom_op_onnx_export({opname}, {opset_version}) succeeded.")


# TODO: _unregister_custom_op_symbolic is introduced PyTorch>=1.10
#       Remove after PyTorch 1.10+ is used by ALL detectron2's CI
try:
    from torch.onnx import unregister_custom_op_symbolic as _unregister_custom_op_symbolic
except ImportError:

    def _unregister_custom_op_symbolic(symbolic_name, opset_version):
        import torch.onnx.symbolic_registry as sym_registry
        from torch.onnx.utils import get_ns_op_name_from_custom_op
        from torch.onnx.symbolic_helper import _onnx_main_opset, _onnx_stable_opsets

        ns, op_name = get_ns_op_name_from_custom_op(symbolic_name)

        for ver in _onnx_stable_opsets + [_onnx_main_opset]:
            if ver >= opset_version:
                sym_registry.unregister_op(op_name, ns, ver)
