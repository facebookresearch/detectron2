import copy
import io
import numpy as np
import os
import unittest
import warnings
import torch
from torch.onnx.utils import unpack_quantized_tensor

from detectron2 import model_zoo
from detectron2.export.flatten import TracingAdapter
from detectron2.utils.testing import get_sample_coco_image

from .helper import skipIfUnsupportedMinOpsetVersion

try:
    # Make sure ORT has https://github.com/microsoft/onnxruntime/pull/8564
    import onnxruntime
except ImportError:
    raise unittest.SkipTest("Skipping all tests in {__file__}. ONNX Runtime not installed")


class _TestONNXRuntime:

    opset_version = -1  # Sub-classes must override
    keep_initializers_as_inputs = True  # For IR version 3 type export.

    def setUp(self):
        torch.manual_seed(0)
        onnxruntime.set_seed(0)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(0)
        np.random.seed(seed=0)
        os.environ["ALLOW_RELEASED_ONNX_OPSET_ONLY"] = "0"

    def run_test(
        self,
        model,
        input,
        rtol=1e-3,
        atol=1e-7,
        do_constant_folding=True,
        batch_size=1,
        use_gpu=True,
        dynamic_axes=None,
        test_with_inputs=None,
        input_names=None,
        output_names=None,
        dict_check=True,
        training=None,
        remained_onnx_input_idx=None,
        flatten=True,
        verbose=False,
    ):
        """Export PyTorch to ONNX in tracing mode and run it with ONNX Runtime backend.

        Optionally, it can also assert the results with PyTorch's outputs.

        The ONNX model may have less inputs than the pytorch model because of const folding.
        This mostly happens in unit test, due to the widely use of `torch.size` or `torch.shape`.
        In these cases, output is only dependent on the input shape, not value.
        `remained_onnx_input_idx` is used to indicate which pytorch model input idx
        is remained in ONNX model.
        """
        # Only tracing tests are supported
        assert not isinstance(model, (torch.jit.ScriptModule, torch.jit.ScriptFunction))
        run_model_test(
            self,
            model,
            batch_size=batch_size,
            input=input,
            use_gpu=use_gpu,
            rtol=rtol,
            atol=atol,
            do_constant_folding=do_constant_folding,
            dynamic_axes=dynamic_axes,
            test_with_inputs=test_with_inputs,
            input_names=input_names,
            output_names=output_names,
            dict_check=dict_check,
            training=training,
            remained_onnx_input_idx=remained_onnx_input_idx,
            flatten=flatten,
            verbose=verbose,
        )

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_detectron2_mask_rcnnfpn(self):
        def _test_model_zoo_from_config_path(config_path, inference_func, batch=1):
            model = model_zoo.get(config_path, trained=True)
            image = get_sample_coco_image()
            inputs = tuple(image.clone() for _ in range(batch))
            inputs_copy = copy.deepcopy(inputs)
            adapter_model = TracingAdapter(model, inputs, inference_func)
            adapter_model.eval()
            torch.onnx.enable_log()
            self.keep_initializers_as_inputs = None
            self.run_test(
                adapter_model,
                (adapter_model.flattened_inputs),
                training=torch.onnx.TrainingMode.EVAL,
                batch_size=batch,
                do_constant_folding=False,
                test_with_inputs=[inputs_copy],
                input_names=["images"],
                output_names=["pred_boxes", "pred_classes", "pred_masks", "scores"],
                dynamic_axes={
                    "images": {0: "channels", 1: "height", 2: "width"},
                    "pred_boxes": {0: "batch", 1: "static_four"},
                    "pred_classes": {0: "batch"},
                    "scores": {0: "batch"},
                    "pred_masks": {
                        0: "batch",
                        1: "static_one",
                        2: "pred_masks_size",
                        3: "pred_masks_size",
                    },
                },
            )

        def inference_func(model, images):
            inputs = [{"image": image} for image in images]
            inst = model.inference(inputs, do_postprocess=False)[0]
            return [{"instances": inst}]

        _test_model_zoo_from_config_path(
            "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml", inference_func
        )


################################################################################
# Testcase setup - DO NOT add tests below this point
################################################################################


_ORT_PROVIDERS = ["CPUExecutionProvider", "CUDAExecutionProvider"]


def flatten_tuples(elem):
    flattened = []
    for t in elem:
        if isinstance(t, tuple):
            flattened.extend(flatten_tuples(t))
        else:
            flattened.append(t)
    return flattened


def to_numpy(elem):
    if isinstance(elem, torch.Tensor):
        if elem.requires_grad:
            return elem.detach().cpu().numpy()
        else:
            return elem.cpu().numpy()
    elif isinstance(elem, (list, tuple)):
        return [to_numpy(inp) for inp in elem]
    elif isinstance(elem, bool):
        return np.array(elem, dtype=bool)
    elif isinstance(elem, int):
        return np.array(elem, dtype=int)
    elif isinstance(elem, float):
        return np.array(elem, dtype=float)
    elif isinstance(elem, dict):
        flattened = []
        for k in elem:
            flattened += [to_numpy(k)] + [to_numpy(elem[k])]
        return flattened
    return elem


def convert_to_onnx(
    model,
    input=None,
    opset_version=11,
    do_constant_folding=True,
    keep_initializers_as_inputs=True,
    dynamic_axes=None,
    input_names=None,
    output_names=None,
    training=None,
    verbose=False,
):
    f = io.BytesIO()
    input_copy = copy.deepcopy(input)
    torch.onnx.export(
        model,
        input_copy,
        f,
        opset_version=opset_version,
        do_constant_folding=do_constant_folding,
        keep_initializers_as_inputs=keep_initializers_as_inputs,
        dynamic_axes=dynamic_axes,
        input_names=input_names,
        output_names=output_names,
        training=training,
        verbose=verbose,
    )

    # compute onnxruntime output prediction
    so = onnxruntime.SessionOptions()
    # suppress ort warnings.
    # 0:Verbose, 1:Info, 2:Warning. 3:Error, 4:Fatal. Default is 2.
    so.log_severity_level = 2
    ort_sess = onnxruntime.InferenceSession(f.getvalue(), so, providers=_ORT_PROVIDERS)
    return ort_sess


def inline_flatten_list(inputs, res_list):
    for i in inputs:
        res_list.append(i) if not isinstance(i, (list, tuple)) else inline_flatten_list(i, res_list)
    return res_list


def unpack_to_numpy(values):
    value_unpacked = []
    for value in values:
        value_unpacked.extend(unpack_quantized_tensor(value))
    return [to_numpy(v) for v in value_unpacked]


def run_ort(ort_sess, inputs):
    kw_inputs = {}
    if inputs and isinstance(inputs[-1], dict):
        kw_inputs = inputs[-1]
        inputs = inputs[:-1]
    inputs = unpack_to_numpy(flatten_tuples(inputs))
    ort_inputs = {}
    for input_name, input in kw_inputs.items():
        ort_inputs[input_name] = to_numpy(input)
    inputs = to_numpy(inputs)
    ort_sess_inputs = ort_sess.get_inputs()
    for i, input in enumerate(inputs):
        if i == len(ort_sess_inputs) or ort_sess_inputs[i].name in ort_inputs:
            raise ValueError(
                f"got too many positional inputs. inputs: {inputs}. kw_inputs: {kw_inputs}"
            )
        ort_inputs[ort_sess_inputs[i].name] = input
    ort_outs = ort_sess.run(None, ort_inputs)
    return inline_flatten_list(ort_outs, [])


def ort_compare_with_pytorch(ort_outs, output, rtol, atol):
    output, _ = torch.jit._flatten(output)
    outputs = unpack_to_numpy(output)

    # compare onnxruntime and PyTorch results
    assert len(outputs) == len(ort_outs), "number of outputs differ"

    # compare onnxruntime and PyTorch results
    [
        np.testing.assert_allclose(out, ort_out, rtol=rtol, atol=atol)
        for out, ort_out in zip(outputs, ort_outs)
    ]


def run_model_test(
    self,
    model,
    batch_size=1,
    state_dict=None,
    input=None,
    use_gpu=True,
    rtol=0.001,
    atol=1e-7,
    do_constant_folding=True,
    dynamic_axes=None,
    test_with_inputs=None,
    input_names=None,
    output_names=None,
    dict_check=True,
    training=None,
    remained_onnx_input_idx=None,
    flatten=True,
    verbose=False,
):
    if training is not None and training == torch.onnx.TrainingMode.TRAINING:
        model.train()
    elif training is None or training == torch.onnx.TrainingMode.EVAL:
        model.eval()
    if input is None:
        input = torch.randn(batch_size, 3, 224, 224, requires_grad=True)
    with torch.no_grad():
        if isinstance(input, (torch.Tensor, dict)):
            input = (input,)
        # In-place operators will update input tensor data as well.
        # Thus inputs are replicated before every forward call.
        input_args = copy.deepcopy(input)
        input_kwargs = {}
        if dict_check and isinstance(input_args[-1], dict):
            input_kwargs = input_args[-1]
            input_args = input_args[:-1]
        try:
            model_copy = copy.deepcopy(model)
            output = model_copy(*input_args, **input_kwargs)
        except Exception:
            warnings.warn(
                "Original model will be used because it could not be deep copied."
                " Tracing can be incorrect!"
            )
            output = model(*input_args, **input_kwargs)
        if isinstance(output, torch.Tensor):
            output = (output,)

        if not dict_check and isinstance(input[-1], dict):
            input = input + ({},)

        ort_sess = convert_to_onnx(
            model,
            input=input,
            opset_version=self.opset_version,
            do_constant_folding=do_constant_folding,
            keep_initializers_as_inputs=self.keep_initializers_as_inputs,
            dynamic_axes=dynamic_axes,
            input_names=input_names,
            output_names=output_names,
            training=training,
            verbose=verbose,
        )
        # compute onnxruntime output prediction
        if remained_onnx_input_idx is not None:
            input_onnx = []
            for idx in remained_onnx_input_idx:
                input_onnx.append(input[idx])
            input = input_onnx

        input_copy = copy.deepcopy(input)
        if flatten:
            input_copy, _ = torch.jit._flatten(input_copy)
        elif input_copy and input_copy[-1] == {}:
            # Handle empty kwargs (normally removed by flatten).
            input_copy = input_copy[:-1]
        ort_outs = run_ort(ort_sess, input_copy)
        ort_compare_with_pytorch(ort_outs, output, rtol, atol)

        # if additional test inputs are provided run the onnx
        # model with these inputs and check the outputs
        if test_with_inputs is not None:
            for test_input in test_with_inputs:
                if isinstance(test_input, torch.Tensor):
                    test_input = (test_input,)
                test_input_copy = copy.deepcopy(test_input)
                output = model(*test_input_copy)
                if isinstance(output, torch.Tensor):
                    output = (output,)
                if remained_onnx_input_idx is not None:
                    test_input_onnx = []
                    for idx in remained_onnx_input_idx:
                        test_input_onnx.append(test_input[idx])
                    test_input = test_input_onnx
                if flatten:
                    test_input, _ = torch.jit._flatten(test_input)
                ort_outs = run_ort(ort_sess, test_input)
                ort_compare_with_pytorch(ort_outs, output, rtol, atol)


def MakeTestCase(opset_version: int, keep_initializers_as_inputs: bool = True) -> type:
    name = f"TestONNXRuntime_opset{opset_version}"
    if not keep_initializers_as_inputs:
        name += "_IRv4"
    return type(
        str(name),
        (unittest.TestCase,),
        dict(
            _TestONNXRuntime.__dict__,
            opset_version=opset_version,
            keep_initializers_as_inputs=keep_initializers_as_inputs,
        ),
    )


TestONNXRuntime_opset11 = MakeTestCase(11)
TestONNXRuntime_opset11_IRv4 = MakeTestCase(11, keep_initializers_as_inputs=False)
TestONNXRuntime_opset12 = MakeTestCase(12)
TestONNXRuntime_opset12_IRv4 = MakeTestCase(12, keep_initializers_as_inputs=False)
TestONNXRuntime_opset13 = MakeTestCase(13, keep_initializers_as_inputs=False)
TestONNXRuntime_opset14 = MakeTestCase(14, keep_initializers_as_inputs=False)
TestONNXRuntime_opset15 = MakeTestCase(15, keep_initializers_as_inputs=False)
TestONNXRuntime_opset16 = MakeTestCase(16, keep_initializers_as_inputs=False)


if __name__ == "__main__":
    unittest.main()
