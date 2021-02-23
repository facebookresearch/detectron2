import collections
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple
import torch
from torch import nn

from detectron2.structures import Boxes, Instances


@dataclass
class Schema:
    """
    A Schema defines how to flatten a possibly hierarchical object into tuple of
    primitive objects, so it can be used as inputs/outputs of PyTorch's tracing.

    PyTorch does not support tracing a function that produces rich output
    structures (e.g. dict, Instances, Boxes). To trace such a function, we
    flatten the rich object into tuple of tensors, and return this tuple of tensors
    instead. Meanwhile, we also need to know how to "rebuild" the original object
    from the flattened results, so we can evaluate the flattened results.
    A Schema defines how to flatten an object, and while flattening it, it records
    necessary schemas so that the object can be rebuilt using the flattened outputs.

    The flattened object and the schema object is returned by ``.flatten`` classmethod.
    Then the original object can be rebuilt with the ``__call__`` method of schema.

    A Schema is a dataclass that can be serialized easily.
    """

    # inspired by FetchMapper in tensorflow/python/client/session.py

    @classmethod
    def flatten(cls, obj):
        raise NotImplementedError

    def __call__(self, values):
        raise NotImplementedError

    @staticmethod
    def _concat(values):
        ret = ()
        idx_mapping = []
        for v in values:
            assert isinstance(v, tuple), "Flattened results must be a tuple"
            oldlen = len(ret)
            ret = ret + v
            idx_mapping.append([oldlen, len(ret)])
        return ret, idx_mapping

    @staticmethod
    def _split(values, idx_mapping):
        if len(idx_mapping):
            expected_len = idx_mapping[-1][-1]
            assert (
                len(values) == expected_len
            ), f"Values has length {len(values)} but expect length {expected_len}."
        ret = []
        for (start, end) in idx_mapping:
            ret.append(values[start:end])
        return ret


@dataclass
class ListSchema(Schema):
    schemas: List[Schema]
    idx_mapping: List[List[int]]
    is_tuple: bool

    def __call__(self, values):
        values = self._split(values, self.idx_mapping)
        if len(values) != len(self.schemas):
            raise ValueError(
                f"Values has length {len(values)} but schemas " f"has length {len(self.schemas)}!"
            )
        values = [m(v) for m, v in zip(self.schemas, values)]
        return list(values) if not self.is_tuple else tuple(values)

    @classmethod
    def flatten(cls, obj):
        is_tuple = isinstance(obj, tuple)
        res = [flatten_to_tuple(k) for k in obj]
        values, idx = cls._concat([k[0] for k in res])
        return values, cls([k[1] for k in res], idx, is_tuple)


@dataclass
class IdentitySchema(Schema):
    def __call__(self, values):
        return values[0]

    @classmethod
    def flatten(cls, obj):
        return (obj,), cls()


@dataclass
class DictSchema(Schema):
    keys: List[str]
    value_schema: ListSchema

    def __call__(self, values):
        values = self.value_schema(values)
        return dict(zip(self.keys, values))

    @classmethod
    def flatten(cls, obj):
        for k in obj.keys():
            if not isinstance(k, str):
                raise KeyError("Only support flattening dictionaries if keys are str.")
        keys = sorted(obj.keys())
        values = [obj[k] for k in keys]
        ret, schema = ListSchema.flatten(values)
        return ret, cls(keys, schema)


@dataclass
class InstancesSchema(Schema):
    field_names: List[str]
    field_schema: ListSchema

    def __call__(self, values):
        image_size, fields = values[-1], values[:-1]
        fields = self.field_schema(fields)
        fields = dict(zip(self.field_names, fields))
        return Instances(image_size, **fields)

    @classmethod
    def flatten(cls, obj):
        field_names = sorted(obj.get_fields().keys())
        values = [obj.get(f) for f in field_names]
        ret, schema = ListSchema.flatten(values)
        size = obj.image_size
        if not isinstance(size, torch.Tensor):
            size = torch.tensor(size)
        return ret + (size,), cls(field_names, schema)


@dataclass
class BoxesSchema(Schema):
    def __call__(self, values):
        return Boxes(values[0])

    @classmethod
    def flatten(cls, obj):
        return (obj.tensor,), cls()


# if more custom structures needed in the future, can allow
# passing in extra schemas for custom types
def flatten_to_tuple(obj):
    """
    Flatten an object so it can be used for PyTorch tracing.
    Also returns how to rebuild the original object from the flattened outputs.

    Returns:
        res (tuple): the flattened results that can be used as tracing outputs
        schema: an object with a ``__call__`` method such that ``schema(res) == obj``.
             It is a pure dataclass that can be serialized.
    """
    schemas = [
        ((str, bytes), IdentitySchema),
        (collections.abc.Sequence, ListSchema),
        (collections.abc.Mapping, DictSchema),
        (Instances, InstancesSchema),
        (Boxes, BoxesSchema),
    ]
    for klass, schema in schemas:
        if isinstance(obj, klass):
            F = schema
            break
    else:
        F = IdentitySchema

    return F.flatten(obj)


class TracingAdapter(nn.Module):
    """
    A model may take rich input/output format (e.g. dict or custom classes).
    This adapter flattens input/output format of a model so it becomes traceable.

    It also records the necessary schema to rebuild model's inputs/outputs from flattened
    inputs/outputs.

    Example:

    ::
        outputs = model(inputs)   # inputs/outputs may be rich structure
        adapter = TracingAdapter(model, inputs)

        # can now trace the model, with adapter.flattened_inputs, or another
        # tuple of tensors with the same length and meaning
        traced = torch.jit.trace(adapter, adapter.flattened_inputs)

        # traced model can only produce flattened outputs (tuple of tensors)
        flattened_outputs = traced(*adapter.flattened_inputs)
        # adapter knows the schema to convert it back (new_outputs == outputs)
        new_outputs = adapter.outputs_schema(flattened_outputs)
    """

    flattened_inputs: Tuple[torch.Tensor] = None
    """
    Flattened version of inputs given to this class's constructor.
    """

    inputs_schema: Schema = None
    """
    Schema of the inputs given to this class's constructor.
    """

    outputs_schema: Schema = None
    """
    Schema of the output produced by calling the given model with inputs.
    """

    def __init__(self, model: nn.Module, inputs, inference_func: Optional[Callable] = None):
        """
        Args:
            model: an nn.Module
            inputs: An input argument or a tuple of input arguments used to call model.
                After flattening, it has to only consist of tensors.
            inference_func: a callable that takes (model, *inputs), calls the
                model with inputs, and return outputs. By default it
                is ``lambda model, *inputs: model(*inputs)``. Can be override
                if you need to call the model differently.
        """
        super().__init__()
        if isinstance(model, (nn.parallel.distributed.DistributedDataParallel, nn.DataParallel)):
            model = model.module
        self.model = model
        if not isinstance(inputs, tuple):
            inputs = (inputs,)
        self.inputs = inputs

        if inference_func is None:
            inference_func = lambda model, *inputs: model(*inputs)  # noqa
        self.inference_func = inference_func

        self.flattened_inputs, self.inputs_schema = flatten_to_tuple(inputs)
        for input in self.flattened_inputs:
            if not isinstance(input, torch.Tensor):
                raise ValueError(
                    f"Inputs for tracing must only contain tensors. Got a {type(input)} instead."
                )

    def forward(self, *args: torch.Tensor):
        with torch.no_grad():
            inputs_orig_format = self.inputs_schema(args)
            outputs = self.inference_func(self.model, *inputs_orig_format)
            flattened_outputs, schema = flatten_to_tuple(outputs)
            if self.outputs_schema is None:
                self.outputs_schema = schema
            else:
                assert (
                    self.outputs_schema == schema
                ), "Model should always return outputs with the same structure so it can be traced!"
            return flattened_outputs

    def _create_wrapper(self, traced_model):
        """
        Return a function that has an input/output interface the same as the
        original model, but it calls the given traced model under the hood.
        """

        def forward(*args):
            flattened_inputs, _ = flatten_to_tuple(args)
            flattened_outputs = traced_model(*flattened_inputs)
            return self.outputs_schema(flattened_outputs)

        return forward
