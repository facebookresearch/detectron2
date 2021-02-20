import collections
from dataclasses import dataclass
from typing import List
import torch

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
