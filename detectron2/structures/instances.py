# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import itertools
from typing import Any, Dict, List, Tuple, Union
import torch

from detectron2.layers import cat


class Instances:
    """
    This class represents a list of instances in an image.
    It stores the attributes of instances (e.g., boxes, masks, labels, scores) as "fields".
    All fields must have the same `__len__` which is the number of instances.

    All other (non-field) attributes of this class are considered private:
    they must start with '_' and are not modifiable by a user.

    Some basic usage:

    1. Set/Get a field:
       instances.gt_boxes = Boxes(...)
       print(instances.pred_masks)
       print('gt_masks' in instances)
    2. `len(instances)` returns the number of instances
    3. Indexing: `instances[indices]` will apply the indexing on all the fields
       and returns a new `Instances`.
       Typically, `indices` is a binary vector of length num_instances,
       or a vector of integer indices.
    """

    def __init__(self, image_size: Tuple[int, int], **kwargs: Any):
        """
        Args:
            image_size (height, width): the spatial size of the image.
            kwargs: fields to add to this `Instances`.
        """
        self._image_size = image_size
        self._fields: Dict[str, Any] = {}
        for k, v in kwargs.items():
            self.set(k, v)

    @property
    def image_size(self) -> Tuple[int, int]:
        """
        Returns:
            tuple: height, width
        """
        return self._image_size

    def __setattr__(self, name: str, val: Any) -> None:
        if name.startswith("_"):
            super().__setattr__(name, val)
        else:
            self.set(name, val)

    def __getattr__(self, name: str) -> Any:
        if name == "_fields" or name not in self._fields:
            raise AttributeError("Cannot find field '{}' in the given Instances!".format(name))
        return self._fields[name]

    def set(self, name: str, value: Any) -> None:
        """
        Set the field named `name` to `value`.
        The length of `value` must be the number of instances,
        and must agree with other existing fields in this object.
        """
        data_len = len(value)
        if len(self._fields):
            assert (
                len(self) == data_len
            ), "Adding a field of length {} to a Instances of length {}".format(data_len, len(self))
        self._fields[name] = value

    def has(self, name: str) -> bool:
        """
        Returns:
            bool: whether the field called `name` exists.
        """
        return name in self._fields

    def remove(self, name: str) -> None:
        """
        Remove the field called `name`.
        """
        del self._fields[name]

    def get(self, name: str) -> Any:
        """
        Returns the field called `name`.
        """
        return self._fields[name]

    def get_fields(self) -> Dict[str, Any]:
        """
        Returns:
            dict: a dict which maps names (str) to data of the fields

        Modifying the returned dict will modify this instance.
        """
        return self._fields

    # Tensor-like methods
    def to(self, device: str) -> "Instances":
        """
        Returns:
            Instances: all fields are called with a `to(device)`, if the field has this method.
        """
        ret = Instances(self._image_size)
        for k, v in self._fields.items():
            if hasattr(v, "to"):
                v = v.to(device)
            ret.set(k, v)
        return ret

    def __getitem__(self, item: Union[int, slice, torch.BoolTensor]) -> "Instances":
        """
        Args:
            item: an index-like object and will be used to index all the fields.

        Returns:
            If `item` is a string, return the data in the corresponding field.
            Otherwise, returns an `Instances` where all fields are indexed by `item`.
        """
        ret = Instances(self._image_size)
        for k, v in self._fields.items():
            ret.set(k, v[item])
        return ret

    def __len__(self) -> int:
        for v in self._fields.values():
            return len(v)
        raise NotImplementedError("Empty Instances does not support __len__!")

    @staticmethod
    def cat(instance_lists: List["Instances"]) -> "Instances":
        """
        Args:
            instance_lists (list[Instances])

        Returns:
            Instances
        """
        assert all(isinstance(i, Instances) for i in instance_lists)
        assert len(instance_lists) > 0
        if len(instance_lists) == 1:
            return instance_lists[0]

        image_size = instance_lists[0].image_size
        for i in instance_lists[1:]:
            assert i.image_size == image_size
        ret = Instances(image_size)
        for k in instance_lists[0]._fields.keys():
            values = [i.get(k) for i in instance_lists]
            v0 = values[0]
            if isinstance(v0, torch.Tensor):
                values = cat(values, dim=0)
            elif isinstance(v0, list):
                values = list(itertools.chain(*values))
            elif hasattr(type(v0), "cat"):
                values = type(v0).cat(values)
            else:
                raise ValueError("Unsupported type {} for concatenation".format(type(v0)))
            ret.set(k, values)
        return ret

    def __str__(self) -> str:
        s = self.__class__.__name__ + "("
        s += "num_instances={}, ".format(len(self))
        s += "image_height={}, ".format(self._image_size[0])
        s += "image_width={}, ".format(self._image_size[1])
        s += "fields=[{}])".format(", ".join(self._fields.keys()))
        return s

    def __repr__(self) -> str:
        s = self.__class__.__name__ + "("
        s += "num_instances={}, ".format(len(self))
        s += "image_height={}, ".format(self._image_size[0])
        s += "image_width={}, ".format(self._image_size[1])
        s += "fields=["
        for k, v in self._fields.items():
            s += "{} = {}, ".format(k, v)
        s += "])"
        return s

    def classes_reindex(self, index_lists, idx):
        if idx == 0: index_tensor = torch.LongTensor(index_lists[0])
        if idx == 1: index_tensor = torch.LongTensor(index_lists[1])
        if idx == 2: index_tensor = torch.LongTensor(index_lists[2])
        new_pred_classes = index_tensor[self._fields['pred_classes']] - 1
        self._fields['pred_classes'] = new_pred_classes
        
    def classes_fcr_reindex(self, index_lists, num_classes):
#         print(num_classes) #315, 776, 1230
#         index_tensor_f = torch.LongTensor(index_lists[0])
#         index_tensor_c = torch.LongTensor(index_lists[1])
#         index_tensor_r = torch.LongTensor(index_lists[2])
        pred_re_classes = []
#         freq_mask = (self._fields['pred_classes'] >= 0) * (self._fields['pred_classes'] < num_classes[0]) 
#         comm_mask = (self._fields['pred_classes'] >= num_classes[0]) * (self._fields['pred_classes'] < num_classes[1]) 
#         rare_mask = (self._fields['pred_classes'] >= num_classes[1]) * (self._fields['pred_classes'] < num_classes[2])
#         self._fields['pred_classes'][freq_mask] = index_tensor_f[self._fields['pred_classes'][freq_mask]] - 1
#         self._fields['pred_classes'][comm_mask] = index_tensor_c[self._fields['pred_classes'][comm_mask] - num_classes[0]] - 1
#         self._fields['pred_classes'][rare_mask] = index_tensor_r[self._fields['pred_classes'][rare_mask] - num_classes[1]] - 1
#         del index_tensor_f
#         del index_tensor_c
#         del index_tensor_r

        for i, pred_class in enumerate(self._fields['pred_classes']):
            if pred_class >= 0 and pred_class < num_classes[0]: # frequent
                new_pred_classes = index_lists[0][pred_class] - 1
            elif pred_class >= num_classes[0] and pred_class < num_classes[1]: # common
                new_pred_classes = index_lists[1][pred_class - (num_classes[0])] - 1
            else:
                if pred_class == num_classes[2]:
                    new_pred_classes = num_classes[2]
                else:
                    new_pred_classes = index_lists[2][pred_class - (num_classes[1])] - 1
            pred_re_classes.append(new_pred_classes)            
        self._fields['pred_classes'] = torch.LongTensor(pred_re_classes)
        
    def classes_shifter(self, num_shift, train = True):
        if train:
            self._fields['gt_classes'] = self._fields['gt_classes'] + num_shift
        else:
            self._fields['pred_classes'] = self._fields['pred_classes'] + num_shift
            self.classes_filter()
    
    def classes_relu(self, fg_indx, bg_indx, total_class_number):

        mask = (self._fields['gt_classes'] >= fg_indx) * (self._fields['gt_classes'] < total_class_number)
        self._fields['gt_classes'][mask] = fg_indx
        self._fields['gt_classes'][self._fields['gt_classes'] == total_class_number] = bg_indx
    
    def classes_filter(self):
        mask = self._fields['pred_classes'] > 0
        self.pred_classes = self.pred_classes[mask]
        self.pred_boxes = self.pred_boxes[mask]
        self.pred_masks = self.pred_masks[mask]

        
    def frequency_split(self):
        """
        Returns:
            list: [Instance,Instance,Instance] according to f, c, r frequency
        """
        freq_Inst = Instances(self._image_size)
        comm_Inst = Instances(self._image_size)
        rare_Inst = Instances(self._image_size)
        
        num_instances = len(self)
        img_height = self._image_size[0]
        img_width = self._image_size[1]
        
        freqencys = self._fields['gt_frequencys']
        freq_mask = freqencys == 0
        comm_mask = freqencys == 1
        rare_mask = freqencys == 2
        
        boxes = self._fields['gt_boxes']
        classes = self._fields['gt_classes']
        if self.has('gt_masks'):
            masks = self._fields['gt_masks']
        
        freq_Inst.gt_boxes = boxes[freq_mask]
        freq_Inst.gt_classes = classes[freq_mask]
        freq_Inst.gt_frequencys = freqencys[freq_mask]
        if self.has('gt_masks'):
            freq_Inst.gt_masks = masks[freq_mask]
        
        comm_Inst.gt_boxes = boxes[comm_mask]
        comm_Inst.gt_classes = classes[comm_mask]
        comm_Inst.gt_frequencys = freqencys[comm_mask]
        if self.has('gt_masks'):
            comm_Inst.gt_masks = masks[comm_mask]
        
        rare_Inst.gt_boxes = boxes[rare_mask]
        rare_Inst.gt_classes = classes[rare_mask]
        rare_Inst.gt_frequencys = freqencys[rare_mask]
        if self.has('gt_masks'):
            rare_Inst.gt_masks = masks[rare_mask]

        return [freq_Inst, comm_Inst, rare_Inst]
   