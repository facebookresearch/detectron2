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
from caffe2.proto import caffe2_pb2
from caffe2.python import core


def get_device_option_cpu():
    device_option = core.DeviceOption(caffe2_pb2.CPU)
    return device_option


def get_device_option_cuda(gpu_id=0):
    device_option = caffe2_pb2.DeviceOption()
    device_option.device_type = caffe2_pb2.CUDA
    device_option.device_id = gpu_id
    return device_option
