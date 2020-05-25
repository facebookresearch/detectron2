# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# See https://pytorch.org/tutorials/advanced/cpp_frontend.html
cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(caffe2_mask_rcnn)

find_package(Torch REQUIRED)
find_package(gflags REQUIRED)
find_package(OpenCV REQUIRED)

add_executable(caffe2_mask_rcnn caffe2_mask_rcnn.cpp)
target_link_libraries(
  caffe2_mask_rcnn
  "${TORCH_LIBRARIES}" gflags glog ${OpenCV_LIBS})
set_property(TARGET caffe2_mask_rcnn PROPERTY CXX_STANDARD 14)


add_executable(torchscript_traced_mask_rcnn torchscript_traced_mask_rcnn.cpp)
target_link_libraries(
  torchscript_traced_mask_rcnn
  "${TORCH_LIBRARIES}" ${OpenCV_LIBS})
set_property(TARGET torchscript_traced_mask_rcnn PROPERTY CXX_STANDARD 14)
