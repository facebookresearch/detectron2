// Copyright (c) Facebook, Inc. and its affiliates.

#include <c10/util/Flags.h>
#include <caffe2/core/blob.h>
#include <caffe2/core/common.h>
#include <caffe2/core/init.h>
#include <caffe2/core/net.h>
#include <caffe2/core/workspace.h>
#include <caffe2/utils/proto_utils.h>

#include <opencv2/opencv.hpp>
#include <cassert>
#include <chrono>
#include <iostream>
#include <string>

C10_DEFINE_string(predict_net, "", "path to model.pb");
C10_DEFINE_string(init_net, "", "path to model_init.pb");
C10_DEFINE_string(input, "", "path to input image");

using namespace std;
using namespace caffe2;

int main(int argc, char** argv) {
  caffe2::GlobalInit(&argc, &argv);
  string predictNetPath = FLAGS_predict_net;
  string initNetPath = FLAGS_init_net;
  cv::Mat input = cv::imread(FLAGS_input, cv::IMREAD_COLOR);

  const int height = input.rows;
  const int width = input.cols;
  // FPN models require divisibility of 32
  assert(height % 32 == 0 && width % 32 == 0);
  const int batch = 1;
  const int channels = 3;

  // initialize Net and Workspace
  caffe2::NetDef initNet_, predictNet_;
  CAFFE_ENFORCE(ReadProtoFromFile(initNetPath, &initNet_));
  CAFFE_ENFORCE(ReadProtoFromFile(predictNetPath, &predictNet_));

  Workspace workSpace;
  for (auto& str : predictNet_.external_input()) {
    workSpace.CreateBlob(str);
  }
  CAFFE_ENFORCE(workSpace.CreateNet(predictNet_));
  CAFFE_ENFORCE(workSpace.RunNetOnce(initNet_));

  // setup inputs
  auto data = BlobGetMutableTensor(workSpace.GetBlob("data"), caffe2::CPU);
  data->Resize(batch, channels, height, width);
  float* ptr = data->mutable_data<float>();
  // HWC to CHW
  for (int c = 0; c < 3; ++c) {
    for (int i = 0; i < height * width; ++i) {
      ptr[c * height * width + i] = static_cast<float>(input.data[3 * i + c]);
    }
  }

  auto im_info =
      BlobGetMutableTensor(workSpace.GetBlob("im_info"), caffe2::CPU);
  im_info->Resize(batch, 3);
  float* im_info_ptr = im_info->mutable_data<float>();
  im_info_ptr[0] = height;
  im_info_ptr[1] = width;
  im_info_ptr[2] = 1.0;

  // run the network
  CAFFE_ENFORCE(workSpace.RunNet(predictNet_.name()));

  // run 3 more times to benchmark
  int N_benchmark = 3;
  auto start_time = chrono::high_resolution_clock::now();
  for (int i = 0; i < N_benchmark; ++i) {
    CAFFE_ENFORCE(workSpace.RunNet(predictNet_.name()));
  }
  auto end_time = chrono::high_resolution_clock::now();
  auto ms = chrono::duration_cast<chrono::microseconds>(end_time - start_time)
                .count();
  cout << "Latency (should vary with different inputs): "
       << ms * 1.0 / 1e6 / N_benchmark << " seconds" << endl;

  // parse Mask R-CNN outputs
  caffe2::Tensor bbox(
      workSpace.GetBlob("bbox_nms")->Get<caffe2::Tensor>(), caffe2::CPU);
  caffe2::Tensor scores(
      workSpace.GetBlob("score_nms")->Get<caffe2::Tensor>(), caffe2::CPU);
  caffe2::Tensor labels(
      workSpace.GetBlob("class_nms")->Get<caffe2::Tensor>(), caffe2::CPU);
  caffe2::Tensor mask_probs(
      workSpace.GetBlob("mask_fcn_probs")->Get<caffe2::Tensor>(), caffe2::CPU);
  cout << "bbox:" << bbox.DebugString() << endl;
  cout << "scores:" << scores.DebugString() << endl;
  cout << "labels:" << labels.DebugString() << endl;
  cout << "mask_probs: " << mask_probs.DebugString() << endl;

  int num_instances = bbox.sizes()[0];
  for (int i = 0; i < num_instances; ++i) {
    float score = scores.data<float>()[i];
    if (score < 0.6)
      continue; // skip them

    const float* box = bbox.data<float>() + i * 4;
    int label = labels.data<float>()[i];

    cout << "Prediction " << i << ", xyxy=(";
    cout << box[0] << ", " << box[1] << ", " << box[2] << ", " << box[3]
         << "); score=" << score << "; label=" << label << endl;

    const float* mask = mask_probs.data<float>() +
        i * mask_probs.size_from_dim(1) + label * mask_probs.size_from_dim(2);

    // save the 28x28 mask
    cv::Mat cv_mask(28, 28, CV_32FC1);
    memcpy(cv_mask.data, mask, 28 * 28 * sizeof(float));
    cv::imwrite("mask" + std::to_string(i) + ".png", cv_mask * 255.);
  }
  return 0;
}
