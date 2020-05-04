// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>

#include <torch/csrc/autograd/grad_mode.h>
#include <torch/script.h>

using namespace std;

// experimental. don't use
int main(int argc, const char* argv[]) {
  if (argc != 3) {
    return 1;
  }
  std::string image_file = argv[2];

  torch::autograd::AutoGradMode guard(false);
  auto module = torch::jit::load(argv[1]);

  assert(module.buffers().size() > 0);
  // Assume that the entire model is on the same device.
  // We just put input to this device.
  auto device = (*begin(module.buffers())).device();

  cv::Mat input_img = cv::imread(image_file, cv::IMREAD_COLOR);
  const int height = input_img.rows;
  const int width = input_img.cols;
  // FPN models require divisibility of 32
  assert(height % 32 == 0 && width % 32 == 0);
  const int channels = 3;

  auto input = torch::from_blob(
      input_img.data, {1, height, width, channels}, torch::kUInt8);
  // NHWC to NCHW
  input = input.to(device, torch::kFloat).permute({0, 3, 1, 2}).contiguous();

  std::array<float, 3> im_info_data{height * 1.0f, width * 1.0f, 1.0f};
  auto im_info = torch::from_blob(im_info_data.data(), {1, 3}).to(device);

  // run the network
  auto output = module.forward({std::make_tuple(input, im_info)});

  // run 3 more times to benchmark
  int N_benchmark = 3;
  auto start_time = chrono::high_resolution_clock::now();
  for (int i = 0; i < N_benchmark; ++i) {
    output = module.forward({std::make_tuple(input, im_info)});
  }
  auto end_time = chrono::high_resolution_clock::now();
  auto ms = chrono::duration_cast<chrono::microseconds>(end_time - start_time)
                .count();
  cout << "Latency (should vary with different inputs): "
       << ms * 1.0 / 1e6 / N_benchmark << " seconds" << endl;

  auto outputs = output.toTuple()->elements();
  // parse Mask R-CNN outputs
  auto bbox = outputs[0].toTensor(), scores = outputs[1].toTensor(),
       labels = outputs[2].toTensor(), mask_probs = outputs[3].toTensor();

  cout << "bbox: " << bbox.toString() << " " << bbox.sizes() << endl;
  cout << "scores: " << scores.toString() << " " << scores.sizes() << endl;
  cout << "labels: " << labels.toString() << " " << labels.sizes() << endl;
  cout << "mask_probs: " << mask_probs.toString() << " " << mask_probs.sizes()
       << endl;

  int num_instances = bbox.sizes()[0];
  cout << bbox << endl;
  return 0;
}
