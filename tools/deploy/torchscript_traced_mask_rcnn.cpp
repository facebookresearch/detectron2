// Copyright (c) Facebook, Inc. and its affiliates.
// @lint-ignore-every CLANGTIDY

#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>

#include <c10/cuda/CUDAStream.h>
#include <torch/csrc/autograd/grad_mode.h>
#include <torch/csrc/jit/runtime/graph_executor.h>
#include <torch/script.h>

// only needed for export_method=tracing
#include <torchvision/vision.h> // @oss-only
// @fb-only: #include <torchvision/csrc/vision.h>

using namespace std;

c10::IValue get_caffe2_tracing_inputs(cv::Mat& img, c10::Device device) {
  const int height = img.rows;
  const int width = img.cols;
  // FPN models require divisibility of 32.
  // Tracing mode does padding inside the graph, but caffe2_tracing does not.
  assert(height % 32 == 0 && width % 32 == 0);
  const int channels = 3;

  auto input =
      torch::from_blob(img.data, {1, height, width, channels}, torch::kUInt8);
  // NHWC to NCHW
  input = input.to(device, torch::kFloat).permute({0, 3, 1, 2}).contiguous();

  std::array<float, 3> im_info_data{height * 1.0f, width * 1.0f, 1.0f};
  auto im_info =
      torch::from_blob(im_info_data.data(), {1, 3}).clone().to(device);
  return std::make_tuple(input, im_info);
}

c10::IValue get_tracing_inputs(cv::Mat& img, c10::Device device) {
  const int height = img.rows;
  const int width = img.cols;
  const int channels = 3;

  auto input =
      torch::from_blob(img.data, {height, width, channels}, torch::kUInt8);
  // HWC to CHW
  input = input.to(device, torch::kFloat).permute({2, 0, 1}).contiguous();
  return input;
}

int main(int argc, const char* argv[]) {
  if (argc != 4) {
    cerr << R"xx(
Usage:
   ./torchscript_traced_mask_rcnn model.ts input.jpg EXPORT_METHOD

   EXPORT_METHOD can be "tracing" or "caffe2_tracing".
)xx";
    return 1;
  }
  std::string image_file = argv[2];
  std::string export_method = argv[3];
  assert(export_method == "caffe2_tracing" || export_method == "tracing");
  bool is_caffe2 = export_method == "caffe2_tracing";

  torch::jit::getBailoutDepth() = 1;
  torch::autograd::AutoGradMode guard(false);
  auto module = torch::jit::load(argv[1]);

  assert(module.buffers().size() > 0);
  // Assume that the entire model is on the same device.
  // We just put input to this device.
  auto device = (*begin(module.buffers())).device();

  cv::Mat input_img = cv::imread(image_file, cv::IMREAD_COLOR);
  auto inputs = is_caffe2 ? get_caffe2_tracing_inputs(input_img, device)
                          : get_tracing_inputs(input_img, device);

  // run the network
  auto output = module.forward({inputs});
  if (device.is_cuda())
    c10::cuda::getCurrentCUDAStream().synchronize();

  // run 3 more times to benchmark
  int N_benchmark = 3, N_warmup = 1;
  auto start_time = chrono::high_resolution_clock::now();
  for (int i = 0; i < N_benchmark + N_warmup; ++i) {
    if (i == N_warmup)
      start_time = chrono::high_resolution_clock::now();
    output = module.forward({inputs});
    if (device.is_cuda())
      c10::cuda::getCurrentCUDAStream().synchronize();
  }
  auto end_time = chrono::high_resolution_clock::now();
  auto ms = chrono::duration_cast<chrono::microseconds>(end_time - start_time)
                .count();
  cout << "Latency (should vary with different inputs): "
       << ms * 1.0 / 1e6 / N_benchmark << " seconds" << endl;

  auto outputs = output.toTuple()->elements();
  cout << "Number of output tensors: " << outputs.size() << endl;
  at::Tensor bbox, pred_classes, pred_masks, scores;
  // parse Mask R-CNN outputs
  if (is_caffe2) {
    bbox = outputs[0].toTensor(), scores = outputs[1].toTensor(),
    pred_classes = outputs[2].toTensor(), pred_masks = outputs[3].toTensor();
  } else {
    bbox = outputs[0].toTensor(), pred_classes = outputs[1].toTensor(),
    pred_masks = outputs[2].toTensor(), scores = outputs[3].toTensor();
    // outputs[-1] is image_size, others fields ordered by their field name in
    // Instances
  }

  cout << "bbox: " << bbox.toString() << " " << bbox.sizes() << endl;
  cout << "scores: " << scores.toString() << " " << scores.sizes() << endl;
  cout << "pred_classes: " << pred_classes.toString() << " "
       << pred_classes.sizes() << endl;
  cout << "pred_masks: " << pred_masks.toString() << " " << pred_masks.sizes()
       << endl;

  int num_instances = bbox.sizes()[0];
  cout << bbox << endl;
  return 0;
}
