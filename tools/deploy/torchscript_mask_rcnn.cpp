// Copyright (c) Facebook, Inc. and its affiliates.
// @lint-ignore-every CLANGTIDY
// This is an example code that demonstrates how to run inference
// with a torchscript format Mask R-CNN model exported by ./export_model.py
// using export method=tracing, caffe2_tracing & scripting.

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

// create a Tuple[Dict[str, Tensor]] which is the input type of scripted model
c10::IValue get_scripting_inputs(cv::Mat& img, c10::Device device) {
  const int height = img.rows;
  const int width = img.cols;
  const int channels = 3;

  auto img_tensor =
      torch::from_blob(img.data, {height, width, channels}, torch::kUInt8);
  // HWC to CHW
  img_tensor =
      img_tensor.to(device, torch::kFloat).permute({2, 0, 1}).contiguous();
  auto dic = c10::Dict<std::string, torch::Tensor>();
  dic.insert("image", img_tensor);
  return std::make_tuple(dic);
}

c10::IValue
get_inputs(std::string export_method, cv::Mat& img, c10::Device device) {
  // Given an image, create inputs in the format required by the model.
  if (export_method == "tracing")
    return get_tracing_inputs(img, device);
  if (export_method == "caffe2_tracing")
    return get_caffe2_tracing_inputs(img, device);
  if (export_method == "scripting")
    return get_scripting_inputs(img, device);
  abort();
}

struct MaskRCNNOutputs {
  at::Tensor pred_boxes, pred_classes, pred_masks, scores;
  int num_instances() const {
    return pred_boxes.sizes()[0];
  }
};

MaskRCNNOutputs get_outputs(std::string export_method, c10::IValue outputs) {
  // Given outputs of the model, extract tensors from it to turn into a
  // common MaskRCNNOutputs format.
  if (export_method == "tracing") {
    auto out_tuple = outputs.toTuple()->elements();
    // They are ordered alphabetically by their field name in Instances
    return MaskRCNNOutputs{
        out_tuple[0].toTensor(),
        out_tuple[1].toTensor(),
        out_tuple[2].toTensor(),
        out_tuple[3].toTensor()};
  }
  if (export_method == "caffe2_tracing") {
    auto out_tuple = outputs.toTuple()->elements();
    // A legacy order used by caffe2 models
    return MaskRCNNOutputs{
        out_tuple[0].toTensor(),
        out_tuple[2].toTensor(),
        out_tuple[3].toTensor(),
        out_tuple[1].toTensor()};
  }
  if (export_method == "scripting") {
    // With the ScriptableAdapter defined in export_model.py, the output is
    // List[Dict[str, Any]].
    auto out_dict = outputs.toList().get(0).toGenericDict();
    return MaskRCNNOutputs{
        out_dict.at("pred_boxes").toTensor(),
        out_dict.at("pred_classes").toTensor(),
        out_dict.at("pred_masks").toTensor(),
        out_dict.at("scores").toTensor()};
  }
  abort();
}

int main(int argc, const char* argv[]) {
  if (argc != 4) {
    cerr << R"xx(
Usage:
   ./torchscript_mask_rcnn model.ts input.jpg EXPORT_METHOD

   EXPORT_METHOD can be "tracing", "caffe2_tracing" or "scripting".
)xx";
    return 1;
  }
  std::string image_file = argv[2];
  std::string export_method = argv[3];
  assert(
      export_method == "caffe2_tracing" || export_method == "tracing" ||
      export_method == "scripting");

  torch::jit::getBailoutDepth() = 1;
  torch::autograd::AutoGradMode guard(false);
  auto module = torch::jit::load(argv[1]);

  assert(module.buffers().size() > 0);
  // Assume that the entire model is on the same device.
  // We just put input to this device.
  auto device = (*begin(module.buffers())).device();

  cv::Mat input_img = cv::imread(image_file, cv::IMREAD_COLOR);
  auto inputs = get_inputs(export_method, input_img, device);

  // Run the network
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

  // Parse Mask R-CNN outputs
  auto rcnn_outputs = get_outputs(export_method, output);
  cout << "Number of detected objects: " << rcnn_outputs.num_instances()
       << endl;

  cout << "pred_boxes: " << rcnn_outputs.pred_boxes.toString() << " "
       << rcnn_outputs.pred_boxes.sizes() << endl;
  cout << "scores: " << rcnn_outputs.scores.toString() << " "
       << rcnn_outputs.scores.sizes() << endl;
  cout << "pred_classes: " << rcnn_outputs.pred_classes.toString() << " "
       << rcnn_outputs.pred_classes.sizes() << endl;
  cout << "pred_masks: " << rcnn_outputs.pred_masks.toString() << " "
       << rcnn_outputs.pred_masks.sizes() << endl;

  cout << rcnn_outputs.pred_boxes << endl;
  return 0;
}
