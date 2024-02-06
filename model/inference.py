import numpy as np
import onnxruntime as rt
import onnx
import cv2

from effizency.utils.general_utils import draw_boxes_masks

# Load the ONNX model
model = onnx.load("eval_model.onnx")
sess = rt.InferenceSession("eval_model.onnx")

# Prepare input data (ensure compatible format and shape)
input_data = cv2.imread('../data/roof_1.png')
input_data = cv2.cvtColor(input_data, cv2.COLOR_BGR2RGB)
input_data = input_data / 1.0  # Scale to [0, 1]
pixel_mean = (103.53, 116.28, 123.675)
std = (1.0, 1.0, 1.0)
input_data -= pixel_mean  # Subtract mean
input_data /= std
input_data = cv2.resize(input_data, dsize=(800, 800), interpolation=cv2.INTER_LINEAR)
input_data = np.expand_dims(np.transpose(input_data, (2, 0, 1)), 0)

# Run inference
input_name = sess.get_inputs()[0].name  # Get input name from model
output_names = [output.name for output in sess.get_outputs()]  # Get output name from model

result = sess.run(output_names, {input_name: input_data.astype(np.float32)})

