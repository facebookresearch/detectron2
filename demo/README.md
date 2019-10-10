## Webcam and Jupyter notebook demo

This folder contains a simple webcam demo that illustrates how you can use `detectron2` for inference.

You can start it by running it from this folder, using one of the following commands:
```bash
# by default, it runs on the GPU
# for best results, use min-image-size 800
python webcam.py --min-image-size 800
# can also run it on the CPU
python webcam.py --min-image-size 300 MODEL.DEVICE cpu
# or change the model that you want to use
python webcam.py --config-file ../configs/quick_schedules/mask_rcnn_R_50_FPN_inference_acc_test.yaml --min-image-size 300 MODEL.DEVICE cpu
# in order to see the probability heatmaps, pass --show-mask-heatmaps
python webcam.py --min-image-size 300 --show-mask-heatmaps MODEL.DEVICE cpu
```
