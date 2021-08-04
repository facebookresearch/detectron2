# Getting Started with DensePose

## Inference with Pre-trained Models

1. Pick a model and its config file from [Model Zoo](MODEL_ZOO.md), for example [densepose_rcnn_R_50_FPN_s1x.yaml](../configs/densepose_rcnn_R_50_FPN_s1x.yaml)
2. Run the [Apply Net](TOOL_APPLY_NET.md) tool to visualize the results or save the to disk. For example, to use contour visualization for DensePose, one can run:
```bash
python apply_net.py show configs/densepose_rcnn_R_50_FPN_s1x.yaml densepose_rcnn_R_50_FPN_s1x.pkl image.jpg dp_contour,bbox --output image_densepose_contour.png
```
Please see [Apply Net](TOOL_APPLY_NET.md) for more details on the tool.

## Training

First, prepare the [dataset](http://densepose.org/#dataset) into the following structure under the directory you'll run training scripts:
<pre>
datasets/coco/
  annotations/
    densepose_{train,minival,valminusminival}2014.json
    <a href="https://dl.fbaipublicfiles.com/detectron2/densepose/densepose_minival2014_100.json">densepose_minival2014_100.json </a>  (optional, for testing only)
  {train,val}2014/
    # image files that are mentioned in the corresponding json
</pre>

To train a model one can use the [train_net.py](../train_net.py) script.
This script was used to train all DensePose models in [Model Zoo](MODEL_ZOO.md).
For example, to launch end-to-end DensePose-RCNN training with ResNet-50 FPN backbone
on 8 GPUs following the s1x schedule, one can run
```bash
python train_net.py --config-file configs/densepose_rcnn_R_50_FPN_s1x.yaml --num-gpus 8
```
The configs are made for 8-GPU training. To train on 1 GPU, one can apply the
[linear learning rate scaling rule](https://arxiv.org/abs/1706.02677):
```bash
python train_net.py --config-file configs/densepose_rcnn_R_50_FPN_s1x.yaml \
    SOLVER.IMS_PER_BATCH 2 SOLVER.BASE_LR 0.0025
```

## Evaluation

Model testing can be done in the same way as training, except for an additional flag `--eval-only` and
model location specification through `MODEL.WEIGHTS model.pth` in the command line
```bash
python train_net.py --config-file configs/densepose_rcnn_R_50_FPN_s1x.yaml \
    --eval-only MODEL.WEIGHTS model.pth
```

## Tools

We provide tools which allow one to:
 - easily view DensePose annotated data in a dataset;
 - perform DensePose inference on a set of images;
 - visualize DensePose model results;

`query_db` is a tool to print or visualize DensePose data in a dataset.
Please refer to [Query DB](TOOL_QUERY_DB.md) for more details on this tool

`apply_net` is a tool to print or visualize DensePose results.
Please refer to [Apply Net](TOOL_APPLY_NET.md) for more details on this tool


## Installation as a package

DensePose can also be installed as a Python package for integration with other software.

The following dependencies are needed:
- Python >= 3.6
- [PyTorch](https://pytorch.org/get-started/locally/#start-locally)

DensePose can then be installed from this repository with:

```
pip install git+https://github.com/facebookresearch/detectron2@master#subdirectory=projects/DensePose
```

After installation, the package will be importable as `densepose`.
