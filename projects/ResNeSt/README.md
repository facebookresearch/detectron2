[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/resnest-split-attention-networks/instance-segmentation-on-coco)](https://paperswithcode.com/sota/instance-segmentation-on-coco?p=resnest-split-attention-networks)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/resnest-split-attention-networks/object-detection-on-coco)](https://paperswithcode.com/sota/object-detection-on-coco?p=resnest-split-attention-networks)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/resnest-split-attention-networks/panoptic-segmentation-on-coco-panoptic)](https://paperswithcode.com/sota/panoptic-segmentation-on-coco-panoptic?p=resnest-split-attention-networks)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/resnest-split-attention-networks/instance-segmentation-on-coco-minival)](https://paperswithcode.com/sota/instance-segmentation-on-coco-minival?p=resnest-split-attention-networks)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/resnest-split-attention-networks/object-detection-on-coco-minival)](https://paperswithcode.com/sota/object-detection-on-coco-minival?p=resnest-split-attention-networks)

# ResNeSt (Detectron2 Wrapper)

Code for detection and instance segmentation experiments in [ResNeSt](https://hangzhang.org/files/resnest.pdf).


## Training and Inference
Please follow [INSTALL.md](https://github.com/facebookresearch/detectron2/blob/master/INSTALL.md) to install detectron2. 

Then please run the following command to install this project
```shell
python setup.py install
```

To train a model with 8 gpus, please run
```shell
python train_net.py  --num-gpus 8 --config-file your_config.yaml
```

For inference
```shell
python train_net.py  \
    --config-file your_config.yaml
    --eval-only MODEL.WEIGHTS /path/to/checkpoint_file
```

For the inference demo, please see [GETTING_STARTED.md](https://github.com/facebookresearch/detectron2/blob/master/GETTING_STARTED.md).

## Pretrained Models

### Object Detection
<table class="tg">
  <tr>
    <th class="tg-0pky">Method</th>
    <th class="tg-0pky">Backbone</th>
    <th class="tg-0pky">mAP%</th>
    <th class="tg-0pky">download</th>
  </tr>
  <tr>
    <td rowspan="5" class="tg-0pky">Faster R-CNN</td>
    <td class="tg-0pky">ResNet-50</td>
    <td class="tg-0pky">39.25</td>
    <td class="tg-0lax"><a href="./configs/COCO-Detection/faster_rcnn_R_50_FPN_syncbn_range-scale_1x.yaml">config</a> | <a href="https://s3.us-west-1.wasabisys.com/resnest/detectron/faster_rcnn_R_50_FPN_syncbn_range-scale_1x-fde56e2b.pth ">model</a> | <a href="https://s3.us-west-1.wasabisys.com/resnest/detectron/faster_rcnn_R_50_FPN_syncbn_range-scale_1x.txt">log</a> </td>
  </tr>
  <tr>
    <td class="tg-0lax">ResNet-101</td>
    <td class="tg-0lax">41.37</td>
     <td class="tg-0lax"><a href="./configs/COCO-Detection/faster_rcnn_R_101_FPN_syncbn_range-scale_1x.yaml">config</a> | <a href="https://s3.us-west-1.wasabisys.com/resnest/detectron/faster_rcnn_R_101_FPN_syncbn_range-scale_1x-57c73356.pth">model</a> | <a href="https://s3.us-west-1.wasabisys.com/resnest/detectron/faster_rcnn_R_101_FPN_syncbn_range-scale_1x.txt">log</a> </td>
  </tr>
  <tr>
    <td class="tg-0lax">ResNeSt-50 (<span style="color:red">ours</span>)</td>
    <td class="tg-0lax"><b>42.33</b></td>
     <td class="tg-0lax"><a href="./configs/COCO-Detection/faster_rcnn_ResNeSt_50_FPN_syncbn_range-scale_1x.yaml">config</a> | <a href="https://s3.us-west-1.wasabisys.com/resnest/detectron/faster_rcnn_ResNeSt_50_FPN_syncbn_range-scale_1x-ad123c0b.pth">model</a> | <a href="https://s3.us-west-1.wasabisys.com/resnest/detectron/faster_rcnn_ResNeSt_50_FPN_syncbn_range-scale_1x.txt">log</a> </td>
  </tr>
  <tr>
    <td class="tg-0lax">ResNeSt-50-DCNv2 (<span style="color:red">ours</span>)</td>
    <td class="tg-0lax"><b>44.11</b></td>
     <td class="tg-0lax"><a href="./configs/COCO-Detection/faster_rcnn_ResNeSt_50_FPN_dcn_syncbn_range-scale_1x.yaml">config</a> | <a href="https://s3.us-west-1.wasabisys.com/resnest/detectron/faster_rcnn_ResNeSt_50_FPN_dcn_syncbn_range-scale_1x.pth">model</a> | <a href="https://s3.us-west-1.wasabisys.com/resnest/detectron/faster_rcnn_ResNeSt_50_FPN_dcn_syncbn_range-scale_1x.txt">log</a> </td>
  </tr> 
  <tr>
    <td class="tg-0lax">ResNeSt-101 (<span style="color:red">ours</span>)</td>
    <td class="tg-0lax"><b>44.72</b></td>
    <td class="tg-0lax"><a href="./configs/COCO-Detection/faster_rcnn_ResNeSt_101_FPN_syncbn_range-scale_1x.yaml">config</a> | <a href="https://s3.us-west-1.wasabisys.com/resnest/detectron/faster_rcnn_ResNeSt_101_FPN_syncbn_range-scale_1x-d8f284b6.pth">model</a> | <a href="https://s3.us-west-1.wasabisys.com/resnest/detectron/faster_rcnn_ResNeSt_101_FPN_syncbn_range-scale_1x.txt">log</a> </td>
  </tr>
  <tr>
    <td rowspan="5" class="tg-0lax">Cascade R-CNN</td>
    <td class="tg-0lax">ResNet-50</td>
    <td class="tg-0lax">42.52</td>
    <td class="tg-0lax"><a href="./configs/COCO-Detection/faster_cascade_rcnn_R_50_FPN_syncbn_range-scale_1x.yaml">config</a> | <a href="https://s3.us-west-1.wasabisys.com/resnest/detectron/faster_cascade_rcnn_R_50_FPN_syncbn_range-scale_1x-3c7f2ef2.pth">model</a> | <a href="https://s3.us-west-1.wasabisys.com/resnest/detectron/faster_cascade_rcnn_R_50_FPN_syncbn_range-scale_1x.txt">log</a> </td>
  </tr>
  <tr>
    <td class="tg-0lax">ResNet-101</td>
    <td class="tg-0lax">44.03</td>
    <td class="tg-0lax"><a href="./configs/COCO-Detection/faster_cascade_rcnn_R_101_FPN_syncbn_range-scale_1x.yaml">config</a> | <a href="https://s3.us-west-1.wasabisys.com/resnest/detectron/faster_cascade_rcnn_R_101_FPN_syncbn_range-scale_1x-4073359b.pth">model</a> | <a href="https://s3.us-west-1.wasabisys.com/resnest/detectron/faster_cascade_rcnn_R_101_FPN_syncbn_range-scale_1x.txt">log</a> </td>
  </tr>
  <tr>
    <td class="tg-0lax">ResNeSt-50 (<span style="color:red">ours</span>)</td>
    <td class="tg-0lax"><b>45.41</b></td>
    <td class="tg-0lax"><a href="./configs/COCO-Detection/faster_cascade_rcnn_ResNeSt_50_FPN_syncbn_range-scale-1x.yaml">config</a> | <a href="https://s3.us-west-1.wasabisys.com/resnest/detectron/faster_cascade_rcnn_ResNeSt_50_FPN_syncbn_range-scale-1x-e9955232.pth">model</a> | <a href="https://s3.us-west-1.wasabisys.com/resnest/detectron/faster_cascade_rcnn_ResNeSt_50_FPN_syncbn_range-scale-1x.txt">log</a> </td>
  </tr>
  <tr>
    <td class="tg-0lax">ResNeSt-101 (<span style="color:red">ours</span>)</td>
    <td class="tg-0lax"><b>47.50</b></td>
    <td class="tg-0lax"><a href="./configs/COCO-Detection/faster_cascade_rcnn_ResNeSt_101_FPN_syncbn_range-scale_1x.yaml">config</a> | <a href="https://s3.us-west-1.wasabisys.com/resnest/detectron/faster_cascade_rcnn_ResNeSt_101_FPN_syncbn_range-scale_1x-3627ef78.pth">model</a> | <a href="https://s3.us-west-1.wasabisys.com/resnest/detectron/faster_cascade_rcnn_ResNeSt_101_FPN_syncbn_range-scale_1x.txt">log</a> </td>
  </tr>
  <tr>
    <td class="tg-0lax">ResNeSt-200 (<span style="color:red">ours</span>)</td>
    <td class="tg-0lax"><b>49.03</b></td>
    <td class="tg-0lax"><a href="./configs/COCO-Detection/faster_cascade_rcnn_ResNeSt_200_FPN_syncbn_range-scale_1x.yaml">config</a> | <a href="https://s3.us-west-1.wasabisys.com/resnest/detectron/faster_cascade_rcnn_ResNeSt_200_FPN_syncbn_range-scale_1x-1be2a87e.pth">model</a> | <a href="https://s3.us-west-1.wasabisys.com/resnest/detectron/faster_cascade_rcnn_ResNeSt_200_FPN_syncbn_range-scale_1x.txt">log</a> </td>
  </tr>
</table>

We train all models with FPN, SyncBN and image scale augmentation (short size of a image is pickedrandomly from 640 to 800). 1x learning rate schedule is used. All of them are reported on COCO-2017 validation dataset.



### Instance Segmentation
<table class="tg">
  <tr>
    <th class="tg-0pky">Method</th>
    <th class="tg-0pky">Backbone</th>
    <th class="tg-0pky">bbox</th>
    <th class="tg-0lax">mask</th>
    <th class="tg-0pky">download</th>
  </tr>
  <tr>
    <td rowspan="4" class="tg-0pky">Mask R-CNN</td>
    <td class="tg-0pky">ResNet-50</td>
    <td class="tg-0pky">39.97</td>
    <td class="tg-0lax">36.05</td>
    <td class="tg-0lax"><a href="./configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_syncbn_1x.yaml">config</a> | <a href="https://s3.us-west-1.wasabisys.com/resnest/detectron/mask_rcnn_R_50_FPN_syncbn_1x-4939bd58.pth">model</a> | <a href="https://s3.us-west-1.wasabisys.com/resnest/detectron/mask_rcnn_R_50_FPN_syncbn_1x.txt">log</a> </td>
</tr>
  <tr>
    <td class="tg-0lax">ResNet-101</td>
    <td class="tg-0lax">41.78</td>
    <td class="tg-0lax">37.51</td>
    <td class="tg-0lax"><a href="./configs/COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_syncbn_1x.yaml">config</a> | <a href="https://s3.us-west-1.wasabisys.com/resnest/detectron/mask_rcnn_R_101_FPN_syncbn_1x-55493cc2.pth">model</a> | <a href="https://s3.us-west-1.wasabisys.com/resnest/detectron/mask_rcnn_R_101_FPN_syncbn_1x.txt">log</a> </td>
  </tr>
  <tr>
    <td class="tg-0lax">ResNeSt-50 (<span style="color:red">ours</span>)</td>
    <td class="tg-0lax"><b>42.81</b></td>
    <td class="tg-0lax"><b>38.14</td>
    <td class="tg-0lax"><a href="./configs/COCO-InstanceSegmentation/mask_rcnn_ResNeSt_50_FPN_syncBN_1x.yaml">config</a> | <a href="https://s3.us-west-1.wasabisys.com/resnest/detectron/mask_rcnn_ResNeSt_50_FPN_syncBN_1x-f442d863.pth">model</a> | <a href="https://s3.us-west-1.wasabisys.com/resnest/detectron/mask_rcnn_ResNeSt_50_FPN_syncBN_1x.txt">log</a> </td>
  </tr>
  <tr>
    <td class="tg-0lax">ResNeSt-101 (<span style="color:red">ours</span>)</td>
    <td class="tg-0lax"><b>45.75</b></td>
    <td class="tg-0lax"><b>40.65</b></td>
     <td class="tg-0lax"><a href="./configs/COCO-InstanceSegmentation/mask_rcnn_ResNeSt_101_FPN_syncBN_1x.yaml">config</a> | <a href="https://s3.us-west-1.wasabisys.com/resnest/detectron/mask_rcnn_ResNeSt_101_FPN_syncBN_1x-528502c6.pth">model</a> | <a href="https://s3.us-west-1.wasabisys.com/resnest/detectron/mask_rcnn_ResNeSt_101_FPN_syncBN_1x.txt">log</a> </td>   
  </tr>
  <tr>
    <td rowspan="7" class="tg-0lax">Cascade R-CNN</td>
    <td class="tg-0lax">ResNet-50</td>
    <td class="tg-0lax">43.06</td>
    <td class="tg-0lax">37.19</td>
    <td class="tg-0lax"><a href="./configs/COCO-InstanceSegmentation/mask_cascade_rcnn_R_50_FPN_syncbn_1x.yaml">config</a> | <a href="https://s3.us-west-1.wasabisys.com/resnest/detectron/mask_cascade_rcnn_R_50_FPN_syncbn_1x-03310c9b.pth">model</a> | <a href="https://s3.us-west-1.wasabisys.com/resnest/detectron/mask_cascade_rcnn_R_50_FPN_syncbn_1x.txt">log</a> </td>
  </tr>
  <tr>
    <td class="tg-0lax">ResNet-101</td>
    <td class="tg-0lax">44.79</td>
    <td class="tg-0lax">38.52</td>
    <td class="tg-0lax"><a href="./configs/COCO-InstanceSegmentation/mask_cascade_rcnn_R_101_FPN_syncbn_1x.yaml">config</a> | <a href="https://s3.us-west-1.wasabisys.com/resnest/detectron/mask_cascade_rcnn_R_101_FPN_syncbn_1x-8cec1631.pth">model</a> | <a href="https://s3.us-west-1.wasabisys.com/resnest/detectron/mask_cascade_rcnn_R_101_FPN_syncbn_1x.txt">log</a> </td>
  </tr>
  <tr>
    <td class="tg-0lax">ResNeSt-50 (<span style="color:red">ours</span>)</td>
    <td class="tg-0lax"><b>46.19</b></td>
    <td class="tg-0lax"><b>39.55</b></td>
    <td class="tg-0lax"><a href="./configs/COCO-InstanceSegmentation/mask_cascade_rcnn_ResNeSt_50_FPN_syncBN_1x.yaml">config</a> | <a href="https://s3.us-west-1.wasabisys.com/resnest/detectron/mask_cascade_rcnn_ResNeSt_50_FPN_syncBN_1x-c58bd325.pth">model</a> | <a href="https://s3.us-west-1.wasabisys.com/resnest/detectron/mask_cascade_rcnn_ResNeSt_50_FPN_syncBN_1x.txt">log</a> </td>
  </tr>
  <tr>
    <td class="tg-0lax">ResNeSt-101 (<span style="color:red">ours</span>)</td>
    <td class="tg-0lax"><b>48.30</b></td>
    <td class="tg-0lax"><b>41.56</b></td>
     <td class="tg-0lax"><a href="./configs/COCO-InstanceSegmentation/mask_cascade_rcnn_ResNeSt_101_FPN_syncBN_1x.yaml">config</a> | <a href="https://s3.us-west-1.wasabisys.com/resnest/detectron/mask_cascade_rcnn_ResNeSt_101_FPN_syncBN_1x-62448b9c.pth">model</a> | <a href="https://s3.us-west-1.wasabisys.com/resnest/detectron/mask_cascade_rcnn_ResNeSt_101_FPN_syncBN_1x.txt">log</a> </td>
  </tr>
  <tr>
    <td class="tg-0lax">ResNeSt-200-tricks-3x (<span style="color:red">ours</span>)</td>
    <td class="tg-0lax"><b>50.54</b></td>
    <td class="tg-0lax"><b>44.21</b></td>
     <td class="tg-0lax"><a href="./configs/COCO-InstanceSegmentation/mask_cascade_rcnn_ResNeSt_200_FPN_syncBN_all_tricks_3x.yaml">config</a> | <a href="https://s3.us-west-1.wasabisys.com/resnest/detectron/mask_cascade_rcnn_ResNeSt_200_FPN_syncBN_all_tricks_3x.pth">model</a> | <a href="https://s3.us-west-1.wasabisys.com/resnest/detectron/mask_cascade_rcnn_ResNeSt_200_FPN_syncBN_all_tricks_3x.txt">log</a> </td>
  </tr>
  <tr>
    <td rowspan="2" class="tg-0lax">ResNeSt-200-dcn-tricks-3x (<span style="color:red">ours</span>)</td>
    <td class="tg-0lax"><b>50.91</b></td>
    <td class="tg-0lax"><b>44.50</b></td>
     <td rowspan="2"class="tg-0lax"><a href="./configs/COCO-InstanceSegmentation/mask_cascade_rcnn_ResNeSt_200_FPN_dcn_syncBN_all_tricks_3x.yaml">config</a> | <a href="https://s3.us-west-1.wasabisys.com/resnest/detectron/mask_cascade_rcnn_ResNeSt_200_FPN_dcn_syncBN_all_tricks_3x-e1901134.pth">model</a> | <a href="https://s3.us-west-1.wasabisys.com/resnest/detectron/mask_cascade_rcnn_ResNeSt_200_FPN_dcn_syncBN_all_tricks_3x.txt">log</a> </td>
  </tr>
  <tr>
    <td class="tg-0lax"><b>53.30*</b></td>
    <td class="tg-0lax"><b>47.10*</b></td>
  </tr>
</table>

All models are trained along with FPN and SyncBN. For data augmentation,input images’ shorter side are randomly scaled to one of (640, 672, 704, 736, 768, 800). 1x learning rate schedule is used, if not otherwise specified. All of them are reported on COCO-2017 validation dataset. The values with * demonstrate the mutli-scale testing performance on the test-dev2019.



### Panoptic Segmentation
<table class="tg">
  <tr>
    <th class="tg-0pky">Backbone</th>
    <th class="tg-0pky">bbox</th>
    <th class="tg-0lax">mask</th>
    <th class="tg-0lax">PQ</th>
    <th class="tg-0pky">download</th>
  </tr>
  <tr>
    <td class="tg-0pky">ResNeSt-200</td>
    <td class="tg-0pky">51.00</td>
    <td class="tg-0lax">43.68</td>
    <td class="tg-0lax">47.90</td>
    <td class="tg-0lax"><a href="./configs/COCO-PanopticSegmentation/panoptic_ResNeSt_200_FPN_syncBN_tricks_3x.yaml">config</a> | <a href="https://s3.us-west-1.wasabisys.com/resnest/detectron/panoptic_ResNeSt_200_FPN_syncBN_tricks_3x-43f8b731.pth">model</a> | <a href="https://s3.us-west-1.wasabisys.com/resnest/detectron/panoptic_ResNeSt_200_FPN_syncBN_tricks_3x.txt">log</a> </td>
</tr> 
</table>


## Reference

**ResNeSt: Split-Attention Networks** [[arXiv](https://arxiv.org/pdf/2004.08955.pdf)]

Hang Zhang, Chongruo Wu, Zhongyue Zhang, Yi Zhu, Zhi Zhang, Haibin Lin, Yue Sun, Tong He, Jonas Muller, R. Manmatha, Mu Li and Alex Smola

```
@article{zhang2020resnest,
title={ResNeSt: Split-Attention Networks},
author={Zhang, Hang and Wu, Chongruo and Zhang, Zhongyue and Zhu, Yi and Zhang, Zhi and Lin, Haibin and Sun, Yue and He, Tong and Muller, Jonas and Manmatha, R. and Li, Mu and Smola, Alexander},
journal={arXiv preprint arXiv:2004.08955},
year={2020}
}
```
