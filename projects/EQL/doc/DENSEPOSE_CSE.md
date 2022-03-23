# Continuous Surface Embeddings for Dense Pose Estimation for Humans and Animals

## <a name="Overview"></a> Overview

<div align="center">
  <img src="https://dl.fbaipublicfiles.com/densepose/web/densepose_cse_teaser.png" width="700px" />
</div>

The pipeline uses [Faster R-CNN](https://arxiv.org/abs/1506.01497)
with [Feature Pyramid Network](https://arxiv.org/abs/1612.03144) meta architecture
outlined in Figure 1. For each detected object, the model predicts
its coarse segmentation `S` (2 channels: foreground / background)
and the embedding `E` (16 channels). At the same time, the embedder produces vertex
embeddings `Ê` for the corresponding mesh. Universal positional embeddings `E`
and vertex embeddings `Ê` are matched to derive for each pixel its continuous
surface embedding.

<div align="center">
  <img src="https://dl.fbaipublicfiles.com/densepose/web/densepose_pipeline_cse.png" width="700px" />
</div>
<p class="image-caption"><b>Figure 1.</b> DensePose continuous surface embeddings architecture based on Faster R-CNN with Feature Pyramid Network (FPN).</p>

### Datasets

For more details on datasets used for training and validation of
continuous surface embeddings models,
please refer to the [DensePose Datasets](DENSEPOSE_DATASETS.md) page.

## <a name="ModelZoo"></a> Model Zoo and Baselines

### Human CSE Models

Continuous surface embeddings models for humans trained using the protocols from [Neverova et al, 2020](https://arxiv.org/abs/2011.12438).

Models trained with hard assignment loss &#x2112;:

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Name</th>
<th valign="bottom">lr<br/>sched</th>
<th valign="bottom">train<br/>time<br/>(s/iter)</th>
<th valign="bottom">inference<br/>time<br/>(s/im)</th>
<th valign="bottom">train<br/>mem<br/>(GB)</th>
<th valign="bottom">box<br/>AP</th>
<th valign="bottom">segm<br/>AP</th>
<th valign="bottom">dp. AP<br/>GPS</th>
<th valign="bottom">dp. AP<br/>GPSm</th>
<th valign="bottom">model id</th>
<th valign="bottom">download</th>
<!-- TABLE BODY -->
<!-- ROW: densepose_rcnn_R_50_FPN_s1x -->
<tr><td align="left"><a href="../configs/cse/densepose_rcnn_R_50_FPN_s1x.yaml">R_50_FPN_s1x</a></td>
 <td align="center">s1x</td>
 <td align="center">0.349</td>
 <td align="center">0.060</td>
 <td align="center">6.3</td>
 <td align="center">61.1</td>
 <td align="center">67.1</td>
 <td align="center">64.4</td>
 <td align="center">65.7</td>
 <td align="center">251155172</td>
 <td align="center"><a href="https://dl.fbaipublicfiles.com/densepose/cse/densepose_rcnn_R_50_FPN_s1x/251155172/model_final_c4ea5f.pkl">model</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/densepose/cse/densepose_rcnn_R_50_FPN_s1x/251155172/metrics.json">metrics</a></td>
</tr>
<!-- ROW: densepose_rcnn_R_101_FPN_s1x -->
<tr><td align="left"><a href="../configs/cse/densepose_rcnn_R_101_FPN_s1x.yaml">R_101_FPN_s1x</a></td>
  <td align="center">s1x</td>
  <td align="center">0.461</td>
  <td align="center">0.071</td>
  <td align="center">7.4</td>
  <td align="center">62.3</td>
  <td align="center">67.2</td>
  <td align="center">64.7</td>
  <td align="center">65.8</td>
  <td align="center">251155500</td>
  <td align="center"><a href="https://dl.fbaipublicfiles.com/densepose/cse/densepose_rcnn_R_101_FPN_s1x/251155500/model_final_5c995f.pkl">model</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/densepose/cse/densepose_rcnn_R_101_FPN_s1x/251155500/metrics.json">metrics</a></td>
</tr>
<!-- ROW: densepose_rcnn_R_50_FPN_DL_s1x -->
 <tr><td align="left"><a href="../configs/cse/densepose_rcnn_R_50_FPN_DL_s1x.yaml">R_50_FPN_DL_s1x</a></td>
 <td align="center">s1x</td>
 <td align="center">0.399</td>
 <td align="center">0.061</td>
 <td align="center">7.0</td>
 <td align="center">60.8</td>
 <td align="center">67.8</td>
 <td align="center">65.5</td>
 <td align="center">66.4</td>
 <td align="center">251156349</td>
 <td align="center"><a href="https://dl.fbaipublicfiles.com/densepose/cse/densepose_rcnn_R_50_FPN_DL_s1x/251156349/model_final_e96218.pkl">model</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/densepose/cse/densepose_rcnn_R_50_FPN_DL_s1x/251156349/metrics.json">metrics</a></td>
</tr>
<!-- ROW: densepose_rcnn_R_101_FPN_DL_s1x -->
<tr><td align="left"><a href="../configs/cse/densepose_rcnn_R_101_FPN_DL_s1x.yaml">R_101_FPN_DL_s1x</a></td>
  <td align="center">s1x</td>
  <td align="center">0.504</td>
  <td align="center">0.074</td>
  <td align="center">8.3</td>
  <td align="center">61.5</td>
  <td align="center">68.0</td>
  <td align="center">65.6</td>
  <td align="center">66.6</td>
  <td align="center">251156606</td>
  <td align="center"><a href="https://dl.fbaipublicfiles.com/densepose/cse/densepose_rcnn_R_101_FPN_DL_s1x/251156606/model_final_b236ce.pkl">model</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/densepose/cse/densepose_rcnn_R_101_FPN_DL_s1x/251156606/metrics.json">metrics</a></td>
</tr>
</tbody></table>

Models trained with soft assignment loss &#x2112;<sub>&sigma;</sub>:

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Name</th>
<th valign="bottom">lr<br/>sched</th>
<th valign="bottom">train<br/>time<br/>(s/iter)</th>
<th valign="bottom">inference<br/>time<br/>(s/im)</th>
<th valign="bottom">train<br/>mem<br/>(GB)</th>
<th valign="bottom">box<br/>AP</th>
<th valign="bottom">segm<br/>AP</th>
<th valign="bottom">dp. AP<br/>GPS</th>
<th valign="bottom">dp. AP<br/>GPSm</th>
<th valign="bottom">model id</th>
<th valign="bottom">download</th>
<!-- TABLE BODY -->
<!-- ROW: densepose_rcnn_R_50_FPN_soft_s1x -->
<tr><td align="left"><a href="../configs/cse/densepose_rcnn_R_50_FPN_soft_s1x.yaml">R_50_FPN_soft_s1x</a></td>
<td align="center">s1x</td>
<td align="center">0.357</td>
<td align="center">0.057</td>
<td align="center">9.7</td>
<td align="center">61.3</td>
<td align="center">66.9</td>
<td align="center">64.3</td>
<td align="center">65.4</td>
<td align="center">250533982</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/densepose/cse/densepose_rcnn_R_50_FPN_soft_s1x/250533982/model_final_2c4512.pkl">model</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/densepose/cse/densepose_rcnn_R_50_FPN_soft_s1x/250533982/metrics.json">metrics</a></td>
</tr>
<!-- ROW: densepose_rcnn_R_101_FPN_soft_s1x -->
<tr><td align="left"><a href="../configs/cse/densepose_rcnn_R_101_FPN_soft_s1x.yaml">R_101_FPN_soft_s1x</a></td>
<td align="center">s1x</td>
<td align="center">0.464</td>
<td align="center">0.071</td>
<td align="center">10.5</td>
<td align="center">62.1</td>
<td align="center">67.3</td>
<td align="center">64.5</td>
<td align="center">66.0</td>
<td align="center">250712522</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/densepose/cse/densepose_rcnn_R_101_FPN_soft_s1x/250712522/model_final_4637da.pkl">model</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/densepose/cse/densepose_rcnn_R_101_FPN_soft_s1x/250712522/metrics.json">metrics</a></td>
</tr>
<!-- ROW: densepose_rcnn_R_50_FPN_DL_soft_s1x -->
<tr><td align="left"><a href="../configs/cse/densepose_rcnn_R_50_FPN_DL_soft_s1x.yaml">R_50_FPN_DL_soft_s1x</a></td>
<td align="center">s1x</td>
<td align="center">0.427</td>
<td align="center">0.062</td>
<td align="center">11.3</td>
<td align="center">60.8</td>
<td align="center">68.0</td>
<td align="center">66.1</td>
<td align="center">66.7</td>
<td align="center">250713703</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/densepose/cse/densepose_rcnn_R_50_FPN_DL_soft_s1x/250713703/model_final_9199f5.pkl">model</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/densepose/cse/densepose_rcnn_R_50_FPN_DL_soft_s1x/250713703/metrics.json">metrics</a></td>
</tr>
<!-- ROW: densepose_rcnn_R_101_FPN_DL_soft_s1x -->
<tr><td align="left"><a href="../configs/cse/densepose_rcnn_R_101_FPN_DL_soft_s1x.yaml">R_101_FPN_DL_soft_s1x</a></td>
<td align="center">s1x</td>
<td align="center">0.483</td>
<td align="center">0.071</td>
<td align="center">12.2</td>
<td align="center">61.5</td>
<td align="center">68.2</td>
<td align="center">66.2</td>
<td align="center">67.1</td>
<td align="center">250713061</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/densepose/cse/densepose_rcnn_R_101_FPN_DL_soft_s1x/250713061/model_final_1d3314.pkl">model</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/densepose/cse/densepose_rcnn_R_101_FPN_DL_soft_s1x/250713061/metrics.json">metrics</a></td>
</tr>
</tbody></table>

### Animal CSE Models

Models obtained by finetuning human CSE models on animals data with soft assignment loss &#x2112;<sub>&sigma;</sub>:

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Name</th>
<th valign="bottom">lr<br/>sched</th>
<th valign="bottom">train<br/>time<br/>(s/iter)</th>
<th valign="bottom">inference<br/>time<br/>(s/im)</th>
<th valign="bottom">train<br/>mem<br/>(GB)</th>
<th valign="bottom">box<br/>AP</th>
<th valign="bottom">segm<br/>AP</th>
<th valign="bottom">dp. AP<br/>GPS</th>
<th valign="bottom">dp. AP<br/>GPSm</th>
<th valign="bottom">model id</th>
<th valign="bottom">download</th>
<!-- TABLE BODY -->
<!-- ROW: densepose_rcnn_R_50_FPN_soft_chimps_finetune_4k -->
 <tr><td align="left"><a href="../configs/cse/densepose_rcnn_R_50_FPN_soft_chimps_finetune_4k.yaml">R_50_FPN_soft_chimps_finetune_4k</a></td>
<td align="center">4K</td>
<td align="center">0.569</td>
<td align="center">0.051</td>
<td align="center">4.7</td>
<td align="center">62.0</td>
<td align="center">59.0</td>
<td align="center">32.2</td>
<td align="center">39.6</td>
<td align="center">253146869</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/densepose/cse/densepose_rcnn_R_50_FPN_soft_chimps_finetune_4k/253146869/model_final_52f649.pkl">model</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/densepose/cse/densepose_rcnn_R_50_FPN_soft_chimps_finetune_4k/253146869/metrics.json">metrics</a></td>
</tr>
<!-- ROW: densepose_rcnn_R_50_FPN_soft_animals_finetune_4k -->
<tr><td align="left"><a href="../configs/cse/densepose_rcnn_R_50_FPN_soft_animals_finetune_4k.yaml">R_50_FPN_soft_animals_finetune_4k</a></td>
<td align="center">4K</td>
<td align="center">0.381</td>
<td align="center">0.061</td>
<td align="center">7.3</td>
<td align="center">44.9</td>
<td align="center">55.5</td>
<td align="center">21.3</td>
<td align="center">28.8</td>
<td align="center">253145793</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/densepose/cse/densepose_rcnn_R_50_FPN_soft_animals_finetune_4k/253145793/model_final_8f8ba2.pkl">model</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/densepose/cse/densepose_rcnn_R_50_FPN_soft_animals_finetune_4k/253145793/metrics.json">metrics</a></td>
</tr>
<!-- ROW: densepose_rcnn_R_50_FPN_soft_animals_CA_finetune_4k -->
 <tr><td align="left"><a href="../configs/cse/densepose_rcnn_R_50_FPN_soft_animals_CA_finetune_4k.yaml">R_50_FPN_soft_animals_CA_finetune_4k</a></td>
<td align="center">4K</td>
<td align="center">0.412</td>
<td align="center">0.059</td>
<td align="center">7.1</td>
<td align="center">53.4</td>
<td align="center">59.5</td>
<td align="center">25.4</td>
<td align="center">33.4</td>
<td align="center">253498611</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/densepose/cse/densepose_rcnn_R_50_FPN_soft_animals_CA_finetune_4k/253498611/model_final_6d69b7.pkl">model</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/densepose/cse/densepose_rcnn_R_50_FPN_soft_animals_CA_finetune_4k/253498611/metrics.json">metrics</a></td>
</tr>
</tbody></table>

Acronyms:

`CA`: class agnostic training, where all annotated instances are mapped into a single category

## <a name="References"></a> References

If you use DensePose methods based on continuous surface embeddings, please take the
references from the following BibTeX entries:

Continuous surface embeddings:
```
@InProceedings{Neverova2020ContinuousSurfaceEmbeddings,
    title = {Continuous Surface Embeddings},
    author = {Neverova, Natalia and Novotny, David and Khalidov, Vasil and Szafraniec, Marc and Labatut, Patrick and Vedaldi, Andrea},
    journal = {Advances in Neural Information Processing Systems},
    year = {2020},
}
```
