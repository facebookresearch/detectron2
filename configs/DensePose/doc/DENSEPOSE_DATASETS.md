# DensePose Datasets

We summarize the datasets used in various DensePose training
schedules and describe different available annotation types.

## Table of Contents

[General Information](#general-information)

[DensePose COCO](#densepose-coco)

[DensePose PoseTrack](#densepose-posetrack)

[DensePose Chimps](#densepose-chimps)

[DensePose LVIS](#densepose-lvis)

## General Information

DensePose annotations are typically stored in JSON files. Their
structure follows the [COCO Data Format](https://cocodataset.org/#format-data),
the basic data structure is outlined below:

```
{
    "info": info,
    "images": [image],
    "annotations": [annotation],
    "licenses": [license],
}

info{
    "year": int,
    "version": str,
    "description": str,
    "contributor": str,
    "url": str,
    "date_created": datetime,
}

image{
    "id": int,
    "width": int,
    "height": int,
    "file_name": str,
    "license": int,
    "flickr_url": str,
    "coco_url": str,
    "date_captured": datetime,
}

license{
    "id": int, "name": str, "url": str,
}
```

DensePose annotations can be of two types:
*chart-based annotations* or *continuous surface embeddings annotations*.
We give more details on each of the two annotation types below.

### Chart-based Annotations

These annotations assume a single 3D model which corresponds to
all the instances in a given dataset.
3D model is assumed to be split into *charts*. Each chart has its own
2D parametrization through inner coordinates `U` and `V`, typically
taking values in `[0, 1]`.

Chart-based annotations consist of *point-based annotations* and
*segmentation annotations*. Point-based annotations specify, for a given
image point, which model part it belongs to and what are its coordinates
in the corresponding chart. Segmentation annotations specify regions
in an image that are occupied by a given part. In some cases, charts
associated with point annotations are more detailed than the ones
associated with segmentation annotations. In this case we distinguish
*fine segmentation* (associated with points) and *coarse segmentation*
(associated with masks).

**Point-based annotations**:

`dp_x` and `dp_y`:  image coordinates of the annotated points along
the horizontal and vertical axes respectively. The coordinates are defined
with respect to the top-left corner of the annotated bounding box and are
normalized assuming the bounding box size to be `256x256`;

`dp_I`: for each point specifies the index of the fine segmentation chart
it belongs to;

`dp_U` and `dp_V`: point coordinates on the corresponding chart.
Each fine segmentation part has its own parametrization in terms of chart
coordinates.

**Segmentation annotations**:

`dp_masks`: RLE encoded dense masks (`dict` containing keys `counts` and `size`).
The masks are typically of size `256x256`, they define segmentation within the
bounding box.

### Continuous Surface Embeddings Annotations

Continuous surface embeddings annotations also consist of *point-based annotations*
and *segmentation annotations*. Point-based annotations establish correspondence
between image points and 3D model vertices. Segmentation annotations specify
foreground regions for a given instane.

**Point-based annotations**:

`dp_x` and `dp_y` specify image point coordinates the same way as for chart-based
annotations;

`dp_vertex` gives indices of 3D model vertices, which the annotated image points
correspond to;

`ref_model` specifies 3D model name.

**Segmentation annotations**:

Segmentations can either be given by `dp_masks` field or by `segmentation` field.

`dp_masks`: RLE encoded dense masks (`dict` containing keys `counts` and `size`).
The masks are typically of size `256x256`, they define segmentation within the
bounding box.

`segmentation`: polygon-based masks stored as a 2D list
`[[x1 y1 x2 y2...],[x1 y1 ...],...]` of polygon vertex coordinates in a given
image.

## DensePose COCO

<div align="center">
  <img src="http://cocodataset.org/images/densepose-splash.png" width="700px" />
</div>
<p class="image-caption">
  <b>Figure 1.</b> Annotation examples from the DensePose COCO dataset.
</p>

DensePose COCO dataset contains about 50K annotated persons on images from the
[COCO dataset](https://cocodataset.org/#home)
The images are available for download from the
[COCO Dataset download page](https://cocodataset.org/#download):
[train2014](http://images.cocodataset.org/zips/train2014.zip),
[val2014](http://images.cocodataset.org/zips/val2014.zip).
The details on available annotations and their download links are given below.

### Chart-based Annotations

Chart-based DensePose COCO annotations are available for the instances of category
`person` and correspond to the model shown in Figure 2.
They include `dp_x`, `dp_y`, `dp_I`, `dp_U` and `dp_V` fields for annotated points
(~100 points per annotated instance) and `dp_masks` field, which encodes
coarse segmentation into 14 parts in the following order:
`Torso`, `Right Hand`, `Left Hand`, `Left Foot`, `Right Foot`,
`Upper Leg Right`, `Upper Leg Left`, `Lower Leg Right`, `Lower Leg Left`,
`Upper Arm Left`, `Upper Arm Right`, `Lower Arm Left`, `Lower Arm Right`,
`Head`.

<div align="center">
  <img src="https://dl.fbaipublicfiles.com/densepose/web/densepose_human_charts_wcoarse.png" width="500px" />
</div>
<p class="image-caption">
  <b>Figure 2.</b> Human body charts (<i>fine segmentation</i>)
  and the associated 14 body parts depicted with rounded rectangles
  (<i>coarse segmentation</i>).
</p>

The dataset splits used in the training schedules are
`train2014`, `valminusminival2014` and `minival2014`.
`train2014` and `valminusminival2014` are used for training,
and `minival2014` is used for validation.
The table with annotation download links, which summarizes the number of annotated
instances and images for each of the dataset splits is given below:
<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Name</th>
<th valign="bottom"># inst</th>
<th valign="bottom"># images</th>
<th valign="bottom">file size</th>
<th valign="bottom">download</th>
<!-- TABLE BODY -->
<!-- ROW: densepose_train2014 -->
<tr><td align="left">densepose_train2014</td>
<td align="center">39210</td>
<td align="center">26437</td>
<td valign="center">526M</td>
<td align="left"><a href="https://dl.fbaipublicfiles.com/densepose/annotations/coco/densepose_train2014.json">densepose_train2014.json</a></td>
</tr>
<!-- ROW: densepose_valminusminival2014 -->
<tr><td align="left">densepose_valminusminival2014</td>
<td align="center">7297</td>
<td align="center">5984</td>
<td valign="center">105M</td>
<td align="left"><a href="https://dl.fbaipublicfiles.com/densepose/annotations/coco/densepose_valminusminival2014.json">densepose_valminusminival2014.json</a></td>
</tr>
<!-- ROW: densepose_minival2014 -->
<tr><td align="left">densepose_minival2014</td>
<td align="center">2243</td>
<td align="center">1508</td>
<td valign="center">31M</td>
<td align="left"><a href="https://dl.fbaipublicfiles.com/densepose/annotations/coco/densepose_minival2014.json">densepose_minival2014.json</a></td>
</tr>
</tbody></table>

### Continuous Surface Embeddings Annotations

DensePose COCO continuous surface embeddings annotations are available for the instances
of category `person`. The annotations correspond to the 3D model shown in Figure 2,
and include `dp_x`, `dp_y` and `dp_vertex` and `ref_model` fields.
All chart-based annotations were also kept for convenience.

As with chart-based annotations, the dataset splits used in the training schedules are
`train2014`, `valminusminival2014` and `minival2014`.
`train2014` and `valminusminival2014` are used for training,
and `minival2014` is used for validation.
The table with annotation download links, which summarizes the number of annotated
instances and images for each of the dataset splits is given below:
<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Name</th>
<th valign="bottom"># inst</th>
<th valign="bottom"># images</th>
<th valign="bottom">file size</th>
<th valign="bottom">download</th>
<!-- TABLE BODY -->
<!-- ROW: densepose_train2014_cse -->
<tr><td align="left">densepose_train2014_cse</td>
<td align="center">39210</td>
<td align="center">26437</td>
<td valign="center">554M</td>
<td align="left"><a href="https://dl.fbaipublicfiles.com/densepose/annotations/coco_cse/densepose_train2014_cse.json">densepose_train2014_cse.json</a></td>
</tr>
<!-- ROW: densepose_valminusminival2014_cse -->
<tr><td align="left">densepose_valminusminival2014_cse</td>
<td align="center">7297</td>
<td align="center">5984</td>
<td valign="center">110M</td>
<td align="left"><a href="https://dl.fbaipublicfiles.com/densepose/annotations/coco_cse/densepose_valminusminival2014_cse.json">densepose_valminusminival2014_cse.json</a></td>
</tr>
<!-- ROW: densepose_minival2014_cse -->
<tr><td align="left">densepose_minival2014_cse</td>
<td align="center">2243</td>
<td align="center">1508</td>
<td valign="center">32M</td>
<td align="left"><a href="https://dl.fbaipublicfiles.com/densepose/annotations/coco_cse/densepose_minival2014_cse.json">densepose_minival2014_cse.json</a></td>
</tr>
</tbody></table>

## DensePose PoseTrack

<div align="center">
  <img src="https://posetrack.net/workshops/eccv2018/assets/images/densepose-posetrack_examples.jpg" width="700px" />
</div>
<p class="image-caption">
  <b>Figure 3.</b> Annotation examples from the PoseTrack dataset.
</p>

DensePose PoseTrack dataset contains annotated image sequences.
To download the images for this dataset, please follow the instructions
from the [PoseTrack Download Page](https://posetrack.net/users/download.php).

### Chart-based Annotations

Chart-based DensePose PoseTrack annotations are available for the instances with category
`person` and correspond to the model shown in Figure 2.
They include `dp_x`, `dp_y`, `dp_I`, `dp_U` and `dp_V` fields for annotated points
(~100 points per annotated instance) and `dp_masks` field, which encodes
coarse segmentation into the same 14 parts as in DensePose COCO.

The dataset splits used in the training schedules are
`posetrack_train2017` (train set) and `posetrack_val2017` (validation set).
The table with annotation download links, which summarizes the number of annotated
instances, instance tracks and images for the dataset splits is given below:
<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Name</th>
<th valign="bottom"># inst</th>
<th valign="bottom"># images</th>
<th valign="bottom"># tracks</th>
<th valign="bottom">file size</th>
<th valign="bottom">download</th>
<!-- TABLE BODY -->
<!-- ROW: densepose_posetrack_train2017 -->
<tr><td align="left">densepose_posetrack_train2017</td>
<td align="center">8274</td>
<td align="center">1680</td>
<td align="center">36</td>
<td valign="center">118M</td>
<td align="left"><a href="https://dl.fbaipublicfiles.com/densepose/annotations/coco/densepose_posetrack_train2017.json">densepose_posetrack_train2017.json</a></td>
</tr>
<!-- ROW: densepose_posetrack_val2017 -->
<tr><td align="left">densepose_posetrack_val2017</td>
<td align="center">4753</td>
<td align="center">782</td>
<td align="center">46</td>
<td valign="center">59M</td>
<td align="left"><a href="https://dl.fbaipublicfiles.com/densepose/annotations/coco/densepose_posetrack_val2017.json">densepose_posetrack_val2017.json</a></td>
</tr>
</tbody></table>

## DensePose Chimps

<div align="center">
  <img src="https://dl.fbaipublicfiles.com/densepose/web/densepose_chimps_preview.jpg" width="700px" />
</div>
<p class="image-caption">
  <b>Figure 4.</b> Example images from the DensePose Chimps dataset.
</p>

DensePose Chimps dataset contains annotated images of chimpanzees.
To download the images for this dataset, please use the URL specified in
`image_url` field in the annotations.

### Chart-based Annotations

Chart-based DensePose Chimps annotations correspond to the human model shown in Figure 2,
the instances are thus annotated to belong to the `person` category.
They include `dp_x`, `dp_y`, `dp_I`, `dp_U` and `dp_V` fields for annotated points
(~3 points per annotated instance) and `dp_masks` field, which encodes
foreground mask in RLE format.

Chart-base DensePose Chimps annotations are used for validation only.
The table with annotation download link, which summarizes the number of annotated
instances and images is given below:

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Name</th>
<th valign="bottom"># inst</th>
<th valign="bottom"># images</th>
<th valign="bottom">file size</th>
<th valign="bottom">download</th>
<!-- TABLE BODY -->
<!-- ROW: densepose_chimps -->
<tr><td align="left">densepose_chimps</td>
<td align="center">930</td>
<td align="center">654</td>
<td valign="center">6M</td>
<td align="left"><a href="https://dl.fbaipublicfiles.com/densepose/annotations/densepose_chimps/densepose_chimps_full_v2.json">densepose_chimps_full_v2.json</a></td>
</tr>
</tbody></table>

### Continuous Surface Embeddings Annotations

Continuous surface embeddings annotations for DensePose Chimps
include `dp_x`, `dp_y` and `dp_vertex` point-based annotations
(~3 points per annotated instance), `dp_masks` field with the same
contents as for chart-based annotations and `ref_model` field
which refers to a chimpanzee 3D model `chimp_5029`.

The dataset is split into training and validation subsets.
The table with annotation download links, which summarizes the number of annotated
instances and images for each of the dataset splits is given below:

The table below outlines the dataset splits:
<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Name</th>
<th valign="bottom"># inst</th>
<th valign="bottom"># images</th>
<th valign="bottom">file size</th>
<th valign="bottom">download</th>
<!-- TABLE BODY -->
<!-- ROW: densepose_chimps_cse_train -->
<tr><td align="left">densepose_chimps_cse_train</td>
<td align="center">500</td>
<td align="center">350</td>
<td valign="center">3M</td>
<td align="left"><a href="https://dl.fbaipublicfiles.com/densepose/annotations/densepose_chimps/densepose_chimps_cse_train.json">densepose_chimps_cse_train.json</a></td>
</tr>
<!-- ROW: densepose_chimps_cse_val -->
<tr><td align="left">densepose_chimps_cse_val</td>
<td align="center">430</td>
<td align="center">304</td>
<td valign="center">3M</td>
<td align="left"><a href="https://dl.fbaipublicfiles.com/densepose/annotations/densepose_chimps/densepose_chimps_cse_val.json">densepose_chimps_cse_val.json</a></td>
</tr>
</tbody></table>

## DensePose LVIS

<div align="center">
  <img src="https://dl.fbaipublicfiles.com/densepose/web/lvis_selected_animals_preview.jpg" width="700px" />
</div>
<p class="image-caption">
  <b>Figure 5.</b> Example images from the DensePose LVIS dataset.
</p>

DensePose LVIS dataset contains segmentation and DensePose annotations for animals
on images from the [LVIS dataset](https://www.lvisdataset.org/dataset).
The images are available for download through the links:
[train2017](http://images.cocodataset.org/zips/train2017.zip),
[val2017](http://images.cocodataset.org/zips/val2017.zip).

### Continuous Surface Embeddings Annotations

Continuous surface embeddings (CSE) annotations for DensePose LVIS
include `dp_x`, `dp_y` and `dp_vertex` point-based annotations
(~3 points per annotated instance) and a `ref_model` field
which refers to a 3D model that corresponds to the instance.
Instances from 9 animal categories were annotated with CSE DensePose data:
bear, cow, cat, dog, elephant, giraffe, horse, sheep and zebra.

Foreground masks are available from instance segmentation annotations
(`segmentation` field) in polygon format, they are stored as a 2D list
`[[x1 y1 x2 y2...],[x1 y1 ...],...]`.

We used two datasets, each constising of one training (`train`)
and validation (`val`) subsets: the first one (`ds1`)
was used in [Neverova et al, 2020](https://arxiv.org/abs/2011.12438).
The second one (`ds2`), was used in [Neverova et al, 2021]().

The summary of the available datasets is given below:
<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<tr>
<th valign="bottom"></th>
<th valign="bottom" colspan="3">All Data</th>
<th valign="bottom" colspan="3">Selected Animals<br>(9 categories)</th>
<th valign="bottom" colspan="2">File</th>
</tr>
<tr>
<th valign="bottom">Name</th>
<th valign="bottom"># cat</th>
<th valign="bottom"># img</th>
<th valign="bottom"># segm</th>
<th valign="bottom"># img</th>
<th valign="bottom"># segm</th>
<th valign="bottom"># dp</th>
<th valign="bottom">size</th>
<th valign="bottom">download</th>
</tr>
<!-- TABLE BODY -->
<!-- ROW: densepose_lvis_v1_ds1_train_v1 -->
<tr><td align="left">ds1_train</td>
<td align="center">556</td>
<td align="center">4141</td>
<td align="center">23985</td>
<td align="center">4141</td>
<td align="center">9472</td>
<td align="center">5184</td>
<td valign="center">46M</td>
<td align="left"><a href="https://dl.fbaipublicfiles.com/densepose/annotations/lvis/densepose_lvis_v1_ds1_train_v1.json">densepose_lvis_v1_ds1_train_v1.json</a></td>
</tr>
<!-- ROW: densepose_lvis_v1_ds1_val_v1 -->
<tr><td align="left">ds1_val</td>
<td align="center">251</td>
<td align="center">571</td>
<td align="center">3281</td>
<td align="center">571</td>
<td align="center">1537</td>
<td align="center">1036</td>
<td valign="center">5M</td>
<td align="left"><a href="https://dl.fbaipublicfiles.com/densepose/annotations/lvis/densepose_lvis_v1_ds1_val_v1.json">densepose_lvis_v1_ds1_val_v1.json</a></td>
</tr>
<!-- ROW: densepose_lvis_v1_ds2_train_v1 -->
<tr><td align="left">ds2_train</td>
<td align="center">1203</td>
<td align="center">99388</td>
<td align="center">1270141</td>
<td align="center">13746</td>
<td align="center">46964</td>
<td align="center">18932</td>
<td valign="center">1051M</td>
<td align="left"><a href="https://dl.fbaipublicfiles.com/densepose/annotations/lvis/densepose_lvis_v1_ds2_train_v1.json">densepose_lvis_v1_ds2_train_v1.json</a></td>
</tr>
<!-- ROW: densepose_lvis_v1_ds2_val_v1 -->
<tr><td align="left">ds2_val</td>
<td align="center">9</td>
<td align="center">2690</td>
<td align="center">9155</td>
<td align="center">2690</td>
<td align="center">9155</td>
<td align="center">3604</td>
<td valign="center">24M</td>
<td align="left"><a href="https://dl.fbaipublicfiles.com/densepose/annotations/lvis/densepose_lvis_v1_ds2_val_v1.json">densepose_lvis_v1_ds2_val_v1.json</a></td>
</tr>
</tbody></table>

Legend:

`#cat` - number of categories in the dataset for which annotations are available;

`#img` - number of images with annotations in the dataset;

`#segm` - number of segmentation annotations;

`#dp` - number of DensePose annotations.


Important Notes:

1. The reference models used for `ds1_train` and `ds1_val` are
`bear_4936`, `cow_5002`, `cat_5001`, `dog_5002`, `elephant_5002`, `giraffe_5002`,
`horse_5004`, `sheep_5004` and `zebra_5002`. The reference models used for
`ds2_train` and `ds2_val` are `bear_4936`, `cow_5002`, `cat_7466`,
`dog_7466`, `elephant_5002`, `giraffe_5002`, `horse_5004`, `sheep_5004` and `zebra_5002`.
So reference models for categories `cat` aind `dog` are different for `ds1` and `ds2`.

2. Some annotations from `ds1_train` are reused in `ds2_train` (4538 DensePose annotations
and 21275 segmentation annotations). The ones for cat and dog categories were remapped
from `cat_5001` and `dog_5002` reference models used in `ds1` to `cat_7466` and `dog_7466`
used in `ds2`.

3. All annotations from `ds1_val` are included into `ds2_val` after the remapping
procedure mentioned in note 2.

4. Some annotations from `ds1_train` are part of `ds2_val` (646 DensePose annotations and
1225 segmentation annotations). Thus one should not train on `ds1_train` if evaluating on `ds2_val`.
