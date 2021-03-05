# Bootstrapping Pipeline

Bootstrapping pipeline for DensePose was proposed in
[Sanakoyeu et al., 2020](https://arxiv.org/pdf/2003.00080.pdf)
to extend DensePose from humans to proximal animal classes
(chimpanzees). Currently, the pipeline is only implemented for
[chart-based models](DENSEPOSE_IUV.md).
Bootstrapping proceeds in two steps.

## Master Model Training

Master model is trained on data from source domain (humans)
and supporting domain (animals). Instances from the source domain
contain full DensePose annotations (`S`, `I`, `U` and `V`) and
instances from the supporting domain have segmentation annotations only.
To ensure segmentation quality in the target domain, only a subset of
supporting domain classes is included into the training. This is achieved
through category filters, e.g.
(see [configs/evolution/Base-RCNN-FPN-Atop10P_CA.yaml](../configs/evolution/Base-RCNN-FPN-Atop10P_CA.yaml)):

```
  WHITELISTED_CATEGORIES:
    "base_coco_2017_train":
      - 1  # person
      - 16 # bird
      - 17 # cat
      - 18 # dog
      - 19 # horse
      - 20 # sheep
      - 21 # cow
      - 22 # elephant
      - 23 # bear
      - 24 # zebra
      - 25 # girafe
```
The acronym `Atop10P` in config file names indicates that categories are filtered to
only contain top 10 animals and person.

The training is performed in a *class-agnostic* manner: all instances
are mapped into the same class (person), e.g.
(see [configs/evolution/Base-RCNN-FPN-Atop10P_CA.yaml](../configs/evolution/Base-RCNN-FPN-Atop10P_CA.yaml)):

```
  CATEGORY_MAPS:
    "base_coco_2017_train":
      "16": 1 # bird -> person
      "17": 1 # cat -> person
      "18": 1 # dog -> person
      "19": 1 # horse -> person
      "20": 1 # sheep -> person
      "21": 1 # cow -> person
      "22": 1 # elephant -> person
      "23": 1 # bear -> person
      "24": 1 # zebra -> person
      "25": 1 # girafe -> person
```
The acronym `CA` in config file names indicates that the training is class-agnostic.

## Student Model Training

Student model is trained on data from source domain (humans),
supporting domain (animals) and target domain (chimpanzees).
Annotations in source and supporting domains are similar to the ones
used for the master model training.
Annotations in target domain are obtained by applying the master model
to images that contain instances from the target category and sampling
sparse annotations from dense results. This process is called *bootstrapping*.
Below we give details on how the bootstrapping pipeline is implemented.

### Data Loaders

The central components that enable bootstrapping are
[`InferenceBasedLoader`](../densepose/data/inference_based_loader.py) and
[`CombinedDataLoader`](../densepose/data/combined_loader.py).

`InferenceBasedLoader` takes images from a data loader, applies a model
to the images, filters the model outputs based on the selected criteria and
samples the filtered outputs to produce annotations.

`CombinedDataLoader` combines data obtained from the loaders based on specified
ratios. The standard data loader has the default ratio of 1.0,
ratios for bootstrap datasets are specified in the configuration file.
The higher the ratio the higher the probability to include samples from the
particular data loader into a batch.

Here is an example of the bootstrapping configuration taken from
[`configs/evolution/densepose_R_50_FPN_DL_WC1M_3x_Atop10P_CA_B_uniform.yaml`](../configs/evolution/densepose_R_50_FPN_DL_WC1M_3x_Atop10P_CA_B_uniform.yaml):
```
BOOTSTRAP_DATASETS:
  - DATASET: "chimpnsee"
    RATIO: 1.0
    IMAGE_LOADER:
      TYPE: "video_keyframe"
      SELECT:
        STRATEGY: "random_k"
        NUM_IMAGES: 4
      TRANSFORM:
        TYPE: "resize"
        MIN_SIZE: 800
        MAX_SIZE: 1333
      BATCH_SIZE: 8
      NUM_WORKERS: 1
    INFERENCE:
      INPUT_BATCH_SIZE: 1
      OUTPUT_BATCH_SIZE: 1
    DATA_SAMPLER:
      # supported types:
      #   densepose_uniform
      #   densepose_UV_confidence
      #   densepose_fine_segm_confidence
      #   densepose_coarse_segm_confidence
      TYPE: "densepose_uniform"
      COUNT_PER_CLASS: 8
    FILTER:
      TYPE: "detection_score"
      MIN_VALUE: 0.8
BOOTSTRAP_MODEL:
  WEIGHTS: https://dl.fbaipublicfiles.com/densepose/evolution/densepose_R_50_FPN_DL_WC1M_3x_Atop10P_CA/217578784/model_final_9fe1cc.pkl
```

The above example has one bootstrap dataset (`chimpnsee`). This dataset is registered as
a [VIDEO_LIST](../densepose/data/datasets/chimpnsee.py) dataset, which means that
it consists of a number of videos specified in a text file. For videos there can be
different strategies to sample individual images. Here we use `video_keyframe` strategy
which considers only keyframes; this ensures temporal offset between sampled images and
faster seek operations. We select at most 4 random keyframes in each video:

```
SELECT:
  STRATEGY: "random_k"
  NUM_IMAGES: 4
```

The frames are then resized

```
TRANSFORM:
  TYPE: "resize"
  MIN_SIZE: 800
  MAX_SIZE: 1333
```

and batched using the standard
[PyTorch DataLoader](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader):

```
BATCH_SIZE: 8
NUM_WORKERS: 1
```

`InferenceBasedLoader` decomposes those batches into batches of size `INPUT_BATCH_SIZE`
and applies the master model specified by `BOOTSTRAP_MODEL`. Models outputs are filtered
by detection score:

```
FILTER:
  TYPE: "detection_score"
  MIN_VALUE: 0.8
```

and sampled using the specified sampling strategy:

```
DATA_SAMPLER:
  # supported types:
  #   densepose_uniform
  #   densepose_UV_confidence
  #   densepose_fine_segm_confidence
  #   densepose_coarse_segm_confidence
  TYPE: "densepose_uniform"
  COUNT_PER_CLASS: 8
```

The current implementation supports
[uniform sampling](../densepose/data/samplers/densepose_uniform.py) and
[confidence-based sampling](../densepose/data/samplers/densepose_confidence_based.py)
to obtain sparse annotations from dense results. For confidence-based
sampling one needs to use the master model which produces confidence estimates.
The `WC1M` master model used in the example above produces all three types of confidence
estimates.

Finally, sampled data is grouped into batches of size `OUTPUT_BATCH_SIZE`:

```
INFERENCE:
  INPUT_BATCH_SIZE: 1
  OUTPUT_BATCH_SIZE: 1
```

The proportion of data from annotated datasets and bootstrapped dataset can be tracked
in the logs, e.g.:

```
[... densepose.engine.trainer]: batch/ 1.8, batch/base_coco_2017_train 6.4, batch/densepose_coco_2014_train 3.85
```

which means that over the last 20 iterations, on average for 1.8 bootstrapped data samples there were 6.4 samples from `base_coco_2017_train` and 3.85 samples from `densepose_coco_2014_train`.
