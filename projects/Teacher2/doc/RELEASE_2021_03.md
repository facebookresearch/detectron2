# DensePose CSE and DensePose Evolution

* [DensePose Evolution pipeline](DENSEPOSE_IUV.md#ModelZooBootstrap), a framework to bootstrap
  DensePose on unlabeled data
  * [`InferenceBasedLoader`](../densepose/data/inference_based_loader.py)
    with data samplers to use inference results from one model
    to train another model (bootstrap);
  * [`VideoKeyframeDataset`](../densepose/data/video/video_keyframe_dataset.py)
    to efficiently load images from video keyframes;
  * Category maps and filters to combine annotations from different categories
    and train in a class-agnostic manner;
  * [Pretrained models](DENSEPOSE_IUV.md#ModelZooBootstrap) for DensePose estimation on chimpanzees;
  * DensePose head training from partial data (segmentation only);
  * [DensePose models with mask confidence estimation](DENSEPOSE_IUV.md#ModelZooMaskConfidence);
  * [DensePose Chimps]() dataset for IUV evaluation
* [DensePose Continuous Surface Embeddings](DENSEPOSE_CSE.md), a framework to extend DensePose
  to various categories using 3D models
  * [Hard embedding](../densepose/modeling/losses/embed.py) and
    [soft embedding](../densepose/modeling/losses/soft_embed.py)
    losses to train universal positional embeddings;
  * [Embedder](../(densepose/modeling/cse/embedder.py) to handle
    mesh vertex embeddings;
  * [Storage](../densepose/evaluation/tensor_storage.py) for evaluation with high volumes of data;
  * [Pretrained models](DENSEPOSE_CSE.md#ModelZoo) for DensePose CSE estimation on humans and animals;
  * [DensePose Chimps](DENSEPOSE_DATASETS.md#densepose-chimps) and
    [DensePose LVIS](DENSEPOSE_DATASETS.md#densepose-lvis) datasets for CSE finetuning and evaluation;
  * [Vertex and texture mapping visualizers](../densepose/vis/densepose_outputs_vertex.py);
* Refactoring of all major components: losses, predictors, model outputs, model results, visualizers;
  * Dedicated structures for [chart outputs](../densepose/structures/chart.py),
    [chart outputs with confidences](../densepose/structures/chart_confidence.py),
    [chart results](../densepose/structures/chart_result.py),
    [CSE outputs](../densepose/structures/cse.py);
  * Dedicated predictors for
    [chart-based estimation](../densepose/modeling/predictors/chart.py),
    [confidence estimation](../densepose/modeling/predictors/chart_confidence.py)
    and [CSE estimation](../densepose/modeling/predictors/cse.py);
  * Generic handling of various [conversions](../densepose/converters) (e.g. from outputs to results);
  * Better organization of various [losses](../densepose/modeling/losses);
  * Segregation of loss data accumulators for
    [IUV setting](../densepose/modeling/losses/utils.py)
    and [CSE setting](../densepose/modeling/losses/embed_utils.py);
  * Splitting visualizers into separate modules;
* [HRNet](../densepose/modeling/hrnet.py) and [HRFPN](../densepose/modeling/hrfpn.py) backbones;
* [PoseTrack](DENSEPOSE_DATASETS.md#densepose-posetrack) dataset;
* [IUV texture visualizer](../densepose/vis/densepose_results_textures.py)
