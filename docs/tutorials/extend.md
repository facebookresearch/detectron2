# Extend Detectron2's Defaults

__Research is about doing things in new ways__.
This brings a tension in how to create abstractions in code,
which is a challenge for any research engineering project of a significant size:

1. On one hand, it needs to have very thin abstractions to allow for the possibility of doing
   everything in new ways. It should be reasonably easy to break existing
   abstractions and replace them with new ones.

2. On the other hand, such a project also needs reasonably high-level
   abstractions, so that users can easily do things in standard ways,
   without worrying too much about the details that only certain researchers care about.

In detectron2, there are two types of interfaces that address this tension together:

1. Functions and classes that take a config (`cfg`) argument
   (sometimes with only a few extra arguments).

   Such functions and classes implement
   the "standard default" behavior: it will read what it needs from the
   config and do the "standard" thing.
   Users only need to load a given config and pass it around, without having to worry about
   which arguments are used and what they all mean.

2. Functions and classes that have well-defined explicit arguments.

   Each of these is a small building block of the entire system.
   They require users' expertise to understand what each argument should be,
   and require more effort to stitch together to a larger system.
   But they can be stitched together in more flexible ways.

   When you need to implement something not supported by the "standard defaults"
   included in detectron2, these well-defined components can be reused.

3. (experimental) A few classes are implemented with the
   [@configurable](../../modules/config.html#detectron2.config.configurable)
   decorator - they can be called with either a config, or with explicit arguments.
   Their explicit argument interfaces are currently __experimental__ and subject to change.

   As an example, a Mask R-CNN model can be built like this:
   <details>
   <summary>
   (click to expand)
   </summary>

   ```python
   model = GeneralizedRCNN(
       backbone=FPN(
           ResNet(
               BasicStem(3, 64),
               [
                   ResNet.make_stage(
                       BottleneckBlock,
                       n,
                       s,
                       in_channels=i,
                       bottleneck_channels=o // 4,
                       out_channels=o,
                       stride_in_1x1=True,
                   )
                   for (n, s, i, o) in zip(
                       [3, 4, 6, 3], [1, 2, 2, 2], [64, 256, 512, 1024], [256, 512, 1024, 2048]
                   )
               ],
               out_features=["res2", "res3", "res4", "res5"],
           ).freeze(2),
           ["res2", "res3", "res4", "res5"],
           256,
           top_block=LastLevelMaxPool(),
       ),
       proposal_generator=RPN(
           in_features=["p2", "p3", "p4", "p5", "p6"],
           head=StandardRPNHead(in_channels=256, num_anchors=3),
           anchor_generator=DefaultAnchorGenerator(
               sizes=[[32], [64], [128], [256], [512]],
               aspect_ratios=[0.5, 1.0, 2.0],
               strides=[4, 8, 16, 32, 64],
               offset=0.0,
           ),
           anchor_matcher=Matcher([0.3, 0.7], [0, -1, 1], allow_low_quality_matches=True),
           box2box_transform=Box2BoxTransform([1.0, 1.0, 1.0, 1.0]),
           batch_size_per_image=256,
           positive_fraction=0.5,
           pre_nms_topk=(2000, 1000),
           post_nms_topk=(1000, 1000),
           nms_thresh=0.7,
       ),
       roi_heads=StandardROIHeads(
           num_classes=80,
           batch_size_per_image=512,
           positive_fraction=0.25,
           proposal_matcher=Matcher([0.5], [0, 1], allow_low_quality_matches=False),
           box_in_features=["p2", "p3", "p4", "p5"],
           box_pooler=ROIPooler(7, (1.0 / 4, 1.0 / 8, 1.0 / 16, 1.0 / 32), 0, "ROIAlignV2"),
           box_head=FastRCNNConvFCHead(
               ShapeSpec(channels=256, height=7, width=7), conv_dims=[], fc_dims=[1024, 1024]
           ),
           box_predictor=FastRCNNOutputLayers(
               ShapeSpec(channels=1024),
               test_score_thresh=0.05,
               box2box_transform=Box2BoxTransform((10, 10, 5, 5)),
               num_classes=80,
           ),
           mask_in_features=["p2", "p3", "p4", "p5"],
           mask_pooler=ROIPooler(14, (1.0 / 4, 1.0 / 8, 1.0 / 16, 1.0 / 32), 0, "ROIAlignV2"),
           mask_head=MaskRCNNConvUpsampleHead(
               ShapeSpec(channels=256, width=14, height=14),
               num_classes=80,
               conv_dims=[256, 256, 256, 256, 256],
           ),
       ),
       pixel_mean=[103.530, 116.280, 123.675],
       pixel_std=[1.0, 1.0, 1.0],
       input_format="BGR",
   )
   ```

  </details>


If you only need the standard behavior, the [Beginner's Tutorial](./getting_started.md)
should suffice. If you need to extend detectron2 to your own needs,
see the following tutorials for more details:

* Detectron2 includes a few standard datasets. To use custom ones, see
  [Use Custom Datasets](./datasets.md).
* Detectron2 contains the standard logic that creates a data loader for training/testing from a
  dataset, but you can write your own as well. See [Use Custom Data Loaders](./data_loading.md).
* Detectron2 implements many standard detection models, and provide ways for you
  to overwrite their behaviors. See [Use Models](./models.md) and [Write Models](./write-models.md).
* Detectron2 provides a default training loop that is good for common training tasks.
  You can customize it with hooks, or write your own loop instead. See [training](./training.md).
