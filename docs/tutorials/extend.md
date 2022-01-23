# 기본 Detectron2 확장하기

__연구의 핵심은 새로운 방식으로 일을 해보는 것입니다__.
위 문장과 관련해 코드 추상화에 대한 상충되는 두 입장이 있을 수 있는데,
이는 모든 대규모 연구 개발 프로젝트에서 도전적인 과제입니다.

1. 한편으로, 모든 일에 새로운 방식을 허용하려면 추상화를 
    최소화할 필요성이 있습니다. 기존의 추상화를 깨고 
    새것으로 대체하는 것이 충분히 쉬워야 합니다.

2. 다른 한편으로, 사용자들이 표준 방식으로 쉽게 작업을 수행하러면
   적당히 높은 수준의 추상화가 필요하기도 합니다. 그러면 소수 
   연구자들만 관심 갖는 내용에 대해 크게 걱정하지 않아도 될 것입니다.

detectron2에는 양쪽 입장을 충족하기 위한 두 종류의 인터페이스가 있습니다.

1. yaml 파일에서 생성된 환경설정 (`cfg`) argument를
   사용하는 함수 및 클래스
   (때로는 다른 argument가 거의 없기도 합니다).

   이러한 함수와 클래스는 
   "표준 기본값 (standard default)" 동작을 구현합니다. 즉, 주어진 환경설정에서
   필요한 것을 읽고 "표준" 작업을 수행합니다.
   이때 사용되는 argument가 무엇이며 어떤 의미를 갖는지 걱정하지 않고 이미 잘 만들어진 환경설정을
   불러와서 전달하기만 하면 됩니다.

   자세한 튜토리얼은 [Yacs Configs](configs.md) 에서 확인하십시오.

2. argument가 명시적으로 잘 정의된 함수 및 클래스.

   이들 각각은 전체 시스템의 작은 구성 요소입니다.
   각 argument가 무엇인지 이해하려면 사용자의 전문 지식이 필요하며,
   이들을 조립해 더 큰 시스템을 만들려면 더 많은 노력이 필요합니다.
   그러나 이들을 더 유연하게 조립할 방법은 존재합니다.

   잘 정의된 각 구성 요소들은 detectron2의 "표준 기본값"이 
   지원하지 않는 무언가를 구현해야 할 때 재사용될 수 있습니다.
   
   [LazyConfig 시스템](lazyconfigs.md) 은 이러한 함수와 클래스를 잘 활용한 사례입니다.

3. A few functions and classes are implemented with the
   [@configurable](../modules/config.html#detectron2.config.configurable)
   decorator - they can be called with either a config, or with explicit arguments, or a mixture of both.
   Their explicit argument interfaces are currently experimental.

   예를 들어 아래와 같은 방법들로 Mask R-CNN 모델을 만들 수 있습니다.

   1. 환경설정만으로:
      ```python
      # load proper yaml config file, then
      model = build_model(cfg)
      ```

   2. 환경설정에 더해 일부 추가 argument를 덮어써서:
      ```python
      model = GeneralizedRCNN(
        cfg,
        roi_heads=StandardROIHeads(cfg, batch_size_per_image=666),
        pixel_std=[57.0, 57.0, 57.0])
      ```

   3. 모든 argument를 명시해서:
   <details>
   <summary>
   (클릭하여 펼치기)
   </summary>

   ```python
   model = GeneralizedRCNN(
       backbone=FPN(
           ResNet(
               BasicStem(3, 64, norm="FrozenBN"),
               ResNet.make_default_stages(50, stride_in_1x1=True, norm="FrozenBN"),
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


표준 동작만 필요한 경우 [초보자용 튜토리얼](./getting_started.md) 으로
충분할 것입니다. 자신의 필요에 맞게 detectron2를 확장해야 한다면,
다음 튜토리얼에서 자세한 내용을 참조하십시오.

* Detectron2는 몇 가지 표준 데이터셋을 제공합니다. 커스텀 데이터셋을 사용하려면 [커스텀 데이터셋 사용](./datasets.md)을 참조하십시오.
* Detectron2는 데이터셋에서 학습/테스트를 위한 데이터로더를 생성하는 표준 로직을 제공하지만,
   직접 작성할 수도 있습니다. [커스텀 데이터로더 사용](./data_loading.md) 을 참조하십시오.
* Detectron2는 표준적인 검출 (detection) 모델 구현체 다수와 그 동작을
  덮어쓸 방법을 제공합니다. [모델 사용](./models.md) 및 [모델 작성](./write-models.md) 을 참조하십시오.
* Detectron2는 일반적인 모델 학습을 위한 기본 학습 루프를 제공합니다.
   이를 hook으로 커스터마이징하거나 직접 루프를 작성할 수도 있습니다. [학습](./training.md) 을 참조하십시오.
