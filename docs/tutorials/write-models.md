# 모델 작성

전혀 새로운 것을 시도할 때에는 밑바닥부터 시작해 모델을
구현하는 것이 좋습니다. 그러나 대부분의 경우 기존 모델의
일부 컴포넌트를 수정하거나 확장하는 데에만 관심이 있을 것입니다.
따라서 사용자가 표준 모델의 특정 내부 컴포넌트의
동작을 재정의할 수 있는 메커니즘을 함께 제공합니다.


## 새로운 컴포넌트 등록

"backbone feature extractor", "box head"와 같이 사용자가 자주 커스터마이징하고 싶어하는 일반적인 개념의 경우,
사용자가 환경설정 파일에서 커스텀 구현을 주입해
바로 사용할 수 있도록 등록 메커니즘을 제공합니다.

예를 들어 새 백본을 추가하려면 다음 코드를 추가하십시오.
```python
from detectron2.modeling import BACKBONE_REGISTRY, Backbone, ShapeSpec

@BACKBONE_REGISTRY.register()
class ToyBackbone(Backbone):
  def __init__(self, cfg, input_shape):
    super().__init__()
    # 여러분만의 백본을 만드십시오
    self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=16, padding=3)

  def forward(self, image):
    return {"conv1": self.conv1(image)}

  def output_shape(self):
    return {"conv1": ShapeSpec(channels=64, stride=16)}
```

이 코드는 [Backbone](../modules/modeling.html#detectron2.modeling.Backbone)
클래스의 인터페이스에 따라 새 백본을 구현합니다.
그리고 `Backbone` 의 서브클래스(subclass)가 필요한
[BACKBONE_REGISTRY](../modules/modeling.html#detectron2.modeling.BACKBONE_REGISTRY) 에 등록합니다.
이를 통해 detectron2가 클래스 이름과 그 구현체를 연결할 수 있습니다. 따라서 다음과 같은 코드를
작성할 수 있습니다.

```python
cfg = ...   # 환경설정을 읽어옵니다
cfg.MODEL.BACKBONE.NAME = 'ToyBackbone'   # 불러온 설정을 덮어씁니다
model = build_model(cfg)  # 이 라인은 앞서 정의한 `ToyBackbone` 를 불러옵니다
```

또 다른 예로 Generalized R-CNN 메타 아키텍처의 ROI 헤드에 새로운 기능을 추가하려면
새로운 [ROIHeads](../modules/modeling.html#detectron2.modeling.ROIHeads) 서브클래스를 구현해
`ROI_HEADS_REGISTRY` 에 추가할 수 있습니다.
[DensePose](../../projects/DensePose)
와 [MeshRCNN](https://github.com/facebookresearch/meshrcnn) 는
새로운 task를 수행하기 위해 새롭게 ROIHead를 구현한 두 가지 예시입니다.
다른 아키텍처를 구현한 더 많은 예시는
[projects/](../../projects/) 에 있습니다.

전체 레지스트리 목록은 [API 문서](../modules/modeling.html#model-registries) 에 있습니다.
이 레지스트리에 컴포넌트를 등록하면 모델의 다양한 부분 혹은 모델 전체를
커스터마이징할 수 있습니다.

## 명시적 argument를 통한 모델 생성

레지스트리는 환경설정 파일 속 이름들을 실제 코드와 연결하는 다리와 같으며,
사용자가 자주 교체해야 하는 몇몇 주요 컴포넌트를 다루는 것이 용도입니다.
그러나 텍스트 기반 환경설정 파일의 기능은 제한적이기며
더 깊은 수준의 커스터마이징은 코드 작성을 통해서만 가능합니다.

detectron2의 모델 컴포넌트 대부분에는 필요한 입력 argument를 명확하게 문서화한
`__init__` 인터페이스가 있습니다. 이를 커스텀 argument로 호출하면 커스텀 변형된
모델이 반환됩니다.

예를 들어 Faster R-CNN의 박스 헤드에서 __커스텀 손실 함수__ 를 사용하는 방법은 다음과 같습니다.

1. 손실은 현재 [FastRCNNOutputLayers](../modules/modeling.html#detectron2.modeling.FastRCNNOutputLayers) 에서 계산됩니다.
   `MyRCNNOutput` 이라는 이름으로 이것의 변형 또는 서브클래스를 구현해야 하며, 여기에 커스텀 손실 함수도 포함해야 합니다.
2. `StandardROIHeads`를 호출하되, 내장된 `FastRCNNOutputLayers` 대신에 `box_predictor=MyRCNNOutput()` 를 argument로 넘깁니다.
   다른 모든 argument를 그대로 유지하고자 하면 [configurable `__init__`](../modules/config.html#detectron2.config.configurable) 메커니즘을 사용하는 것이 좋습니다.

   ```python
   roi_heads = StandardROIHeads(
     cfg, backbone.output_shape(),
     box_predictor=MyRCNNOutput(...)
   )
   ```
3. (선택) 환경설정 파일을 통해 새로운 모델을 활성화하려면 등록이 필요합니다.
   ```python
   @ROI_HEADS_REGISTRY.register()
   class MyStandardROIHeads(StandardROIHeads):
     def __init__(self, cfg, input_shape):
       super().__init__(cfg, input_shape,
                        box_predictor=MyRCNNOutput(...))
   ```
