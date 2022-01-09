# 모델 사용

## Yacs 설정으로부터 모델 빌드
yacs 설정 객체로부터
모델(및 서브모델)을 빌드하기 위해서는
`build_model`, `build_backbone`, `build_roi_heads` 와 같은 함수를 사용하면 됩니다.
```python
from detectron2.modeling import build_model
model = build_model(cfg)  # torch.nn.Module을 반환
```

`build_model` 함수는 모델 구조만을 빌드해 임의의 인자(parameter)로 채웁니다.
미리 저장해둔 체크포인트를 모델에 로드하거나 `model` 객체를 사용하는 방법은 아래를 참조하십시오.

### 체크포인트 로드/저장
```python
from detectron2.checkpoint import DetectionCheckpointer
DetectionCheckpointer(model).load(file_path_or_url)  # 파일 로드 (보통 cfg.MODEL.WEIGHTS에서)

checkpointer = DetectionCheckpointer(model, save_dir="output")
checkpointer.save("model_999")  # output/model_999.pth에 저장
```

Detectron2의 checkpointer는 pytorch의 `.pth` 포맷 모델뿐만 아니라
모델 zoo의 `.pkl` 파일을 인식합니다.
그 사용법에 관한 자세한 내용은 [API 문서](../modules/checkpoint.html#detectron2.checkpoint.DetectionCheckpointer)
에서 확인하십시오.

모델 파일은 임의로 조작할 수 있는데 `.pth` 파일의 경우 `torch.{load,save}` 를, `.pkl` 파일의 경우
`pickle.{dump,load}` 를 사용하면 됩니다.

### 모델 사용

모델은 `outputs = model(inputs)` 와 같이 호출할 수 있으며 여기서 `inputs` 는 `list[dict]` 입니다.
각 dict는 하나의 이미지에 해당하며 이때 필요한 키는
모델의 유형에 따라, 그리고 학습 모드인지 평가 모드인지에 따라 다릅니다.
예를 들어, 추론을 수행하기 위해
모든 모델에서 "image" 키가 필수로 필요하고 "height" 및 "width" 키가 선택적으로 필요합니다.
모델의 입력 및 출력에 대한 자세한 포맷은 아래에 설명되어 있습니다.

__학습__: 학습 모드에서, `EventStorage` 아래에서 모델을 사용해야 합니다.
학습 과정에 대한 통계는 스토리지에 저장됩니다.
```python
from detectron2.utils.events import EventStorage
with EventStorage() as storage:
  losses = model(inputs)
```

__추론__: 미리 준비된 모델을 이용하여 단순히 추론만을 하고 싶다면,
[DefaultPredictor](../modules/engine.html#detectron2.engine.defaults.DefaultPredictor)
라는 모델 래퍼(wrapper)가 이러한 기본 기능을 제공합니다.
이것은 모델 로드, 전처리를 포함한 기본 동작을 지원하며
배치(batch)가 아닌 단일 이미지에 대해 동작합니다. 사용법은 링크의 문서를 참조하십시오.

다음과 같이 직접 추론을 실행할 수도 있습니다.
```python
model.eval()
with torch.no_grad():
  outputs = model(inputs)
```

### 모델 입력 포맷

임의의 입력 포맷을 지원하는 커스텀 모델을 구현할 수 있습니다.
여기에서는 detectron2의 모든 내장(builtin) 모델이 지원하는 표준 입력 포맷을 설명합니다.
내장 모델들 모두 `list[dict]` 를 입력으로 받습니다. 각 dict는
하나의 이미지에 대한 정보를 담고 있습니다.

dict에는 다음 키가 포함될 수 있습니다.

* "image": (C, H, W) 포맷으로 표현된 `Tensor`. 각 채널의 의미는 `cfg.INPUT.FORMAT` 에서 정의합니다.
  이미지 정규화가 있는 경우, `cfg.MODEL.PIXEL_{MEAN,STD}` 를 통해
  모델 내에서 수행됩니다.
* "height", "width": **추론** 에서 **요구되는** 출력 높이 및 너비로, `image` 필드의
  높이나 너비와 같지 않을 수 있습니다.
  예를 들어 전처리 단계에 크기 조정(resize)이 사용되면 `image` 필드는 크기가 조정된 이미지를 담고 있을 것입니다.
  그러나 출력을 **원래** 해상도로 하고 싶을 수 있습니다.
  이럴 때, "height", "width"를 제공하면 모델은 입력된 `image` 의 해상도 대신
  제공된 해상도로 출력을 생성합니다. 이것이 더 효율적이고 정확합니다.
* "instances": 학습을 위한 [Instances](../modules/structures.html#detectron2.structures.Instances)
  객체. 다음 필드들을 갖고 있습니다.
  + "gt_boxes": [Boxes](../modules/structures.html#detectron2.structures.Boxes) 객체. 각 instance에 대응되는 N개의 box를 저장하고 있습니다.
  + "gt_classes": long 타입의 `Tensor`. [0, num_categories) 범위의 레이블(label)을 N개 저장하고 있는 벡터입니다.
  + "gt_masks": [PolygonMasks](../modules/structures.html#detectron2.structures.PolygonMasks)
    또는 [BitMasks](../modules/structures.html#detectron2.structures.BitMasks) 객체. 각 instance에 대응되는 N개의 마스크를 저장하고 있습니다.
  + "gt_keypoints": [Keypoints](../modules/structures.html#detectron2.structures.Keypoints) 객체.
    각 instance에 대응되는 N개의 키포인트 셋을 저장하고 있습니다.
* "sem_seg": (H, W) 포맷의 `Tensor[int]`. 시맨틱 세그멘테이션(semantic segmentation)의 학습을 위한 ground truth입니다.
  각각의 값은 0부터 시작하는 범주 레이블을 나타냅니다.
* "proposals": [Instances](../modules/structures.html#detectron2.structures.Instances)
  객체. Fast R-CNN 스타일 모델에서만 사용되며 다음 필드를 갖고 있습니다.
  + "proposal_boxes": [Boxes](../modules/structures.html#detectron2.structures.Boxes) 객체. P개의 proposal box를 저장하고 있습니다.
  + "objectness_logits": `Tensor`. 각 proposal에 대응되는 P개의 score로 된 벡터입니다.

내장 모델의 추론을 위해 "image" 키만 필수이며 "width/height"는 선택 사항입니다.

현재 팬옵틱 세그멘테이션(panoptic segmentation)의 경우, 모델이 커스텀 데이터로더에서 생성된
커스텀 포맷을 사용하기 때문에 학습을 위한 표준 입력 포맷을 정의하고 있지 않습니다.

#### 데이터로더와 연결하는 방법

기본 [DatasetMapper](../modules/data.html#detectron2.data.DatasetMapper)의 출력은
위 포맷을 따르는 dict입니다.
데이터로더가 배칭(batching)을 수행한 후에는 내장 모델에서 지원하는 `list[dict]` 형태가 됩니다.


### 모델 출력 포맷

내장 모델은 학습 모드에서 손실(loss) 값들과 함께 `dict[str->Scalar Tensor]` 를 출력합니다.

추론 모드에서는 `list[dict]` 를 출력하며 각 이미지에 대해 하나의 dict 니다.
모델이 수행하는 작업에 따라 각 dict에는 아래와 같은 필드가 포함될 수 있습니다.

* "instances": [Instances](../modules/structures.html#detectron2.structures.Instances)
  객체. 다음과 같은 필드를 갖고 있습니다.
  * "pred_boxes": [Boxes](../modules/structures.html#detectron2.structures.Boxes) 객체. 검출된 각각의 instance에 대한 N개의 box를 저장하고 있습니다.
  * "scores": `Tensor`. N개의 confidence score로 된 벡터입니다.
  * "pred_classes": `Tensor`. [0, num_categories) 범위의 레이블 N개로 된 벡터입니다.
  + "pred_masks": (N, H, W) shape의 `Tensor`. 검출된 각각의 instance에 대한 마스크입니다.
  + "pred_keypoints": (N, num_keypoint, 3) shape의 `Tensor`.
    마지막 차원(dimension)의 각 행은 (x, y, score)입니다. Confidence score는 양수입니다.
* "sem_seg": (num_categories, H, W) shape의 `Tensor`. 시맨틱 세그멘테이션에 대한 예측값입니다.
* "proposals": [Instances](../modules/structures.html#detectron2.structures.Instances)
  객체. 다음과 같은 필드를 갖고 있습니다.
  * "proposal_boxes": [Boxes](../modules/structures.html#detectron2.structures.Boxes)
    객체. N개의 box를 저장하고 있습니다.
  * "objectness_logits": N개의 confidence score로 된 torch 벡터입니다.
* "panoptic_seg": `(pred: Tensor, segments_info: Optional[list[dict]])` 형태의 튜플(tuple).
  `pred` 텐서는 (H, W) shape으로, 각 픽셀의 segment id를 저장하고 있습니다.

  * `segments_info` 가 있다면, 각각의 dict는 `pred` 의 segment id 하나에 대한 정보를 담고 있으며, 다음 필드를 갖고 있습니다.

    * "id": segment의 id
    * "isthing": segment가 thing인지 stuff인지 여부
    * "category_id": segment의 범주 id

    픽셀의 id가 `segments_info` 에 없으면
    [팬옵틱 세그멘테이션](https://arxiv.org/abs/1801.00868) 에 정의된 void 레이블로 간주됩니다.

  * `segments_info` 가 None이면 `pred` 의 모든 픽셀 값이 -1보다 이상이어야 합니다.
    값이 -1인 픽셀에는 void 레이블이 할당됩니다.
    그렇지 않으면 각 픽셀의 범주 id는
    `category_id = pixel // metadata.label_divisor` 로 결정됩니다.


### 모델 부분 실행

때로는 모델 실행 과정에 특정 계층의 입력, 후처리 전의 출력과 같은
중간(intermediate) 텐서를 얻고 싶을 수 있습니다.
일반적으로 중간 텐서는 수백 개가 있으므로 우리가 원하는 중간 결과를
얻기 위한 API는 따로 없습니다.
대신 다음과 같은 옵션이 있습니다.

1. 모델(혹은 서브모델)을 작성하십시오. [튜토리얼](./write-models.md) 을 따라
   기존 컴포넌트와 동일한 작업을 수행하면서 필요한 출력을
   반환하도록 모델 컴포넌트(예: 모델 헤드)를 다시 작성할 수
   있습니다.
2. 모델을 부분 실행합니다. 평소와 같이 모델을 생성하되
   `forward()` 대신 사용자 정의 코드를 사용하여 모델을 실행합니다. 예를 들어
   다음 코드는 마스크 헤드 전에 마스크 feature 값을 가져옵니다.

   ```python
   images = ImageList.from_tensors(...)  # 전처리된 입력 텐서
   model = build_model(cfg)
   model.eval()
   features = model.backbone(images.tensor)
   proposals, _ = model.proposal_generator(images, features)
   instances, _ = model.roi_heads(images, features, proposals)
   mask_features = [features[f] for f in model.roi_heads.in_features]
   mask_features = model.roi_heads.mask_pooler(mask_features, [x.pred_boxes for x in instances])
   ```

3. [포워드 훅 (forward hooks)](https://pytorch.org/tutorials/beginner/former_torchies/nnft_tutorial.html#forward-and-backward-function-hooks) 을 사용하십시오.
   특정 모듈의 입력이나 출력을 얻는 데 포워드 훅이 도움을 줄 수 있습니다.
   정확히 원하는 기능이 아니더라도, 부분 실행 등과 함께 사용하여
   다른 텐서를 얻을 수 있을 것입니다.

어떤 옵션을 사용하더라도 내부 텐서를 얻기 위한
코드를 작성하려면 내부 로직을 이해해야 하며, 이를 위해,
문서뿐만 아니라 때로는 기존 모델의 코드를 읽어봐야 합니다.
