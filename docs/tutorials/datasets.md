# 커스텀 데이터셋 사용

이 문서는 데이터셋 API([DatasetCatalog](../modules/data.html#detectron2.data.DatasetCatalog),
[MetadataCatalog](../modules/data.html#detectron2.data.MetadataCatalog)) 의 동작 방식과 이들을 사용해
커스텀 데이터셋을 추가하는 방법을 설명합니다.

detectron2에서 내장되어 지원되는 데이터셋은 [내장 데이터셋](builtin_datasets.md) 에 있습니다.
커스텀 데이터셋을 detectron2의 기본 데이터로더와 함께 사용하려는 경우,
다음을 수행해야 합니다.

1. 데이터셋을 __등록하십시오__ (즉, detectron2가 데이터셋에 접근할 수 있도록 합니다).
2. 필요하다면 데이터셋에 대한 __메타데이터를 등록하십시오__.

이제 이 두 가지 개념에 대해 자세히 살펴보겠습니다.

[Colab 튜토리얼](https://colab.research.google.com/drive/16jcaJoc6bCFAQ96jDe2HwtXj7BMD_-m5)에
커스텁 형식의 데이터셋을 어떻게 등록하고 학습하는지 설명하는 라이브 예시가 있습니다.

### 데이터셋 등록

detectron2이 "my_dataset"이라는 데이터셋에 접근할 수 있도록 사용자는
데이터셋의 아이템을 반환하는 함수를 구현하고 detectron2에게 이 함수를 전달해야
합니다.
```python
def my_dataset_function():
  ...
  return list[dict] in the following format

from detectron2.data import DatasetCatalog
DatasetCatalog.register("my_dataset", my_dataset_function)
# 이후 아래와 같이 데이터에 접근합니다.
data: List[Dict] = DatasetCatalog.get("my_dataset")
```

위 코드는 "my_dataset"이라는 데이터셋을 데이터를 반환하는 함수와 연결합니다.
이 함수는 여러 번 호출되는 경우 동일한 데이터(같은 순서로)를 반환해야 합니다.
등록은 프로세스가 종료될 때까지 유효합니다.

이 함수는 임의의 작업을 수행한 후 `list[dict]` 안에 다음 중 한 가지 포맷으로
데이터를 반환해야 합니다.
1. 아래 설명된 detectron2의 표준 데이터셋 dict 포맷. 이 경우, detectron2의 다른 많은 내장 기능을
   함께 사용할 수 있으므로, 이것으로 충세그멘테이션 때 사용할 것을 권장합니다.
2. 임의의 커스텀 포맷. 새로운 task를 위해 기존에 없는 키(key)를 추가하는 등,
   여러분만의 포맷으로 임의의 dict를 반환할 수도 있습니다.
   이후 다운스트림(downstream)에서도 이를 적절하게 처리해야 합니다.
   자세한 내용은 아래를 참조하십시오.

#### 표준 데이터셋 dict

표준 task인
객체 검출(instance detection), 객체/시맨틱/팬옵틱 세그멘테이션(instance/semantic/panoptic segmentation), 키포인트 검출(keypoint detection)을 수행하는 경우,
COCO의 어노테이션(annotation)과 유사한 스펙으로 원본 데이터셋을 `list[dict]` 에 로드합니다.
이것이 데이터셋 형태에 대한 우리의 표준입니다.

각 dict에는 하나의 이미지에 대한 정보가 들어 있습니다.
dict에는 아래와 같은 필드가 있을 수 있으며
필수 필드는 데이터로더 또는 task에 필요한 항목에 따라 다릅니다(아래 내용 참조).

```eval_rst
.. list-table::
  :header-rows: 1

  * - Task
    - 필드
  * - 공통
    - file_name, height, width, image_id

  * - 객체 검출/세그멘테이션
    - annotations

  * - 시맨틱 세그멘테이션
    - sem_seg_file_name

  * - 팬옵틱 세그멘테이션
    - pan_seg_file_name, segments_info
```

+ `file_name`: 이미지 파일이 위치한 전체 경로입니다.
+ `height`, `width`: 정수형으로 표현된 이미지의 shape입니다.
+ `image_id` (str or int): 이미지의 고유 식별자입니다. 주로 evaluator가
  이미지를 식별할 때 필요하지만, 데이터셋은 이를 다른 목적으로 사용할 수 있습니다.
+ `annotations` (list[dict]): __객체 검출/세그멘테이션 또는 키포인트 검출__ task에 필요합니다.
   각 dict는 이 이미지에 있는 한 객체의 annotation에 해당하며,
   다음과 같은 키를 포함할 수 있습니다.
  + `bbox` (list[float], required): 객체의 바운딩 박스(bounding box)를 나타내는 4개의 숫자 목록입니다.
  + `bbox_mode` (int, required): bbox의 포맷으로,
    [structures.BoxMode](../modules/structures.html#detectron2.structures.BoxMode) 중 하나여야 합니다.
    현재 지원되는 포맷은 `BoxMode.XYXY_ABS`, `BoxMode.XYWH_ABS` 입니다.
  + `category_id` (int, required): 범주형 레이블(category label)을 나타내는 [0, num_categories-1] 범위의 정수입니다.
    num_categories 값은 "배경(background)" 범주가 있는 경우를 위해 예약되어 있습니다.
  + `segmentation` (list[list[float]] or dict): 객체 세그멘테이션 마스크입니다.
    + 값이 `list[list[float]]` 라면 폴리곤(각 connected component 별로 생성됨)의 list를
      의미합니다. 각각의 `list[float]` 는 단순한 `[x1, y1, ..., xn, yn]` (n≥3) 포맷의 폴리곤입니다.
      X와 Y는 픽셀 단위로 표현된 절대 좌표입니다.
    + 값이 `dict` 라면 COCO의 압축 RLE 포맷으로 픽셀별 세그멘테이션 마스크를 나타냅니다.
       dict에는 "size"와 "counts" 키가 있어야 합니다. `pycocotools.mask.encode(np.asarray(mask, order="F"))` 를
       사용해 0과 1로 구성된 uint8 세그멘테이션 마스크를 이러한 dict로 변환할 수 있습니다.
       이러한 포맷을 기본 데이터로더와 함께 사용하려면 `cfg.INPUT.MASK_FORMAT` 을 `bitmask` 로 설정해야 합니다.
  + `keypoints` (list[float]): [x1, y1, v1,..., xn, yn, vn]의 포맷을 갖습니다.
    v[i]는 해당 인덱스의 키포인트의 [가시성](http://cocodataset.org/#format-data) 을 나타냅니다.
    `n` 은 키포인트 범주의 개수와 동일해야 합니다.
    X와 Y는 [0, W 또는 H] 범위의 실수값을 갖는 절대 좌표입니다.

    (COCO의 키포인트 좌표의 포맷은 표준 포맷과 달리 [0, W-1 또는 H-1] 범위의
    정수형입니다. Detectron2는 COCO 이산 픽셀 인덱스를 부동 소수점 좌표로
    변환하기 위해 키포인트 좌표에 0.5를 더합니다.)
  + `iscrowd`: 0(기본값) 또는 1로 이 객체가 COCO의 "crowd region"으로 레이블링되었는지 여부를
    나타냅니다. 이 필드의 의미를 모르면 포함하지 마십시오.

  `annotations` 가 빈(empty) list라는 것은 이미지에 레이블링된 객체가 없다는 의미입니다.
  이러한 이미지는 기본적으로 학습에서 제외되지만
  `DATALOADER.FILTER_EMPTY_ANNOTATIONS` 를 통해 포함시킬 수 있습니다.

+ `sem_seg_file_name` (str):
  시맨틱 세그멘테이션의 ground truth 파일이 위치한 전체 경로입니다.
  픽셀 값이 정수형 레이블인 회색조(grayscale) 이미지여야 합니다.
+ `pan_seg_file_name` (str):
  팬옵틱 세그멘테이션의 ground truth 파일이 위치한 전체 경로입니다.
  [panopticapi.utils.id2rgb](https://github.com/cocodataset/panopticapi/) 함수를 사용해
  픽셀 값을 정수형 id로 인코딩한 RGB 이미지여야 합니다.
  id는 `segments_info` 에 의해 정의됩니다.
  `segments_info` 에 id가 없는 픽셀은 레이블이 없는 것으로 간주되며
  일반적으로 학습 및 평가에서 무시됩니다.
+ `segments_info` (list[dict]): 팬옵틱 세그멘테이션의 ground truth에서 각 id의 의미를 정의합니다.
  각 dict에는 다음과 같은 키가 있습니다.
  + `id` (int): ground truth 이미지에 나타나는 정수입니다.
  + `category_id` (int): 범주형 레이블을 나타내는 [0, num_categories-1] 범위의 정수입니다.
  + `iscrowd`: 0(기본값) 또는 1로 이 객체가 COCO의 "crowd region"으로 레이블링되었는지 여부를 나타냅니다.


```eval_rst

.. note::

   PanopticFPN 모델은 여기에 정의된 팬옵틱 세그멘테이션 포맷 대신에
   객체 세그멘테이션 및 시맨틱 세그멘테이션 데이터의 포맷을 조합하여 사용합니다.
   COCO에 대한 지침은 :doc:`builtin_datasets` 를 참조하십시오.

```

(proposals이 사전에 계산된) Fast R-CNN 모델은 요즘 거의 사용되지 않습니다.
Fast R-CNN을 훈련하려면 다음과 같은 키가 추가적으로 필요합니다.

+ `proposal_boxes` (array): (K, 4) shape의 2D numpy 배열로 이 이미지에 대해 사전 계산된 K개의 proposal 박스를 의미합니다.
+ `proposal_objectness_logits` (array): 'proposal_boxes'에 있는 proposal의 objectness logit에 해당하는
  (K, ) shape의 numpy 배열입니다.
+ `proposal_bbox_mode` (int): 사전 계산된 proposal bbox의 포맷입니다.
 [structures.BoxMode](../modules/structures.html#detectron2.structures.BoxMode) 중에
 하나여야 합니다.
 기본값은 `BoxMode.XYXY_ABS` 입니다.



#### 새로운 task에 대한 커스텀 데이터셋 dict

데이터셋 함수가 반환하는 `list[dict]` 에서 dict는 __임의의 커스텀 데이터__ 일 수 있습니다.
이것은 표준 데이터셋 dict에서 다루지 않는 추가 정보가 필요한
새로운 task에 유용합니다. 이 경우 다운스트림 코드가 데이터를 올바르게 처리할 수 있는지 확인해야
합니다. 일반적으로 데이터로더에 대한 새로운 `mapper` 를 작성해야 합니다([커스텀 데이터로더 사용](./data_loading.md) 참조).

커스텀 포맷을 설계할 때 모든 dict는 (때로는 직렬화되어,
여러 사본과 함께) 메모리에 저장됩니다.
메모리를 절약하기 위해, 각각의 dict에는 샘플 별로 파일명이나 어노테이션과 같은
__작지만__ 충분한 정보가 포함됩니다.
전체 샘플을 로드하는 경우는 보통 데이터로더에서 발생합니다.

데이터셋 전체적으로 공유되는 속성(attribute)은 `메타데이터` 를 사용해 저장하십시오 (아래 참조).
메모리 낭비를 방지하려면 샘플 별로 반복되는 정보를 중복하여 저장하지 마십시오.

### 데이터셋의 "메타데이터"

각 데이터셋은 연관된 메타데이터가 있으며
`MetadataCatalog.get(dataset_name).some_metadata` 를 통해 그것에 접근할 수 있습니다.
메타데이터는 전체 데이터셋 간에 공유되는 정보를 포함하는
키-밸류 매핑이며, 일반적으로 데이터셋에 있는 내용(예: 클래스 이름,
클래스 색상, 파일 루트 등)을 해석하는 데에 사용됩니다.
이 정보는 증강, 평가, 시각화, 로그 생성 등에 유용하게 쓰입니다.
메타데이터의 구조는 다운스트림 코드에서 무엇이 필요한지에 따라 다릅니다.

'DatasetCatalog.register'를 통해 새로운 데이터셋을 등록할 경우,
`MetadataCatalog.get(dataset_name).some_key = some_value` 를 통해
해당 메타데이터를 추가해 메타데이터가 필요한 기능을 활성화할 수도 있습니다.
예를들어 메타데이터 키 "thing_classes"에 대해 다음과 같이 사용할 수 있습니다.

```python
from detectron2.data import MetadataCatalog
MetadataCatalog.get("my_dataset").thing_classes = ["person", "dog"]
```

다음은 detectron2의 내장 기능에서 사용되는 메타데이터 키 목록입니다.
이러한 메타데이터 없이 새로운 데이터셋을 추가하면 일부 기능을
사용하지 못할 수도 있습니다.

* `thing_classes` (list[str]): 모든 객체 검출/세그멘테이션 task에서 사용됩니다.
  객체 또는 thing의 분류명 목록입니다.
  COCO 포맷의 데이터셋을 로드하면 `load_coco_json` 함수에 의해 자동으로 설정됩니다.

* `thing_colors` (list[tuple(r, g, b)]): 각 thing 분류에 대해 지정하는 색상([0, 255])입니다.
  시각화에 사용되며, 지정하지 않으면 임의의 색상으로 설정됩니다.

* `stuff_classes` (list[str]): 시맨틱 및 팬옵틱 세그멘테이션 task에서 사용됩니다.
  stuff의 분류명 목록입니다.

* `stuff_colors` (list[tuple(r, g, b)]): 각 stuff 분류에 대해 지정하는 색상([0, 255])입니다.
  시각화에 사용되며, 지정하지 않으면 임의의 색상으로 설정됩니다.

* `ignore_label` (int): 시맨틱 및 팬옵틱 세그멘테이션 task에서 사용됩니다. 이 범주형 레이블에 속하는
  ground-truth 어노테이션의 픽셀들은 평가 단계에서 무시됩니다. 일반적으로
  "레이블링 되지 않은" 픽셀입니다.

* `keypoint_names` (list[str]): 키포인트 검출에서 사용됩니다. 각 키포인트의 이름 목록입니다.

* `keypoint_flip_map` (list[tuple[str]]): 키포인트 검출에서 사용됩니다. 키포인트 이름 쌍(pair)의 목록입니다.
  여기서 각 쌍은 증강 과정에 이미지가 수평으로 뒤집히면서
  위치가 뒤바뀐 두 개의 키포인트입니다.
* `keypoint_connection_rules`: list[tuple(str, str, (r, g, b))]. 각 튜플은 연결되어 있는 키포인트 쌍 및
  시각화 단계에서 해당 연결선의 색상([0, 255])을 지정합니다.

특정 데이터셋(예: COCO)은 평가와 관련된 다음과 같은 추가적인 메타데이터가 있습니다.

* `thing_dataset_id_to_contiguous_id` (dict[int->int]): COCO 포맷의 모든 객체 검출/세그멘테이션 task에서 사용됩니다.
  데이터셋의 객체 클래스 id에서 [0, #class) 범위의 연속 id로의 매핑입니다.
  `load_coco_json` 함수에 의해 자동으로 설정됩니다.

* `stuff_dataset_id_to_contiguous_id` (dict[int->int]): 시맨틱/팬옵틱 세그멘테이션의 예측값을 json 파일로 생성할 때 사용됩니다.
  데이터셋의 시맨틱 세그멘테이션 클래스 id에서 [0, num_categories]의
  연속 id로의 매핑입니다. 평가 단계에서만 사용됩니다.

* `json_file`: COCO 어노테이션 json 파일입니다. COCO 포맷 데이터셋에 대한 COCO 방식 평가에서 사용됩니다.
* `panoptic_root`, `panoptic_json`: COCO 포맷의 팬옵틱 평가에서 사용됩니다.
* `evaluator_type`: 내장된 기본 학습 스크립트에서 평가자를 선택할 떄
   사용됩니다. 직접 작성한 학습 스크립트에서는 사용하지 마십시오.
   기본 스크립트에서 데이터셋의 [DatasetEvaluator](../modules/evaluation.html#detectron2.evaluation.DatasetEvaluator) 를
   직접 제공하면 됩니다.

```eval_rst
.. note::

   인식(recognition)을 설명할 때 객체 수준 세그멘테이션 task에 대해서는 "thing"이라는 용어를,
   시맨틱 세그멘테이션 task에 대해서는 "stuff"이라는 용어를 사용하기도 합니다.
   팬옵틱 세그멘테이션 task에서는 둘 다 사용됩니다.
   "thing"과 "stuff"의 개념에 대한 배경 설명은
   `On Seeing Stuff: The Perception of Materials by Humans and Machines
   <http://persci.mit.edu/pub_pdfs/adelson_spie_01.pdf>`_ 를 참조하십시오.
```

### COCO 포맷 데이터셋 등록

이미 COCO 포맷의 json 파일로 된 객체 수준(검출, 세그멘테이션, 키포인트) 데이터셋의 경우,
아래와 같이 데이터셋 및 관련 메타데이터를 쉽게 등록할 수 있습니다.
```python
from detectron2.data.datasets import register_coco_instances
register_coco_instances("my_dataset", {}, "json_annotation.json", "path/to/image/dir")
```

데이터셋이 COCO 포맷이지만 추가적인 처리가 필요하거나 객체 단위로 커스텀 어노테이션을 추가해야 하는 경우,
[load_coco_json](../modules/data.html#detectron2.data.datasets.load_coco_json)
함수가 유용하게 쓰일 수 있습니다.

### 새로운 데이터셋을 위한 설정 변경

데이터셋을 등록하면 `cfg.DATASETS.{TRAIN,TEST}` 에서 데이터셋의
이름(예: 위 예시의 "my_dataset")을 사용할 수 있습니다.
이외에 새로운 데이터셋을 학습하거나 평가하기 위해 아래 설정을 변경해야 할 수 있습니다.

* `MODEL.ROI_HEADS.NUM_CLASSES` 와 `MODEL.RETINANET.NUM_CLASSES` 는 각각
  R-CNN 및 RetinaNet 모델의 thing 클래스 수입니다.
* `MODEL.ROI_KEYPOINT_HEAD.NUM_KEYPOINTS` 는 Keypoint R-CNN의 키포인트 수를 설정합니다.
  또한 평가를 위해 `TEST.KEYPOINT_OKS_SIGMAS`로
  [Keypoint OKS](http://cocodataset.org/#keypoints-eval)를 설정해야 합니다.
* `MODEL.SEM_SEG_HEAD.NUM_CLASSES` 는 Semantic FPN 및 Panoptic FPN의 stuff 클래스 수를 설정합니다.
* `TEST.DETECTIONS_PER_IMAGE` 는 검출할 최대 객체 수를 제어합니다.
  테스트 이미지가 100개 이상의 객체를 포함할 수 있는 경우 이 값을 더 크게 설정하십시오.
* (proposals이 사전에 계산된) Fast R-CNN을 학습하는 경우 `DATASETS.PROPOSAL_FILES_{TRAIN,TEST}` 가
  데이터셋과 일치해야 합니다. proposal 파일의 포맷은 [여기](../modules/data.html#detectron2.data.load_proposals_into_dataset)
  문서화되어 있습니다.

새로운 모델들(예:
[TensorMask](../../projects/TensorMask),
[PointRend](../../projects/PointRend))도
종종 마찬가지로 고유 설정을 변경해야 할 수 있습니다.

```eval_rst
.. tip::

   클래스 수를 변경하면 사전 학습된 모델의 몇몇 계층(layer)은 호환되지 않으므로
   새로운 모델에 로드할 수 없습니다.
   이는 의도된 것으로, 이와 같이 사전 학습된 모델을 로드하면 해당 계층에 대해 경고를 출력합니다.
```
