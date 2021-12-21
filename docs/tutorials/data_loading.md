
# 데이터로더

데이터로더란 모델에 데이터를 전달하는 컴포넌트입니다.
일반적으로 데이터로더는 [데이터셋](./datasets.md) 에서 원본 데이터를 불러와
모델에서 필요로 하는 형식으로 처리합니다.

## 기본 데이터로더의 작동 방식

Detectron2에는 데이터 로딩 파이프라인이 기본적으로 내장되어 있습니다.
커스텀 데이터로더를 작성할 경우를 대비해 작동 방식을 이해하면 좋습니다.

Detectron2는 두 가지 기능을 제공합니다.
[build_detection_{train,test}_loader](../modules/data.html#detectron2.data.build_detection_train_loader)
는 입력된 설정에 맞춰 기본 데이터 로더를 생성합니다.
아래는 `build_detection_{train,test}_loader` 가 동작하는 방식입니다.

1. 등록되어 있는 데이터셋 이름(e.g. "coco_2017_train")을 입력받아 `list[dict]` 에 가벼운 형태로 
   데이터셋 아이템들을 로드합니다. 이 데이터셋 아이템들은 모델에서 바로 사용할 수는 
   없습니다 (e.g. 이미지가 메모리에 로드되지 않았거나 랜덤 증강이 적용되지 않은 등).
   데이터셋 형태 및 데이터셋 등록에 대한 자세한 내용은 [datasets](./datasets.md) 에서
   확인하십시오.
2. 이 list의 각 dict는 함수("mapper")에 의해 매핑됩니다.
   * 사용자는 `build_detection_{train,test}_loader` 의 "mapper" argument를 직접 지정함으로써 
         이 매핑 함수를 커스텀 정의할 수 있습니다. 기본 매퍼는 [DatasetMapper](../modules/data.html#detectron2.data.DatasetMapper) 입니다.
   * mapper의 출력은 이 데이터로더의 consumer(모델 등)와 호환이 되는 한, 임의의 형태일 수 있습니다.
     batch 적용 후 기본 mapper의 출력은 [모델 사용](./models.html#model-input-format) 에 설명된
     기본 모델 입력 형태를 따릅니다.
   * mapper의 역할은 데이터셋의 아이템들을
     모델이 사용할 수 있는 형태로 변환하는 것입니다 (e.g. 이미지 읽기, 랜덤 데이터 증강, torch Tensor 변환 작업 등을 포함).
     데이터에 커스텀 변환을 수행하려면 보통 커스텀 mapper가 필요합니다.
3. mapper의 출력은 batch 단위로 묶인 list 형태입니다.
4. 이 batch 단위 데이터가 데이터 로더의 출력이며, 일반적으로 `model.forward()` 의
   입력이기도 합니다.


## 커스텀 데이터로더 작성

커스텀 데이터로더를 사용하는 경우, 보통 `build_detection_{train,test}_loader(mapper=)` 에 다른 "mapper"를
사용하면 작동합니다.
예를 들어 학습을 위해 모든 이미지의 크기를 특정 크기로 변경(resize)하려면 다음 코드를 사용합니다.

```python
import detectron2.data.transforms as T
from detectron2.data import DatasetMapper   # 기본 mapper
dataloader = build_detection_train_loader(cfg,
   mapper=DatasetMapper(cfg, is_train=True, augmentations=[
      T.Resize((800, 800))
   ]))
# 기본 데이터로더 대신 이것을 사용하십시오
```
필요한 arguments가 기본 [DatasetMapper](../modules/data.html#detectron2.data.DatasetMapper) 에
없는 경우, 커스텁 mapper 함수를 작성해 대신 사용하면 됩니다. e.g.

```python
from detectron2.data import detection_utils as utils
 # 기본 DatasetMapper와 유사한 간단한 mapper를 구현하는 방법입니다
def mapper(dataset_dict):
    dataset_dict = copy.deepcopy(dataset_dict)  # 이 값은 아래 코드에 의해 수정됩니다
    # 다른 방법을 사용해 이미지를 읽을 수도 있습니다
    image = utils.read_image(dataset_dict["file_name"], format="BGR")
    # 자세한 사용법은 "데이터 증강" 튜토리얼을 참조하십시오
    auginput = T.AugInput(image)
    transform = T.Resize((800, 800))(auginput)
    image = torch.from_numpy(auginput.image.transpose(2, 0, 1))
    annos = [
        utils.transform_instance_annotations(annotation, [transform], image.shape[1:])
        for annotation in dataset_dict.pop("annotations")
    ]
    return {
       # 모델이 기대하는 형태를 만듭니다
       "image": image,
       "instances": utils.annotations_to_instances(annos, image.shape[1:])
    }
dataloader = build_detection_train_loader(cfg, mapper=mapper)
```

mapper 이외의 다른 부분(e.g. 다른 샘플링 또는 일괄 처리 로직을 구현하기 위해)까지 변경하는 경우에는
`build_detection_train_loader` 가 동작하지 않으므로 직접 데이터로더를 작성해야 합니다.
데이터 로더는 단순히
모델에서 허용되는 [형태](./models.md) 를 생성하는 python iterator입니다.
원하는 도구 어느것이든 사용하여 구현할 수 있습니다.

무엇을 구현하려든 간에
[Detectron2.data의 API 문서](../modules/data) 에서 이 함수들의 API에 대해 자세히 알아볼 것을
권장드립니다.

## 커스텀 데이터로더 사용

[DefaultTrainer](../modules/engine.html#detectron2.engine.defaults.DefaultTrainer) 를 사용하는 경우,
`build_{train,test}_loader` 메서드를 덮어써서 커스텀 데이터로더를 사용할 수 있습니다.
[deeplab 데이터로더](../../projects/DeepLab/train_net.py) 의 예시를
참조하십시오.

직접 학습 루프를 작성한다면 데이터로더와 쉽게 연동할 수 있습니다.
