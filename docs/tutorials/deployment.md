# 배포

Python으로 작성된 모델을 배포 가능한 산출물로 만들기 위해서는 내보내기 (export) 프로세스를 거쳐야 합니다.
다음은 이와 관련된 몇 가지 기본 개념입니다:

__"내보내기 방식(Export Method)"__ 은 Python 모델이 배포 가능한 형태로 완전히 직렬화되는 방식입니다. 지원되는 방식은 다음과 같습니다.

* `tracing`: [pytorch 문서](https://pytorch.org/tutorials/beginner/Intro_to_TorchScript_tutorial.html) 를 확인하십시오.
* `scripting`: [pytorch documentation](https://pytorch.org/tutorials/beginner/Intro_to_TorchScript_tutorial.html) 를 확인하십시오.
* `caffe2_tracing`: 모델의 일부를 caffe2 연산자로 치환한 후 tracing 사용합니다.

__"포맷(Format)"__ 은 직렬화된 모델이 파일에서 설명되는 방식입니다. e.g.
TorchScript, Caffe2 protobuf, ONNX format.
__"런타임(Runtime)"__ 은 직렬화된 모델을 로드하고 실행하는 엔진입니다.
e.g. PyTorch, Caffe2, TensorFlow, onnxruntime, TensorRT 등
런타임은 종종 특정 포맷에 종속됩니다. (e.g. PyTorch에는 TorchScript 형식이 필요하고 Caffe2에는 protobuf 형식이 필요합니다.)
현재 지원되는 배포 조합과 각각의 제약 사항은 다음과 같습니다.


```eval_rst
+----------------------------+-------------+-------------+-----------------------------+
|       Export Method        |   tracing   |  scripting  |       caffe2_tracing        |
+============================+=============+=============+=============================+
| **Formats**                | TorchScript | TorchScript | Caffe2, TorchScript, ONNX   |
+----------------------------+-------------+-------------+-----------------------------+
| **Runtime**                | PyTorch     | PyTorch     | Caffe2, PyTorch             |
+----------------------------+-------------+-------------+-----------------------------+
| C++/Python inference       | ✅          | ✅          | ✅                          |
+----------------------------+-------------+-------------+-----------------------------+
| Dynamic resolution         | ✅          | ✅          | ✅                          |
+----------------------------+-------------+-------------+-----------------------------+
| Batch size requirement     | Constant    | Dynamic     | Batch inference unsupported |
+----------------------------+-------------+-------------+-----------------------------+
| Extra runtime deps         | torchvision | torchvision | Caffe2 ops (usually already |
|                            |             |             |                             |
|                            |             |             | included in PyTorch)        |
+----------------------------+-------------+-------------+-----------------------------+
| Faster/Mask/Keypoint R-CNN | ✅          | ✅          | ✅                          |
+----------------------------+-------------+-------------+-----------------------------+
| RetinaNet                  | ✅          | ✅          | ✅                          |
+----------------------------+-------------+-------------+-----------------------------+
| PointRend R-CNN            | ✅          | ❌          | ❌                          |
+----------------------------+-------------+-------------+-----------------------------+
| Cascade R-CNN              | ✅          | ❌          | ❌                          |
+----------------------------+-------------+-------------+-----------------------------+

```

`caffe2_tracing` 은 더 이상 사용이 권장되지 않습니다.
저희가 다른 포맷 및 런타임에 대한 추가 지원 계획하고 있지는 않으나, 코드 기여는 환영합니다.

## Tracing 혹은 Scripting 방식 배포

[tracing 혹은 scripting](https://pytorch.org/tutorials/beginner/Intro_to_TorchScript_tutorial.html) 를 통해 모델을 TorchScript 형식으로 내보낼 수 있습니다.
생성된 모델 파일은 detectron2에 대한 의존성 없이 Python 또는 C++에서 로드할 수 있습니다.
내보낸 모델은 종종 커스텀 연산을 위해 torchvision (또는 그 C++ 라이브러리) 의존성을 필요로 하는 경우가 많습니다.

이 기능을 사용하려면 PyTorch ≥ 1.8이 필요합니다.

### 지원 범위
메타 아키텍처 `GeneralizedRCNN` 및 `RetinaNet` 아래의 공식 모델 대부분은
tracing 및 scripting 방식에서 모두 지원됩니다.
Cascade R-CNN 및 PointRend는 현재 tracing에서 지원됩니다.
scripting 혹은 tracing이 가능하다면 사용자의 커스텀 확장자 또한 지원됩니다.

tracing 방식으로 내보낸 모델의 경우, 입력 해상도는 동적일 수 있지만 배치 크기
(입력 이미지 수)는 고정해야 합니다.
scripting은 동적 배치 크기를 지원합니다.

### 사용법

tracing 및 scripting 방식 내보내기를 위한 핵심 API는 [TracingAdapter](../modules/export.html#detectron2.export.TracingAdapter)
및 [scripting_with_instances](../modules/export.html#detectron2.export.scripting_with_instances) 입니다.
사용법은 현재 [test_export_torchscript.py](../../tests/test_export_torchscript.py) 와
(`TestScripting` 및 `TestTracing` 를 확인하십시오)
[deployment example](../../tools/deploy) 에 나와 있습니다.
이 예제들이 실행되는지 확인한 다음, 용례에 맞게 수정하십시오.
scripting 및 tracing의 제약을 해결하기 위해서는 모델 별로 사용자 노력과 기초 지식이 요구됩니다.
향후에는 이를 더 간단한 API로 감싸 진입 장벽을 낮출 계획입니다.

## Caffe2-tracing 방식 배포
저희는 내보내기 로직을 위한 [Caffe2Tracer](../modules/export.html#detectron2.export.Caffe2Tracer) 를 제공합니다.
이는 모델의 일부를 Caffe2 연산자로 대체한 후에,
모델을 Caffe2, TorchScript 또는 ONNX 포맷으로 내보냅니다.

변환된 모델은 detectron2/torchvision 의존성 없이 Python 또는 C++에서, 그리고 CPU 또는 GPU에서 실행할 수 있습니다.
런타임이 CPU 및 모바일 추론에 최적화되어 있지만 GPU 추론에는 최적화되어 있지 않습니다.

### 지원 범위

3개의 공용 메타 아키텍처 (`GeneralizedRCNN`, `RetinaNet`, `PanopticFPN`) 아래의 공식 모델 대부분은
지원됩니다. Cascade R-CNN은 지원되지 않습니다. 배치(batch) 추론은 지원되지 않습니다.

Caffe2에서 사용할 수 없는 제어 흐름이나 연산자 (e.g. deformable convolution)를 포함하지 않는 한, 이 아키텍처들 아래에 사용자에 의해 정의된 커스텀 확장 또한 지원됩니다.
예를 들어 커스텀 백본과 헤드는 기본적으로 지원되는 경우가 많습니다.

### 사용법

API 목록은 [API 문서](../modules/export) 에 있습니다.
[export_model.py](../../tools/deploy/) 는 이 API를 사용하여 일반적인 모델을 변환하는 예제입니다.
사용자 지정 모델/데이터셋 또한 예제 스크립트에 추가할 수 있습니다.

### C++/Python에서 모델 사용하기

모델을 C++로 로드해 Caffe2 또는 Pytorch 런타임을 통해 배포할 수 있습니다. 참고를 위해 Mask R-CNN에 대한 [C++ 예시](../../tools/deploy/) 를 제공합니다.

* `caffe2_tracing` 방식으로 내보낸 모델은
  [문서](../modules/export.html#detectron2.export.Caffe2Tracer)에 설명된 특수 입력 포맷을 사용합니다. 위의 C++ 예제에서 이를 다룹니다.

* 변환된 모델은 레이어의 출력값을 포매팅된 예측값으로
  변환하는 후처리 작업을 포함하고 있지 않습니다.
  예를 들어, 위의 C++ 예제는 최종 레이어의 출력값(28x28 마스크)에 대해 어떠한 후처리도 적용하지 않습니다.
  이는 실제 배포 시 애플리케이션이 일반적으로
  자체적인 후처리 과정을 필요로 하기 때문이며, 이에 따라 해당 단계는 사용자의 역할로 맡겨집니다.

Python에서 Caffe2 포맷의 모델 사용을 돕기 위해,
변환된 모델에 대한 python wrapper인
[Caffe2Model.\_\_call\_\_](../modules/export.html#detectron2.export.Caffe2Model.__call__) 메서드를 제공합니다.
이 메서드는 [pytorch 버전 모델들](./models.md) 과 동일한 인터페이스를 가지며,
내부적으로 포맷을 맞추기 위한 전/후처리 코드를 적용합니다.
Caffe2의 python API 사용하거나
실제 배포에서 전/후 처리를 구현할 때 이 wrapper를 참고하십시오.

## TensorFlow로의 변환
[tensorpack Faster R-CNN](https://github.com/tensorpack/tensorpack/tree/master/examples/FasterRCNN/convert_d2) 는
몇 가지 표준 detectron2 R-CNN 모델을 TensorFlow의 pb 포맷으로 변환하는 스크립트를 제공합니다.
설정값이나 가중치를 번환하는 방식으로 동작하기 때문에, 일부 모델에 대해서만 지원합니다.
