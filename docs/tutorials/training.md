# 학습

이전 튜토리얼을 통해 커스텀 모델과 데이터로더를 배워볼 수 있었습니다.
모델 학습의 경우, 많은 사람들이 둘 중 하나의 방식을 선호합니다.

### 커스텀 학습 루프

모델과 데이터로더만 준비됐다면 학습 루프 작성을 위한 다른 모든 것들이
PyTorch에서 제공되므로, 직접 학습 루프를 작성할 수 있습니다.
이러한 방식은 연구자들이 보다 명시적으로 전체 학습 로직을 관리하고 모든 권한을 가질 수 있게 합니다.
[tools/plain_train_net.py](../../tools/plain_train_net.py) 의 사례를 확인하십시오.

작성자가 커스터마이징한 모든 학습 로직을 쉽게 제어할 수 있습니다.

### Trainer 모듈

또한 저희는 표준화된 "Trainer" 모듈과
학습 과정을 단순화하는 데 도움이 되는 hook 시스템을 제공합니다.
여기에는 아래 두 가지 구현체도 포합됩니다.

* [SimpleTrainer](../modules/engine.html#detectron2.engine.SimpleTrainer) 는
  하나의 비용함수와 optimizer를 사용하며 단일 데이터 소스에 대해 학습하는 가장 간단한 학습 루프를 제공합니다.
  다른 작업(체크포인트 저장, 로그 생성 등)은 [hook 시스템](../modules/engine.html#detectron2.engine.HookBase) 을
  사용해 구현할 수 있습니다.
* [DefaultTrainer](../modules/engine.html#detectron2.engine.defaults.DefaultTrainer) 는 yacs를 통해 `SimpleTrainer` 의 환경 변수를
  설정한 것으로, [tools/train_net.py](../../tools/train_net.py) 를
  비롯한 많은 스크립트에서 사용됩니다.
  이는 optimizer, 학습률 (learning rate) 스케줄링, 로그 기록,
  모델 평가 (evaluation), 체크포인트 저장 등에 대한 기본 설정을
  비롯해 다른 수많은 일반적인 기본 동작을 포함합니다.

`DefaultTrainer` 를 커스터마이징하려면:

1. 간단한 커스터마이징(e.g. optimizer, evaluator, LR 스케줄러, 데이터로더 등의 변경)의 경우, [tools/train_net.py](../../tools/train_net.py) 와 같이 하위 클래스에서 [해당 메서드](../modules/engine.html#detectron2.engine.defaults.DefaultTrainer) 를 덮어쓰십시오.
2. 학습 중 추가적인 작업을 수행하려면
   [hook system](../modules/engine.html#detectron2.engine.HookBase) 에서 지원 여부를 확인하십시오.

   예를 들어 다음과 같이 학습 중에 hello를 출력할 수 있습니다.
   ```python
   class HelloHook(HookBase):
     def after_step(self):
       if self.trainer.iter % 100 == 0:
         print(f"Hello at iteration {self.trainer.iter}!")
   ```
3. trainer 및 hook 시스템을 사용할 경우, 지원되지 않는 비표준적인 동작이 언제든 있을 수 있습니다. 특히 연구에서.
   이러한 이유로 trainer 및 hook 시스템은 강력하기보다 미니멀하게 설계됐습니다.
   이 시스템으로 달성할 수 없는 것이 하나라도 있다면 [tools/plain_train_net.py](../../tools/plain_train_net.py) 부터 시작해 커스텀 학습 로직을 직접 구현하는 것이 더 쉬울 것입니다.

### 메트릭 로그 생성

학습 중에 detectron2 모델과 trainer는 [EventStorage](../modules/utils.html#detectron2.utils.events.EventStorage) 에 메트릭을 수집합니다.
여기에 접근해 로그 메트릭을 생성하려면 다음 코드를 사용하십시오.
```python
from detectron2.utils.events import get_event_storage

# 모델 안에서
if self.training:
  value = # compute the value from inputs
  storage = get_event_storage()
  storage.put_scalar("some_accuracy", value)
```

자세한 내용은 해당 문서를 참조하십시오.

이후 메트릭은 [EventWriter](../modules/utils.html#module-detectron2.utils.events) 를 통해 다양한 대상에 기록됩니다.
기본 설정된 DefaultTrainer는 몇 가지 'EventWriter'를 활성화합니다.
사용자 지정 방법은 위 링크를 참조하십시오.
