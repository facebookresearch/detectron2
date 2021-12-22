
# 성능평가

성능평가는 여러 입/출력 쌍을 입력받아 집계(aggregate)하는 절차입니다.
누구든 [모델의](./models.md) 입/출력을 파싱하면 성능 평가를 수행할 수 있습니다.
다른 방법으로, detectron2에서는 [DatasetEvaluator](../modules/evaluation.html#detectron2.evaluation.DatasetEvaluator) 인터페이스를 통해 성능평가를
할 수 있습니다.

Detectron2는 몇 가지 `DatasetEvaluator` 를 제공하는데, 이들은 데이터셋 별 표준 API(e.g. COCO, LVIS)를 사용해
성능지표를 계산합니다.
입/출력 쌍을 사용해 다른 작업을 수행하는 여러분만의 `DatasetEvaluator` 를
직접 구현할 수도 있습니다.
예를 들어 다음과 같이 검증셋(validation set)에서 검출된 객체 수를 계산할 수 있습니다.

```python
class Counter(DatasetEvaluator):
  def reset(self):
    self.count = 0
  def process(self, inputs, outputs):
    for output in outputs:
      self.count += len(output["instances"])
  def evaluate(self):
    # self.count를 어딘가 저장하거나, 출력하거나, 반환하십시오.
    return {"count": self.count}
```

## evaluators 사용

다음처럼 evaluator의 메서드를 사용해 직접 성능평가를 할 수 있습니다.
```python
def get_all_inputs_outputs():
  for data in data_loader:
    yield data, model(data)

evaluator.reset()
for inputs, outputs in get_all_inputs_outputs():
  evaluator.process(inputs, outputs)
eval_results = evaluator.evaluate()
```

또한 evaluator를 [inference_on_dataset](../modules/evaluation.html#detectron2.evaluation.inference_on_dataset) 과 함께 사용할 수도 있습니다.
다음 예시를 확인하십시오.

```python
eval_results = inference_on_dataset(
    model,
    data_loader,
    DatasetEvaluators([COCOEvaluator(...), Counter()]))
```
위 코드는 `data_loader` 의 모든 입력에 대해 `model` 을 실행하고 evaluator를 호출해 성능을 평가합니다.

모델을 사용하여 직접 성능평가를 하는 것에 비해 이 함수의 이점은 다음과 같습니다.
첫째, [DatasetEvaluators](../modules/evaluation.html#detectron2.evaluation.DatasetEvaluators) 를 사용해 evaluator들을 하나로 병합할 수 있습니다.
둘째, 데이터셋을 한 번만 순회하고도 모든 성능평가를 끝낼 수 있습니다.
셋째, 평가하는 모델 및 데이터셋에 대한 정확한 속도 측정 벤치마크를 제공합니다.

## 커스텀 데이터셋을 위한 evaluator

detectron2에는 특정 데이터셋을 위한 evaluator가 다수 존재하는데,
이는 각 데이터셋의 공식 API를 통해 성능을 측정하기 위한 목적입니다.
이외에도 detectron2의 [표준 데이터셋 포맷](./datasets.md) 을 따르는 일반(generic) 데이터셋에 대해
성능을 평가하기 위한 evaluator가 두 개 있으며,
이를 커스텀 데이터셋의 성능을 평가하는 데에도 사용할 수 있습니다.

* [COCOEvaluator](../modules/evaluation.html#detectron2.evaluation.COCOEvaluator) 는 모든 커스텀 데이터셋에 대해 박스 검출(box detection), 객체 분할(instance segmentation), 키포인트 검출(keypoint detection)
   AP(평균 정밀도)를 평가할 수 있습니다.
* [SemSegEvaluator](../modules/evaluation.html#detectron2.evaluation.SemSegEvaluator) 는 모든 커스텀 데이터셋에서 의미론적 분할(semantic segmentation) 성능지표를 평가할 수 있습니다.
