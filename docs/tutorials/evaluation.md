
# Evaluation

Evaluation is a process that takes a number of inputs/outputs pairs and aggregate them.
You can always [use the model](./models.md) directly and just parse its inputs/outputs manually to perform
evaluation.
Alternatively, evaluation is implemented in detectron2 using the [DatasetEvaluator](../modules/evaluation.html#detectron2.evaluation.DatasetEvaluator)
interface.

Detectron2 includes a few `DatasetEvaluator` that computes metrics using standard dataset-specific
APIs (e.g., COCO, LVIS).
You can also implement your own `DatasetEvaluator` that performs some other jobs
using the inputs/outputs pairs.
For example, to count how many instances are detected on the validation set:

```
class Counter(DatasetEvaluator):
  def reset(self):
    self.count = 0
  def process(self, inputs, outputs):
    for output in outputs:
      self.count += len(output["instances"])
  def evaluate(self):
    # save self.count somewhere, or print it, or return it.
    return {"count": self.count}
```

Once you have some `DatasetEvaluator`, you can run it with
[inference_on_dataset](../modules/evaluation.html#detectron2.evaluation.inference_on_dataset).
For example,

```python
val_results = inference_on_dataset(
    model,
    val_data_loader,
    DatasetEvaluators([COCOEvaluator(...), Counter()]))
```
Compared to running the evaluation manually using the model, the benefit of this function is that
you can merge evaluators together using [DatasetEvaluators](../modules/evaluation.html#detectron2.evaluation.DatasetEvaluators).
In this way you can run all evaluations without having to go through the dataset multiple times.

The `inference_on_dataset` function also provides accurate speed benchmarks for the
given model and dataset.
