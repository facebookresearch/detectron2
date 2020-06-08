
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

## Use evaluators

To evaluate using the methods of evaluators manually:
```
def get_all_inputs_outputs():
  for data in data_loader:
    yield data, model(data)

evaluator.reset()
for inputs, outputs in get_all_inputs_outputs():
  evaluator.process(inputs, outputs)
eval_results = evaluator.evaluate()
```

Evaluators can also be used with [inference_on_dataset](../modules/evaluation.html#detectron2.evaluation.inference_on_dataset).
For example,

```python
eval_results = inference_on_dataset(
    model,
    data_loader,
    DatasetEvaluators([COCOEvaluator(...), Counter()]))
```
This will execute `model` on all inputs from `data_loader`, and call evaluator to process them.

Compared to running the evaluation manually using the model, the benefit of this function is that
evaluators can be merged together using [DatasetEvaluators](../modules/evaluation.html#detectron2.evaluation.DatasetEvaluators),
and all the evaluation can fininsh in one forward pass over the dataset.
This function also provides accurate speed benchmarks for the given model and dataset.
