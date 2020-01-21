# Training

From the previous tutorials, you may now have a custom model and data loader.

You are free to create your own optimizer, and write the training logic: it's
usually easy with PyTorch, and allow researchers to see the entire training
logic more clearly.
One such example is provided in [tools/plain_train_net.py](../../tools/plain_train_net.py).

We also provide a standarized "trainer" abstraction with a
[minimal hook system](../modules/engine.html#detectron2.engine.HookBase)
that helps simplify the standard types of training.

You can use
[SimpleTrainer().train()](../modules/engine.html#detectron2.engine.SimpleTrainer)
which does single-cost single-optimizer single-data-source training.
Or use [DefaultTrainer().train()](../modules/engine.html#detectron2.engine.defaults.DefaultTrainer)
which includes more standard behavior that one might want to opt in.
This also means that it's less likely to support some non-standard behavior
you might want during research.


### Logging of Metrics

During training, metrics are logged with a centralized [EventStorage](../modules/utils.html#detectron2.utils.events.EventStorage).
You can use the following code to access it and log metrics to it:
```
from detectron2.utils.events import get_event_storage

# inside the model:
if self.training:
  value = # compute the value from inputs
  storage = get_event_storage()
  storage.put_scalar("some_accuracy", value)
```

Refer to its documentation for more details.
