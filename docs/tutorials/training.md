# Training

From the previous tutorials, you may now have a custom model and data loader.

You are free to create your own optimizer, and write the training logic: it's
usually easy with PyTorch, and allow researchers to see the entire training
logic more clearly and have full control.
One such example is provided in [tools/plain_train_net.py](../../tools/plain_train_net.py).

We also provide a standarized "trainer" abstraction with a
[minimal hook system](../modules/engine.html#detectron2.engine.HookBase)
that helps simplify the standard types of training.

You can use
[SimpleTrainer().train()](../modules/engine.html#detectron2.engine.SimpleTrainer)
which provides minimal abstraction for single-cost single-optimizer single-data-source training.
The builtin `train_net.py` script uses
[DefaultTrainer().train()](../modules/engine.html#detectron2.engine.defaults.DefaultTrainer),
which includes more standard default behavior that one might want to opt in,
including default configurations for learning rate schedule,
logging, evaluation, checkpointing etc.
This also means that it's less likely to support some non-standard behavior
you might want during research.

To customize the training loops, you can:

1. If your customization is similar to what `DefaultTrainer` is already doing,
you can change behavior of `DefaultTrainer` by overwriting [its methods](../modules/engine.html#detectron2.engine.defaults.DefaultTrainer)
in a subclass, like what [tools/train_net.py](../../tools/train_net.py) does.
2. If you need something very novel, you can start from [tools/plain_train_net.py](../../tools/plain_train_net.py) to implement them yourself.

### Logging of Metrics

During training, metrics are saved to a centralized [EventStorage](../modules/utils.html#detectron2.utils.events.EventStorage).
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

Metrics are then saved to various destinations with [EventWriter](../modules/utils.html#module-detectron2.utils.events).
DefaultTrainer enables a few `EventWriter` with default configurations.
See above for how to customize them.
