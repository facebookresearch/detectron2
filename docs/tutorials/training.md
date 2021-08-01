# Training

From the previous tutorials, you may now have a custom model and a data loader.
To run training, users typically have a preference in one of the following two styles:

### Custom Training Loop

With a model and a data loader ready, everything else needed to write a training loop can
be found in PyTorch, and you are free to write the training loop yourself.
This style allows researchers to manage the entire training logic more clearly and have full control.
One such example is provided in [tools/plain_train_net.py](../../tools/plain_train_net.py).

Any customization on the training logic is then easily controlled by the user.

### Trainer Abstraction

We also provide a standarized "trainer" abstraction with a
hook system that helps simplify the standard training behavior.
It includes the following two instantiations:

* [SimpleTrainer](../modules/engine.html#detectron2.engine.SimpleTrainer)
  provides a minimal training loop for single-cost single-optimizer single-data-source training, with nothing else.
  Other tasks (checkpointing, logging, etc) can be implemented using
  [the hook system](../modules/engine.html#detectron2.engine.HookBase).
* [DefaultTrainer](../modules/engine.html#detectron2.engine.defaults.DefaultTrainer) is a `SimpleTrainer` initialized from a
  yacs config, used by
  [tools/train_net.py](../../tools/train_net.py) and many scripts.
  It includes more standard default behaviors that one might want to opt in,
  including default configurations for optimizer, learning rate schedule,
  logging, evaluation, checkpointing etc.

To customize a `DefaultTrainer`:

1. For simple customizations (e.g. change optimizer, evaluator, LR scheduler, data loader, etc.), overwrite [its methods](../modules/engine.html#detectron2.engine.defaults.DefaultTrainer) in a subclass, just like [tools/train_net.py](../../tools/train_net.py).
2. For extra tasks during training, check the
   [hook system](../modules/engine.html#detectron2.engine.HookBase) to see if it's supported.

   As an example, to print hello during training:
   ```python
   class HelloHook(HookBase):
     def after_step(self):
       if self.trainer.iter % 100 == 0:
         print(f"Hello at iteration {self.trainer.iter}!")
   ```
3. Using a trainer+hook system means there will always be some non-standard behaviors that cannot be supported, especially in research.
   For this reason, we intentionally keep the trainer & hook system minimal, rather than powerful.
   If anything cannot be achieved by such a system, it's easier to start from [tools/plain_train_net.py](../../tools/plain_train_net.py) to implement custom training logic manually.

### Logging of Metrics

During training, detectron2 models and trainer put metrics to a centralized [EventStorage](../modules/utils.html#detectron2.utils.events.EventStorage).
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

Metrics are then written to various destinations with [EventWriter](../modules/utils.html#module-detectron2.utils.events).
DefaultTrainer enables a few `EventWriter` with default configurations.
See above for how to customize them.
