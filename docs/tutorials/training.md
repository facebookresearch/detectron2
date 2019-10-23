# Training

From the previous tutorials, you may now have a custom model and data loader.

You are free to create your own optimizer, and write the training logic: it's
usually easy with PyTorch, and allow researchers to see the entire training
logic more clearly.
One such example is provided in [tools/plain_train_net.py](https://github.com/facebookresearch/detectron2/blob/master/tools/plain_train_net.py).

We also provide a standarized "trainer" abstraction with a
[minimal hook system](../modules/engine.html#detectron2.engine.HookBase)
that helps simplify the standard types of training.

You can use
[SimpleTrainer().train()](../modules/engine.html#detectron2.engine.SimpleTrainer)
which does single-cost single-optimizer single-data-source training.
Or use [DefaultTrainer().train()](../modules/engine.html#detectron2.engine.defaults.DefaultTrainer)
which includes more standard behavior that one might want to opt in.
