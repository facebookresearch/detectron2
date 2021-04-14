This directory provides definitions for a few common models, dataloaders, scheduler,
and optimizers that are often used in training.
The definition of these objects are provided in the form of lazy instantiation:
their arguments can be edited by users before constructing the objects.

They can be imported, or loaded by `model_zoo.get_config` API in users' own configs.
