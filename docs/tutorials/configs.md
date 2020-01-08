# Use Configs

Detectron2's config system uses yaml and [yacs](https://github.com/rbgirshick/yacs).
In addition to the basic operations that access and update a config, we provide
the following extra functionalities:

1. The config can have `_BASE_: base.yaml` field, which will load a base config first.
   Values in the base config will be overwritten in sub-configs, if there are any conflicts.
   We provided several base configs for standard model architectures.
2. We provide config versioning, for backward compatibility.
   If your config file is versioned with a config line like `VERSION: 2`,
   detectron2 will still recognize it even if we rename some keys in the future.

### Use Configs

Some basic usage of the `CfgNode` object is shown below:
```python
from detectron2.config import get_cfg
cfg = get_cfg()    # obtain detectron2's default config
cfg.xxx = yyy      # add new configs for your own custom components
cfg.merge_from_file("my_cfg.yaml")   # load values from a file

cfg.merge_from_list(["MODEL.WEIGHTS", "weights.pth"])   # can also load values from a list of str
print(cfg.dump())  # print formatted configs
```

To see a list of available configs in detectron2, see [Config References](../modules/config.html#config-references)


### Best Practice with Configs

1. Treat the configs you write as "code": avoid copying them or duplicating them; use `_BASE_`
   to share common parts between configs.

2. Keep the configs you write simple: don't include keys that do not affect the experimental setting.

3. Keep a version number in your configs (or the base config), e.g., `VERSION: 2`,
   for backward compatibility.
	 We print a warning when reading a config without version number.
   The official configs do not include version number because they are meant to
   be always up-to-date.

4. Save a full config together with a trained model, and use it to run inference.
   This is more robust to changes that may happen to the config definition
   (e.g., if a default value changed), although we will try to avoid such changes.
