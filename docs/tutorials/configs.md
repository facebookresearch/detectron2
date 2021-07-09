# Configs

Detectron2 provides a key-value based config system that can be
used to obtain standard, common behaviors.

Detectron2's config system uses YAML and [yacs](https://github.com/rbgirshick/yacs).
In addition to the [basic operations](../modules/config.html#detectron2.config.CfgNode)
that access and update a config, we provide the following extra functionalities:

1. The config can have `_BASE_: base.yaml` field, which will load a base config first.
   Values in the base config will be overwritten in sub-configs, if there are any conflicts.
   We provided several base configs for standard model architectures.
2. We provide config versioning, for backward compatibility.
   If your config file is versioned with a config line like `VERSION: 2`,
   detectron2 will still recognize it even if we change some keys in the future.

Config file is a very limited language.
We do not expect all features in detectron2 to be available through configs.
If you need something that's not available in the config space,
please write code using detectron2's API.

### Basic Usage

Some basic usage of the `CfgNode` object is shown here. See more in [documentation](../modules/config.html#detectron2.config.CfgNode).
```python
from detectron2.config import get_cfg
cfg = get_cfg()    # obtain detectron2's default config
cfg.xxx = yyy      # add new configs for your own custom components
cfg.merge_from_file("my_cfg.yaml")   # load values from a file

cfg.merge_from_list(["MODEL.WEIGHTS", "weights.pth"])   # can also load values from a list of str
print(cfg.dump())  # print formatted configs
```

Many builtin tools in detectron2 accept command line config overwrite:
Key-value pairs provided in the command line will overwrite the existing values in the config file.
For example, [demo.py](../../demo/demo.py) can be used with
```
./demo.py --config-file config.yaml [--other-options] \
  --opts MODEL.WEIGHTS /path/to/weights INPUT.MIN_SIZE_TEST 1000
```

To see a list of available configs in detectron2 and what they mean,
check [Config References](../modules/config.html#config-references)


### Configs in Projects

A project that lives outside the detectron2 library may define its own configs, which will need to be added
for the project to be functional, e.g.:
```python
from detectron2.projects.point_rend import add_pointrend_config
cfg = get_cfg()    # obtain detectron2's default config
add_pointrend_config(cfg)  # add pointrend's default config
# ... ...
```

### Best Practice with Configs

1. Treat the configs you write as "code": avoid copying them or duplicating them; use `_BASE_`
   to share common parts between configs.

2. Keep the configs you write simple: don't include keys that do not affect the experimental setting.

3. Keep a version number in your configs (or the base config), e.g., `VERSION: 2`,
   for backward compatibility.
	 We print a warning when reading a config without version number.
   The official configs do not include version number because they are meant to
   be always up-to-date.
