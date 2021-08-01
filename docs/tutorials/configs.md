# Yacs Configs

Detectron2 provides a key-value based config system that can be
used to obtain standard, common behaviors.

This system uses YAML and [yacs](https://github.com/rbgirshick/yacs).
Yaml is a very limited language,
so we do not expect all features in detectron2 to be available through configs.
If you need something that's not available in the config space,
please write code using detectron2's API.

With the introduction of a more powerful [LazyConfig system](lazyconfigs.md),
we no longer add functionality / new keys to the Yacs/Yaml-based config system.

### Basic Usage

Some basic usage of the `CfgNode` object is shown here. See more in [documentation](../modules/config.html#detectron2.config.CfgNode).
```python
from detectron2.config import get_cfg
cfg = get_cfg()    # obtain detectron2's default config
cfg.xxx = yyy      # add new configs for your own custom components
cfg.merge_from_file("my_cfg.yaml")   # load values from a file

cfg.merge_from_list(["MODEL.WEIGHTS", "weights.pth"])   # can also load values from a list of str
print(cfg.dump())  # print formatted configs
with open("output.yaml", "w") as f:
  f.write(cfg.dump())   # save config to file
```

In addition to the basic Yaml syntax, the config file can
define a `_BASE_: base.yaml` field, which will load a base config file first.
Values in the base config will be overwritten in sub-configs, if there are any conflicts.
We provided several base configs for standard model architectures.

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
