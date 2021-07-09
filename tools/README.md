
This directory contains a few example scripts that demonstrate features of detectron2.


* `train_net.py`

An example training script that's made to train builtin models of detectron2.

For usage, see [GETTING_STARTED.md](../GETTING_STARTED.md).

* `plain_train_net.py`

Similar to `train_net.py`, but implements a training loop instead of using `Trainer`.
This script includes fewer features but it may be more friendly to hackers.

* `benchmark.py`

Benchmark the training speed, inference speed or data loading speed of a given config.

Usage:
```
python benchmark.py --config-file config.yaml --task train/eval/data [optional DDP flags]
```

* `analyze_model.py`

Analyze FLOPs, parameters, activations of a detectron2 model.  See its `--help` for usage.

* `visualize_json_results.py`

Visualize the json instance detection/segmentation results dumped by `COCOEvalutor` or `LVISEvaluator`

Usage:
```
python visualize_json_results.py --input x.json --output dir/ --dataset coco_2017_val
```
If not using a builtin dataset, you'll need your own script or modify this script.

* `visualize_data.py`

Visualize ground truth raw annotations or training data (after preprocessing/augmentations).

Usage:
```
python visualize_data.py --config-file config.yaml --source annotation/dataloader --output-dir dir/ [--show]
```

NOTE: the script does not stop by itself when using `--source dataloader` because a training
dataloader is usually infinite.
