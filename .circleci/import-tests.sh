#!/bin/bash -e
# Copyright (c) Facebook, Inc. and its affiliates.

# Test that import works without building detectron2.

# Check that _C is not importable
python -c "from detectron2 import _C" > /dev/null 2>&1 && {
  echo "This test should be run without building detectron2."
  exit 1
}

# Check that other modules are still importable, even when _C is not importable
python -c "from detectron2 import modeling"
python -c "from detectron2 import modeling, data"
python -c "from detectron2 import evaluation, export, checkpoint"
python -c "from detectron2 import utils, engine"
