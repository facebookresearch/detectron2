#!/bin/bash -e
# Copyright (c) Facebook, Inc. and its affiliates.

BIN="python tools/train_net.py"
OUTPUT="instant_test_output"
NUM_GPUS=2

CFG_LIST=( "${@:1}" )
if [ ${#CFG_LIST[@]} -eq 0 ]; then
  CFG_LIST=( ./configs/quick_schedules/*instant_test.yaml )
fi

echo "========================================================================"
echo "Configs to run:"
echo "${CFG_LIST[@]}"
echo "========================================================================"

for cfg in "${CFG_LIST[@]}"; do
    echo "========================================================================"
    echo "Running $cfg ..."
    echo "========================================================================"
    $BIN --num-gpus $NUM_GPUS --config-file "$cfg" \
      SOLVER.IMS_PER_BATCH $(($NUM_GPUS * 2)) \
      OUTPUT_DIR "$OUTPUT"
    rm -rf "$OUTPUT"
done

