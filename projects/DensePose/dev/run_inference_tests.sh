#!/bin/bash -e
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

BIN="python train_net.py"
OUTPUT="inference_test_output"
NUM_GPUS=2
IMS_PER_GPU=2
IMS_PER_BATCH=$(( NUM_GPUS * IMS_PER_GPU ))

CFG_LIST=( "${@:1}" )

if [ ${#CFG_LIST[@]} -eq 0 ]; then
  CFG_LIST=( ./configs/quick_schedules/*inference_acc_test.yaml )
fi

echo "========================================================================"
echo "Configs to run:"
echo "${CFG_LIST[@]}"
echo "========================================================================"

for cfg in "${CFG_LIST[@]}"; do
    echo "========================================================================"
    echo "Running $cfg ..."
    echo "========================================================================"
    $BIN \
      --eval-only \
      --num-gpus $NUM_GPUS \
      --config-file "$cfg" \
      OUTPUT_DIR "$OUTPUT" \
      SOLVER.IMS_PER_BATCH $IMS_PER_BATCH
    rm -rf $OUTPUT
done

