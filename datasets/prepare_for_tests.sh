#!/bin/bash -e
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

# Download some files needed for running tests.

cd "${0%/*}"

BASE=https://dl.fbaipublicfiles.com/detectron2
mkdir -p coco/annotations

for anno in instances_val2017_100 \
  person_keypoints_val2017_100 \
  instances_minival2014_100 \
  person_keypoints_minival2014_100; do

  dest=coco/annotations/$anno.json
  [[ -s $dest ]] && {
    echo "$dest exists. Skipping ..."
  } || {
    wget $BASE/annotations/coco/$anno.json -O $dest
  }
done
