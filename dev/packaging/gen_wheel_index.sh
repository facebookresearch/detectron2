#!/bin/bash -e
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved


root=$1
if [[ -z "$root" ]]; then
  echo "Usage: ./gen_wheel_index.sh /path/to/wheels"
  exit
fi

index=$root/index.html

cd "$root"
for cu in cpu cu92 cu100 cu101 cu102; do
  cd $cu
  echo "Creating $PWD/index.html ..."
  for whl in *.whl; do
    echo "<a href=\"${whl/+/%2B}\">$whl</a><br>"
  done > index.html
  cd "$root"
done

echo "Creating $index ..."
for whl in $(find . -type f -name '*.whl' -printf '%P\n' | sort); do
  echo "<a href=\"${whl/+/%2B}\">$whl</a><br>"
done > "$index"

