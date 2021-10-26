#!/bin/bash -e
# Copyright (c) Facebook, Inc. and its affiliates.


root=$(readlink -f $1)
if [[ -z "$root" ]]; then
  echo "Usage: ./gen_wheel_index.sh /absolute/path/to/wheels"
  exit
fi

export LC_ALL=C  # reproducible sort
# NOTE: all sort in this script might not work when xx.10 is released

index=$root/index.html

cd "$root"
for cu in cpu cu92 cu100 cu101 cu102 cu110 cu111 cu113; do
  mkdir -p "$root/$cu"
  cd "$root/$cu"
  echo "Creating $PWD/index.html ..."
  # First sort by torch version, then stable sort by d2 version with unique.
  # As a result, the latest torch version for each d2 version is kept.
  for whl in $(find -type f -name '*.whl' -printf '%P\n' \
    | sort -k 1 -r  | sort -t '/' -k 2 --stable -r --unique); do
    echo "<a href=\"${whl/+/%2B}\">$whl</a><br>"
  done > index.html


  for torch in torch*; do
    cd "$root/$cu/$torch"

    # list all whl for each cuda,torch version
    echo "Creating $PWD/index.html ..."
    for whl in $(find . -type f -name '*.whl' -printf '%P\n' | sort -r); do
      echo "<a href=\"${whl/+/%2B}\">$whl</a><br>"
    done > index.html
  done
done

cd "$root"
# Just list everything:
echo "Creating $index ..."
for whl in $(find . -type f -name '*.whl' -printf '%P\n' | sort -r); do
  echo "<a href=\"${whl/+/%2B}\">$whl</a><br>"
done > "$index"

