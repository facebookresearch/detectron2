#!/bin/bash -e
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

# Run this script at project root by "./dev/linter.sh" before you commit

{
  black --version | grep -E "(19.3b0.*6733274)|(19.3b0\\+8)" > /dev/null
} || {
	echo "Linter requires 'black @ git+https://github.com/psf/black@673327449f86fce558adde153bb6cbe54bfebad2' !"
	exit 1
}

ISORT_VERSION=$(isort --version-number)
if [[ "$ISORT_VERSION" != 4.3* ]]; then
  echo "Linter requires isort==4.3.21 !"
  exit 1
fi

set -v

echo "Running isort ..."
isort -y -sp . --atomic

echo "Running black ..."
black -l 100 .

echo "Running flake8 ..."
if [ -x "$(command -v flake8-3)" ]; then
  flake8-3 .
else
  python3 -m flake8 .
fi

# echo "Running mypy ..."
# Pytorch does not have enough type annotations
# mypy detectron2/solver detectron2/structures detectron2/config

echo "Running clang-format ..."
find . -regex ".*\.\(cpp\|c\|cc\|cu\|cxx\|h\|hh\|hpp\|hxx\|tcc\|mm\|m\)" -print0 | xargs -0 clang-format -i

command -v arc > /dev/null && arc lint
