#!/bin/bash -e
# Copyright (c) Facebook, Inc. and its affiliates.

# Function to retry functions that sometimes timeout or have flaky failures
retry () {
    $*  || (sleep 1 && $*) || (sleep 2 && $*) || (sleep 4 && $*) || (sleep 8 && $*)
}
# Install with pip a bit more robustly than the default
pip_install() {
  retry pip install --progress-bar off "$@"
}


setup_cuda() {
  # Now work out the CUDA settings
  # Like other torch domain libraries, we choose common GPU architectures only.
  # See https://github.com/pytorch/pytorch/blob/master/torch/utils/cpp_extension.py
  # and https://github.com/pytorch/vision/blob/main/packaging/pkg_helpers.bash for reference.
  export FORCE_CUDA=1
  case "$CU_VERSION" in
    cu118)
      export CUDA_HOME=/usr/local/cuda-11.8/
      export TORCH_CUDA_ARCH_LIST="3.7+PTX;5.0;6.0;6.1;7.0;7.5;8.0;8.6;9.0"
      ;;
    *)
      echo "Unrecognized CU_VERSION=$CU_VERSION"
      exit 1
      ;;
  esac
}

setup_wheel_python() {
  case "$PYTHON_VERSION" in
    3.9) python_abi=cp39-cp39 ;;
    3.10) python_abi=cp310-cp310 ;;
    3.11) python_abi=cp311-cp311 ;;
    *)
      echo "Unrecognized PYTHON_VERSION=$PYTHON_VERSION"
      exit 1
      ;;
  esac
  export PATH="/opt/python/$python_abi/bin:$PATH"
}