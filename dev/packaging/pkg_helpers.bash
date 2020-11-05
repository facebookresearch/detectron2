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
  # See more details at https://github.com/pytorch/pytorch/blob/master/torch/utils/cpp_extension.py#L1363
  export FORCE_CUDA=1
  case "$CU_VERSION" in
    cu110)
      export CUDA_HOME=/usr/local/cuda-11.0/
      # NOTE: may need to add 8.6 in the next ver
      export TORCH_CUDA_ARCH_LIST="3.7;5.0;5.2;6.0+PTX;6.1+PTX;7.0+PTX;7.5+PTX;8.0+PTX"
      ;;
    cu102)
      export CUDA_HOME=/usr/local/cuda-10.2/
      export TORCH_CUDA_ARCH_LIST="3.7;5.0;5.2;6.0+PTX;6.1+PTX;7.0+PTX;7.5+PTX"
      ;;
    cu101)
      export CUDA_HOME=/usr/local/cuda-10.1/
      export TORCH_CUDA_ARCH_LIST="3.7;5.0;5.2;6.0+PTX;6.1+PTX;7.0+PTX;7.5+PTX"
      ;;
    cu100)
      export CUDA_HOME=/usr/local/cuda-10.0/
      export TORCH_CUDA_ARCH_LIST="3.7;5.0;5.2;6.0+PTX;6.1+PTX;7.0+PTX;7.5+PTX"
      ;;
    cu92)
      export CUDA_HOME=/usr/local/cuda-9.2/
      export TORCH_CUDA_ARCH_LIST="3.7;5.0;5.2;6.0+PTX;6.1+PTX;7.0+PTX"
      ;;
    cpu)
      unset FORCE_CUDA
      export CUDA_VISIBLE_DEVICES=
      ;;
    *)
      echo "Unrecognized CU_VERSION=$CU_VERSION"
      exit 1
      ;;
  esac
}

setup_wheel_python() {
  case "$PYTHON_VERSION" in
    3.6) python_abi=cp36-cp36m ;;
    3.7) python_abi=cp37-cp37m ;;
    3.8) python_abi=cp38-cp38 ;;
    *)
      echo "Unrecognized PYTHON_VERSION=$PYTHON_VERSION"
      exit 1
      ;;
  esac
  export PATH="/opt/python/$python_abi/bin:$PATH"
}
