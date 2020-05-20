#!/bin/bash -e
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

PYTORCH_VERSION=1.5

build_for_one_cuda() {
  cu=$1

  case "$cu" in
    cu*)
      container_name=manylinux-cuda${cu/cu/}
      ;;
    cpu)
      container_name=manylinux-cuda101
      ;;
    *)
      echo "Unrecognized cu=$cu"
      exit 1
      ;;
  esac

  echo "Launching container $container_name ..."

  for py in 3.6 3.7 3.8; do
    docker run -itd \
      --name $container_name \
      --mount type=bind,source="$(pwd)",target=/detectron2 \
      pytorch/$container_name

    cat <<EOF | docker exec -i $container_name sh
      export CU_VERSION=$cu D2_VERSION_SUFFIX=+$cu PYTHON_VERSION=$py
      export PYTORCH_VERSION=$PYTORCH_VERSION
      cd /detectron2 && ./dev/packaging/build_wheel.sh
EOF

#     if [[ "$cu" == "cu101" ]]; then
#       # build wheel without local version
#       cat <<EOF | docker exec -i $container_name sh
#         export CU_VERSION=$cu D2_VERSION_SUFFIX= PYTHON_VERSION=$py
#         export PYTORCH_VERSION=$PYTORCH_VERSION
#         cd /detectron2 && ./dev/packaging/build_wheel.sh
# EOF
#     fi

    docker exec -i $container_name rm -rf /detectron2/build/$cu
    docker container stop $container_name
    docker container rm $container_name
  done
}

if [[ -n "$1" ]]; then
  build_for_one_cuda "$1"
else
  for cu in cu102 cu101 cu92 cpu; do
    build_for_one_cuda "$cu"
  done
fi
