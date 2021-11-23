#!/bin/bash -e
# Copyright (c) Facebook, Inc. and its affiliates.

[[ -d "dev/packaging" ]] || {
  echo "Please run this script at detectron2 root!"
  exit 1
}

build_one() {
  cu=$1
  pytorch_ver=$2

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
  container_id="$container_name"_"$cu"_"$pytorch_ver"

  py_versions=(3.6 3.7 3.8 3.9)

  for py in "${py_versions[@]}"; do
    docker run -itd \
      --name "$container_id" \
      --mount type=bind,source="$(pwd)",target=/detectron2 \
      pytorch/$container_name

    cat <<EOF | docker exec -i $container_id sh
      export CU_VERSION=$cu D2_VERSION_SUFFIX=+$cu PYTHON_VERSION=$py
      export PYTORCH_VERSION=$pytorch_ver
      cd /detectron2 && ./dev/packaging/build_wheel.sh
EOF

    docker container stop $container_id
    docker container rm $container_id
  done
}


if [[ -n "$1" ]] && [[ -n "$2" ]]; then
  build_one "$1" "$2"
else
  build_one cu113 1.10
  build_one cu111 1.10
  build_one cu102 1.10
  build_one cpu 1.10

  build_one cu111 1.9
  build_one cu102 1.9
  build_one cpu 1.9

  build_one cu111 1.8
  build_one cu102 1.8
  build_one cu101 1.8
  build_one cpu 1.8
fi
