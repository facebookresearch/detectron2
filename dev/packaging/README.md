
## To build a cu101 wheel for release:

```
$ nvidia-docker run -it --storage-opt "size=20GB" --name pt  pytorch/manylinux-cuda101
# inside the container:
# git clone https://github.com/facebookresearch/detectron2/
# cd detectron2
# export CU_VERSION=cu101 D2_VERSION_SUFFIX= PYTHON_VERSION=3.7 PYTORCH_VERSION=1.4
# ./dev/packaging/build_wheel.sh
```

## To build all wheels for `CUDA {9.2,10.0,10.1}` x `Python {3.6,3.7,3.8}`:
```
./dev/packaging/build_all_wheels.sh
./dev/packaging/gen_wheel_index.sh /path/to/wheels
```
