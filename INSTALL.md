## Installation

Our [Colab Notebook](https://colab.research.google.com/drive/16jcaJoc6bCFAQ96jDe2HwtXj7BMD_-m5)
has step-by-step instructions that install detectron2.
The [Dockerfile](docker)
also installs detectron2 with a few simple commands.

### Requirements
- Linux or macOS with Python ≥ 3.6
- PyTorch ≥ 1.4
- [torchvision](https://github.com/pytorch/vision/) that matches the PyTorch installation.
	You can install them together at [pytorch.org](https://pytorch.org) to make sure of this.
- OpenCV, optional, needed by demo and visualization
- pycocotools: `pip install cython; pip install -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'`


### Build Detectron2 from Source

gcc & g++ ≥ 5 are required. [ninja](https://ninja-build.org/) is recommended for faster build.
After having them, run:
```
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
# (add --user if you don't have permission)

# Or, to install it from a local clone:
git clone https://github.com/facebookresearch/detectron2.git
python -m pip install -e detectron2

# Or if you are on macOS
# CC=clang CXX=clang++ python -m pip install -e .
```

To __rebuild__ detectron2 that's built from a local clone, use `rm -rf build/ **/*.so` to clean the
old build first. You often need to rebuild detectron2 after reinstalling PyTorch.

### Install Pre-Built Detectron2 (Linux only)
```
# for CUDA 10.1:
python -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu101/index.html
```
You can replace cu101 with "cu{100,92}" or "cpu".

Note that:
1. Such installation has to be used with certain version of official PyTorch release.
   See [releases](https://github.com/facebookresearch/detectron2/releases) for requirements.
   It will not work with a different version of PyTorch or a non-official build of PyTorch.
2. Such installation is out-of-date w.r.t. master branch of detectron2. It may not be
	 compatible with the master branch of a research project that uses detectron2 (e.g. those in
	 [projects](projects) or [meshrcnn](https://github.com/facebookresearch/meshrcnn/)).

### Common Installation Issues

If you met issues using the pre-built detectron2, please uninstall it and try building it from source.

Click each issue for its solutions:

<details>
<summary>
Undefined torch/aten/caffe2 symbols, or segmentation fault immediately when running the library.
</summary>
<br/>

This usually happens when detectron2 or torchvision is not
compiled with the version of PyTorch you're running.

Pre-built torchvision or detectron2 has to work with the corresponding official release of pytorch.
If the error comes from a pre-built torchvision, uninstall torchvision and pytorch and reinstall them
following [pytorch.org](http://pytorch.org). So the versions will match.

If the error comes from a pre-built detectron2, check [release notes](https://github.com/facebookresearch/detectron2/releases)
to see the corresponding pytorch version required for each pre-built detectron2.

If the error comes from detectron2 or torchvision that you built manually from source,
remove files you built (`build/`, `**/*.so`) and rebuild it so it can pick up the version of pytorch currently in your environment.

If you cannot resolve this problem, please include the output of `gdb -ex "r" -ex "bt" -ex "quit" --args python -m detectron2.utils.collect_env`
in your issue.
</details>

<details>
<summary>
Undefined C++ symbols (e.g. `GLIBCXX`) or C++ symbols not found.
</summary>
<br/>
Usually it's because the library is compiled with a newer C++ compiler but run with an old C++ runtime.

This often happens with old anaconda.
Try `conda update libgcc`. Then rebuild detectron2.

The fundamental solution is to run the code with proper C++ runtime.
One way is to use `LD_PRELOAD=/path/to/libstdc++.so`.

</details>

<details>
<summary>
"Not compiled with GPU support" or "Detectron2 CUDA Compiler: not available".
</summary>
<br/>
CUDA is not found when building detectron2.
You should make sure

```
python -c 'import torch; from torch.utils.cpp_extension import CUDA_HOME; print(torch.cuda.is_available(), CUDA_HOME)'
```

print valid outputs at the time you build detectron2.

Most models can run inference (but not training) without GPU support. To use CPUs, set `MODEL.DEVICE='cpu'` in the config.
</details>

<details>
<summary>
"invalid device function" or "no kernel image is available for execution".
</summary>
<br/>
Two possibilities:

* You build detectron2 with one version of CUDA but run it with a different version.

  To check whether it is the case,
  use `python -m detectron2.utils.collect_env` to find out inconsistent CUDA versions.
	In the output of this command, you should expect "Detectron2 CUDA Compiler", "CUDA_HOME", "PyTorch built with - CUDA"
	to contain cuda libraries of the same version.

	When they are inconsistent,
	you need to either install a different build of PyTorch (or build by yourself)
	to match your local CUDA installation, or install a different version of CUDA to match PyTorch.

* Detectron2 or PyTorch/torchvision is not built for the correct GPU architecture (compute compatibility).

	The GPU architecture for PyTorch/detectron2/torchvision is available in the "architecture flags" in
	`python -m detectron2.utils.collect_env`.

	The GPU architecture flags of detectron2/torchvision by default matches the GPU model detected
	during compilation. This means the compiled code may not work on a different GPU model.
	To overwrite the GPU architecture for detectron2/torchvision, use `TORCH_CUDA_ARCH_LIST` environment variable during compilation.

	For example, `export TORCH_CUDA_ARCH_LIST=6.0,7.0` makes it compile for both P100s and V100s.
	Visit [developer.nvidia.com/cuda-gpus](https://developer.nvidia.com/cuda-gpus) to find out
	the correct compute compatibility number for your device.

</details>

<details>
<summary>
Undefined CUDA symbols; cannot open libcudart.so; other nvcc failures.
</summary>
<br/>
The version of NVCC you use to build detectron2 or torchvision does
not match the version of CUDA you are running with.
This often happens when using anaconda's CUDA runtime.

Use `python -m detectron2.utils.collect_env` to find out inconsistent CUDA versions.
In the output of this command, you should expect "Detectron2 CUDA Compiler", "CUDA_HOME", "PyTorch built with - CUDA"
to contain cuda libraries of the same version.

When they are inconsistent,
you need to either install a different build of PyTorch (or build by yourself)
to match your local CUDA installation, or install a different version of CUDA to match PyTorch.
</details>


<details>
<summary>
"ImportError: cannot import name '_C'".
</summary>
<br/>
Please build and install detectron2 following the instructions above.

If you are running code from detectron2's root directory, `cd` to a different one.
Otherwise you may not import the code that you installed.
</details>

<details>
<summary>
ONNX conversion segfault after some "TraceWarning".
</summary>
<br/>
The ONNX package is compiled with too old compiler.

Please build and install ONNX from its source code using a compiler
whose version is closer to what's used by PyTorch (available in `torch.__config__.show()`).
</details>
