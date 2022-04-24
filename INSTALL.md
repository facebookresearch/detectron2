## Installation

### Requirements
- Linux or macOS with Python ≥ 3.6
- PyTorch ≥ 1.8 and [torchvision](https://github.com/pytorch/vision/) that matches the PyTorch installation.
  Install them together at [pytorch.org](https://pytorch.org) to make sure of this
- OpenCV is optional but needed by demo and visualization


### Build Detectron2 from Source

gcc & g++ ≥ 5.4 are required. [ninja](https://ninja-build.org/) is optional but recommended for faster build.
After having them, run:
```
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
# (add --user if you don't have permission)

# Or, to install it from a local clone:
git clone https://github.com/facebookresearch/detectron2.git
python -m pip install -e detectron2

# On macOS, you may need to prepend the above commands with a few environment variables:
CC=clang CXX=clang++ ARCHFLAGS="-arch x86_64" python -m pip install ...
```

To __rebuild__ detectron2 that's built from a local clone, use `rm -rf build/ **/*.so` to clean the
old build first. You often need to rebuild detectron2 after reinstalling PyTorch.

### Install Pre-Built Detectron2 (Linux only)

Choose from this table to install [v0.6 (Oct 2021)](https://github.com/facebookresearch/detectron2/releases):

<table class="docutils"><tbody><th width="80"> CUDA </th><th valign="bottom" align="left" width="100">torch 1.10</th><th valign="bottom" align="left" width="100">torch 1.9</th><th valign="bottom" align="left" width="100">torch 1.8</th> <tr><td align="left">11.3</td><td align="left"><details><summary> install </summary><pre><code>python -m pip install detectron2 -f \
  https://dl.fbaipublicfiles.com/detectron2/wheels/cu113/torch1.10/index.html
</code></pre> </details> </td> <td align="left"> </td> <td align="left"> </td> </tr> <tr><td align="left">11.1</td><td align="left"><details><summary> install </summary><pre><code>python -m pip install detectron2 -f \
  https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.10/index.html
</code></pre> </details> </td> <td align="left"><details><summary> install </summary><pre><code>python -m pip install detectron2 -f \
  https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.9/index.html
</code></pre> </details> </td> <td align="left"><details><summary> install </summary><pre><code>python -m pip install detectron2 -f \
  https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.8/index.html
</code></pre> </details> </td> </tr> <tr><td align="left">10.2</td><td align="left"><details><summary> install </summary><pre><code>python -m pip install detectron2 -f \
  https://dl.fbaipublicfiles.com/detectron2/wheels/cu102/torch1.10/index.html
</code></pre> </details> </td> <td align="left"><details><summary> install </summary><pre><code>python -m pip install detectron2 -f \
  https://dl.fbaipublicfiles.com/detectron2/wheels/cu102/torch1.9/index.html
</code></pre> </details> </td> <td align="left"><details><summary> install </summary><pre><code>python -m pip install detectron2 -f \
  https://dl.fbaipublicfiles.com/detectron2/wheels/cu102/torch1.8/index.html
</code></pre> </details> </td> </tr> <tr><td align="left">10.1</td><td align="left"> </td> <td align="left"> </td> <td align="left"><details><summary> install </summary><pre><code>python -m pip install detectron2 -f \
  https://dl.fbaipublicfiles.com/detectron2/wheels/cu101/torch1.8/index.html
</code></pre> </details> </td> </tr> <tr><td align="left">cpu</td><td align="left"><details><summary> install </summary><pre><code>python -m pip install detectron2 -f \
  https://dl.fbaipublicfiles.com/detectron2/wheels/cpu/torch1.10/index.html
</code></pre> </details> </td> <td align="left"><details><summary> install </summary><pre><code>python -m pip install detectron2 -f \
  https://dl.fbaipublicfiles.com/detectron2/wheels/cpu/torch1.9/index.html
</code></pre> </details> </td> <td align="left"><details><summary> install </summary><pre><code>python -m pip install detectron2 -f \
  https://dl.fbaipublicfiles.com/detectron2/wheels/cpu/torch1.8/index.html
</code></pre> </details> </td> </tr></tbody></table>

Note that:
1. The pre-built packages have to be used with corresponding version of CUDA and the official package of PyTorch.
   Otherwise, please build detectron2 from source.
2. New packages are released every few months. Therefore, packages may not contain latest features in the main
   branch and may not be compatible with the main branch of a research project that uses detectron2
   (e.g. those in [projects](projects)).

### Common Installation Issues

Click each issue for its solutions:

<details>
<summary>
Undefined symbols that looks like "TH..","at::Tensor...","torch..."
</summary>
<br/>

This usually happens when detectron2 or torchvision is not
compiled with the version of PyTorch you're running.

If the error comes from a pre-built torchvision, uninstall torchvision and pytorch and reinstall them
following [pytorch.org](http://pytorch.org). So the versions will match.

If the error comes from a pre-built detectron2, check [release notes](https://github.com/facebookresearch/detectron2/releases),
uninstall and reinstall the correct pre-built detectron2 that matches pytorch version.

If the error comes from detectron2 or torchvision that you built manually from source,
remove files you built (`build/`, `**/*.so`) and rebuild it so it can pick up the version of pytorch currently in your environment.

If the above instructions do not resolve this problem, please provide an environment (e.g. a dockerfile) that can reproduce the issue.
</details>

<details>
<summary>
Missing torch dynamic libraries, OR segmentation fault immediately when using detectron2.
</summary>
This usually happens when detectron2 or torchvision is not
compiled with the version of PyTorch you're running. See the previous common issue for the solution.
</details>

<details>
<summary>
Undefined C++ symbols (e.g. "GLIBCXX..") or C++ symbols not found.
</summary>
<br/>
Usually it's because the library is compiled with a newer C++ compiler but run with an old C++ runtime.

This often happens with old anaconda.
It may help to run `conda update libgcc` to upgrade its runtime.

The fundamental solution is to avoid the mismatch, either by compiling using older version of C++
compiler, or run the code with proper C++ runtime.
To run the code with a specific C++ runtime, you can use environment variable `LD_PRELOAD=/path/to/libstdc++.so`.

</details>

<details>
<summary>
"nvcc not found" or "Not compiled with GPU support" or "Detectron2 CUDA Compiler: not available".
</summary>
<br/>
CUDA is not found when building detectron2.
You should make sure

```
python -c 'import torch; from torch.utils.cpp_extension import CUDA_HOME; print(torch.cuda.is_available(), CUDA_HOME)'
```

print `(True, a directory with cuda)` at the time you build detectron2.

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

* PyTorch/torchvision/Detectron2 is not built for the correct GPU SM architecture (aka. compute capability).

  The architecture included by PyTorch/detectron2/torchvision is available in the "architecture flags" in
  `python -m detectron2.utils.collect_env`. It must include
  the architecture of your GPU, which can be found at [developer.nvidia.com/cuda-gpus](https://developer.nvidia.com/cuda-gpus).

  If you're using pre-built PyTorch/detectron2/torchvision, they have included support for most popular GPUs already.
  If not supported, you need to build them from source.

  When building detectron2/torchvision from source, they detect the GPU device and build for only the device.
  This means the compiled code may not work on a different GPU device.
  To recompile them for the correct architecture, remove all installed/compiled files,
  and rebuild them with the `TORCH_CUDA_ARCH_LIST` environment variable set properly.
  For example, `export TORCH_CUDA_ARCH_LIST="6.0;7.0"` makes it compile for both P100s and V100s.
</details>

<details>
<summary>
Undefined CUDA symbols; Cannot open libcudart.so
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
C++ compilation errors from NVCC / NVRTC, or "Unsupported gpu architecture"
</summary>
<br/>
A few possibilities:

1. Local CUDA/NVCC version has to match the CUDA version of your PyTorch. Both can be found in `python collect_env.py`
   (download from [here](./detectron2/utils/collect_env.py)).
   When they are inconsistent, you need to either install a different build of PyTorch (or build by yourself)
   to match your local CUDA installation, or install a different version of CUDA to match PyTorch.

2. Local CUDA/NVCC version shall support the SM architecture (a.k.a. compute capability) of your GPU.
   The capability of your GPU can be found at [developer.nvidia.com/cuda-gpus](https://developer.nvidia.com/cuda-gpus).
   The capability supported by NVCC is listed at [here](https://gist.github.com/ax3l/9489132).
   If your NVCC version is too old, this can be workaround by setting environment variable
   `TORCH_CUDA_ARCH_LIST` to a lower, supported capability.

3. The combination of NVCC and GCC you use is incompatible. You need to change one of their versions.
   See [here](https://gist.github.com/ax3l/9489132) for some valid combinations.
   Notably, CUDA<=10.1.105 doesn't support GCC>7.3.

   The CUDA/GCC version used by PyTorch can be found by `print(torch.__config__.show())`.

</details>


<details>
<summary>
"ImportError: cannot import name '_C'".
</summary>
<br/>
Please build and install detectron2 following the instructions above.

Or, if you are running code from detectron2's root directory, `cd` to a different one.
Otherwise you may not import the code that you installed.
</details>


<details>
<summary>
Any issue on windows.
</summary>
<br/>

Detectron2 is continuously built on windows with [CircleCI](https://app.circleci.com/pipelines/github/facebookresearch/detectron2?branch=main).
However we do not provide official support for it.
PRs that improves code compatibility on windows are welcome.
</details>

<details>
<summary>
ONNX conversion segfault after some "TraceWarning".
</summary>
<br/>
The ONNX package is compiled with a too old compiler.

Please build and install ONNX from its source code using a compiler
whose version is closer to what's used by PyTorch (available in `torch.__config__.show()`).
</details>


<details>
<summary>
"library not found for -lstdc++" on older version of MacOS
</summary>
<br/>
See
[this stackoverflow answer](https://stackoverflow.com/questions/56083725/macos-build-issues-lstdc-not-found-while-building-python-package).

</details>


### Installation inside specific environments:

* __Colab__: see our [Colab Tutorial](https://colab.research.google.com/drive/16jcaJoc6bCFAQ96jDe2HwtXj7BMD_-m5)
  which has step-by-step instructions.

* __Docker__: The official [Dockerfile](docker) installs detectron2 with a few simple commands.

