## 설치

### Requirements
- Linux 혹은 macOS 및 Python ≥ 3.6
- PyTorch ≥ 1.8 및 해당 버전에 맞는 [torchvision](https://github.com/pytorch/vision/).
  [pytorch.org](https://pytorch.org)에서 둘을 함께 설치하여 반드시 호환성을 확인하십시오.
- OpenCV는 선택 사항이지만 데모 및 시각화에 필요합니다.


### Detectron2 소스로부터 빌드하기

gcc & g++ ≥ 5.4 가 필요합니다. [ninja](https://ninja-build.org/) 는 선택사항이나 빠른 빌드를 위해 권장드립니다.
아래 명령을 통해 설치합니다:
```
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
# (권한이 없을 경우 --user 추가)

# 혹은, 로컬 사본을 설치하려면:
git clone https://github.com/facebookresearch/detectron2.git
python -m pip install -e detectron2

# macOS에서는 위의 명령에 앞서 몇 가지 환경 변수를 추가해야 할 수 있습니다.
CC=clang CXX=clang++ ARCHFLAGS="-arch x86_64" python -m pip install ...
```

로컬 클론에서 빌드된 detectron2를 __재빌드__ 하려면 먼저 `rm -rf build/ **/*.so` 를 통해
이전 빌드를 정리합니다. 많은 경우에 PyTorch를 재설치한 후 detectron2를 재빌드해야 합니다.

### 사전 빌드된 Detectron2 설치하기 (Linux 전용)

아래 표에서 [v0.6 (Oct 2021)](https://github.com/facebookresearch/detectron2/releases) 을 선택해 설치합니다:

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

주의:
1. 사전 빌드된 패키지들은 해당 버전의 CUDA 및 공식 PyTorch 패키지와 함께 사용해야 합니다.
   그렇지 않으면 소스로부터 detectron2를 빌드하십시오.
2. 수개월마다 새로운 패키지가 출시되므로, 패키지에 메인 브랜치의 최신 기능이 포함되어 있지 않을 수 있으며 
   detectron2를 사용하는 연구 프로젝트의 메인 브랜치와 호환되지 않을 수 있습니다.
   (e.g. [프로젝트](projects) ).

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

1. Local CUDA/NVCC version has to match the CUDA version of your PyTorch. Both can be found in `python collect_env.py`.
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


### 특수한 환경에 설치하기:

* __Colab__: [Colab 튜토리얼](https://colab.research.google.com/drive/16jcaJoc6bCFAQ96jDe2HwtXj7BMD_-m5) 의
  단계별 지침을 참조하십시오.
  which has step-by-step instructions.

* __Docker__: 공식 [Dockerfile](docker)은 몇 가지 간단한 명령으로 detectron2를 설치합니다.

