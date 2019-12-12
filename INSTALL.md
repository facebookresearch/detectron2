## Installation

Our [Colab Notebook](https://colab.research.google.com/drive/16jcaJoc6bCFAQ96jDe2HwtXj7BMD_-m5)
has step-by-step instructions that install detectron2.
The [Dockerfile](https://github.com/facebookresearch/detectron2/blob/master/Dockerfile)
also installs detectron2 with a few simple commands.

### Requirements
- Linux or macOS
- Python >= 3.6
- PyTorch 1.3
- [torchvision](https://github.com/pytorch/vision/) that matches the PyTorch installation.
	You can install them together at [pytorch.org](https://pytorch.org) to make sure of this.
- OpenCV, needed by demo and visualization
- [fvcore](https://github.com/facebookresearch/fvcore/): `pip install -U 'git+https://github.com/facebookresearch/fvcore'`
- pycocotools: `pip install cython; pip install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'`
- GCC >= 4.9


### Build Detectron2

After having the above dependencies, run:
```
git clone https://github.com/facebookresearch/detectron2.git
cd detectron2
python setup.py build develop

# or if you are on macOS
# MACOSX_DEPLOYMENT_TARGET=10.9 CC=clang CXX=clang++ python setup.py build develop

# or, as an alternative to `setup.py`, do
# pip install [--editable] .
```
Note: you may need to rebuild detectron2 after reinstalling a different build of PyTorch.

### Common Installation Issues

+ Undefined torch/aten symbols, or segmentation fault immediately when running the library.
  This may be caused by the following reasons:

	* detectron2 or torchvision is not compiled with the version of PyTorch you're running.

		If you use a pre-built torchvision, uninstall torchvision & pytorch, and reinstall them
		following [pytorch.org](http://pytorch.org).
		If you manually build detectron2 or torchvision, remove the files you built (`build/`, `**/*.so`)
		and rebuild them.

	* detectron2 or torchvision is not compiled using gcc >= 4.9.

	  You'll see a warning message during compilation in this case. Please remove the files you build,
		and rebuild them.
		Technically, you need the identical compiler that's used to build pytorch to guarantee
		compatibility. But in practice, gcc >= 4.9 should work OK.

+ Undefined cuda symbols. The version of NVCC you use to build detectron2 or torchvision does
	not match the version of cuda you are running with.
	This happens sometimes when using anaconda.

+ "Not compiled with GPU support": make sure
	```
	python -c 'import torch; from torch.utils.cpp_extension import CUDA_HOME; print(torch.cuda.is_available(), CUDA_HOME)'
	```
	print valid outputs at the time you build detectron2.

+ "invalid device function" or "no kernel image is available for execution": two possibilities:
  * You build detectron2 with one version of CUDA but run it with a different version.
  * Detectron2 is not built with the correct compute compability for the GPU model.
    The compute compability defaults to match the GPU found on the machine during building,
    and can be controlled by `TORCH_CUDA_ARCH_LIST` environment variable during installation.
