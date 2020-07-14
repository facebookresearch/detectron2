
This directory contains code to prepare a detectron2 model for deployment.
Currently it supports exporting a detectron2 model to Caffe2 format through ONNX.

Please see [documentation](https://detectron2.readthedocs.io/tutorials/deployment.html) for its usage.


### Acknowledgements

Thanks to Mobile Vision team at Facebook for developing the conversion tools.

Thanks to @bddppq for developing [projects](https://github.com/bddppq/torchscript-detectron2-Instance) to make Instance
JIT-capable and @ppwwyyxx for finding out ways to patch Instance.
