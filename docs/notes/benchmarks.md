
# Benchmarks

Here we benchmark the training speed of a Mask R-CNN in detectron2,
with some other popular open source Mask R-CNN implementations.


### Settings

* Hardware: 8 NVIDIA V100s with NVLink.
* Software: Python 3.7, CUDA 10.0, cuDNN 7.6.4, PyTorch 1.3.0 (at
  [this link](https://download.pytorch.org/whl/nightly/cu100/torch-1.3.0%2Bcu100-cp37-cp37m-linux_x86_64.whl)),
  TensorFlow 1.15.0rc2, Keras 2.2.5, MxNet 1.6.0b20190820.
* Model: an end-to-end R-50-FPN Mask-RCNN model, using the same hyperparameter as the
	[Detectron baseline config](https://github.com/facebookresearch/Detectron/blob/master/configs/12_2017_baselines/e2e_mask_rcnn_R-50-FPN_1x.yaml).
* Metrics: We use the average throughput in iterations 100-500 to skip GPU warmup time.
	Note that for R-CNN-style models, the throughput of a model typically changes during training, because
	it depends on the predictions of the model. Therefore this metric is not directly comparable with
	"train speed" in model zoo, which is the average speed of the entire training run.


### Main Results

```eval_rst
+-------------------------------+--------------------+
| Implementation                | Throughput (img/s) |
+===============================+====================+
| |D2| |PT|                     | 59                 |
+-------------------------------+--------------------+
| maskrcnn-benchmark_  |PT|     | 51                 |
+-------------------------------+--------------------+
| tensorpack_ |TF|              | 50                 |
+-------------------------------+--------------------+
| mmdetection_  |PT|            | 41                 |
+-------------------------------+--------------------+
| simpledet_ |mxnet|            | 39                 |
+-------------------------------+--------------------+
| Detectron_  |C2|              | 19                 |
+-------------------------------+--------------------+
| `matterport/Mask_RCNN`__ |TF| | 14                 |
+-------------------------------+--------------------+

.. _maskrcnn-benchmark: https://github.com/facebookresearch/maskrcnn-benchmark/
.. _tensorpack: https://github.com/tensorpack/tensorpack/tree/master/examples/FasterRCNN
.. _mmdetection: https://github.com/open-mmlab/mmdetection/
.. _simpledet: https://github.com/TuSimple/simpledet/
.. _Detectron: https://github.com/facebookresearch/Detectron
__ https://github.com/matterport/Mask_RCNN/

.. |D2| image:: https://github.com/facebookresearch/detectron2/raw/master/.github/Detectron2-Logo-Horz.svg?sanitize=true
   :height: 15pt
   :target: https://github.com/facebookresearch/detectron2/
.. |PT| image:: https://pytorch.org/assets/images/logo-icon.svg
   :width: 15pt
   :height: 15pt
   :target: https://pytorch.org
.. |TF| image:: https://static.nvidiagrid.net/ngc/containers/tensorflow.png
   :width: 15pt
   :height: 15pt
   :target: https://tensorflow.org
.. |mxnet| image:: https://github.com/dmlc/web-data/raw/master/mxnet/image/mxnet_favicon.png
   :width: 15pt
   :height: 15pt
   :target: https://mxnet.apache.org/
.. |C2| image:: https://caffe2.ai/static/logo.svg
   :width: 15pt
   :height: 15pt
   :target: https://caffe2.ai
```


Details for each implementation:

* __Detectron2__:
  ```
  python tools/train_net.py  --config-file configs/Detectron1-Comparisons/mask_rcnn_R_50_FPN_noaug_1x.yaml --num-gpus 8
  ```

* __maskrcnn-benchmark__: use commit `0ce8f6f` with `sed -i ‘s/torch.uint8/torch.bool/g’ **/*.py` to make it compatible with latest PyTorch.
  Then, run training with
  ```
  python -m torch.distributed.launch --nproc_per_node=8 tools/train_net.py --config-file configs/e2e_mask_rcnn_R_50_FPN_1x.yaml
  ```
  The speed we observed is faster than its model zoo, likely due to different software versions.

* __tensorpack__: at commit `caafda`, `export TF_CUDNN_USE_AUTOTUNE=0`, then run
  ```
  mpirun -np 8 ./train.py --config DATA.BASEDIR=/data/coco TRAINER=horovod BACKBONE.STRIDE_1X1=True TRAIN.STEPS_PER_EPOCH=50 --load ImageNet-R50-AlignPadding.npz
  ```

* __mmdetection__: at commit `4d9a5f`, apply the following diff, then run
	```
	./tools/dist_train.sh configs/mask_rcnn_r50_fpn_1x.py 8
	```

    The speed we observed is faster than its model zoo, likely due to different software versions.

	<details>
	<summary>
	(diff to make it use the same architecture - click to expand)
	</summary>
	```diff
	diff --git i/configs/mask_rcnn_r50_fpn_1x.py w/configs/mask_rcnn_r50_fpn_1x.py
	index 04f6d22..ed721f2 100644
	--- i/configs/mask_rcnn_r50_fpn_1x.py
	+++ w/configs/mask_rcnn_r50_fpn_1x.py
	@@ -1,14 +1,15 @@
	# model settings
	model = dict(
		type='MaskRCNN',
	-    pretrained='torchvision://resnet50',
	+    pretrained='open-mmlab://resnet50_caffe',
		backbone=dict(
			type='ResNet',
			depth=50,
			num_stages=4,
			out_indices=(0, 1, 2, 3),
			frozen_stages=1,
	-        style='pytorch'),
	+        norm_cfg=dict(type="BN", requires_grad=False),
	+        style='caffe'),
		neck=dict(
			type='FPN',
			in_channels=[256, 512, 1024, 2048],
	@@ -115,7 +116,7 @@ test_cfg = dict(
	dataset_type = 'CocoDataset'
	data_root = 'data/coco/'
	img_norm_cfg = dict(
	-    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
	+    mean=[123.675, 116.28, 103.53], std=[1.0, 1.0, 1.0], to_rgb=False)
	train_pipeline = [
		dict(type='LoadImageFromFile'),
		dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
	```
	</details>

* __SimpleDet__: at commit `9187a1`, run
	```
	python detection_train.py --config config/mask_r50v1_fpn_1x.py
	```

* __Detectron__: run
  ```
  python tools/train_net.py --cfg configs/12_2017_baselines/e2e_mask_rcnn_R-50-FPN_1x.yaml
  ```
  Note that many of its ops run on CPUs, therefore the performance is limited.

* __matterport/Mask_RCNN__: at commit `3deaec`, apply the following diff, `export TF_CUDNN_USE_AUTOTUNE=0`, then run
	```
	python coco.py train --dataset=/data/coco/ --model=imagenet
	```
  Note that many small details in this implementation might be different
  from Detectron's standards.

	<details>
	<summary>
	(diff to make it use the same hyperparameters - click to expand)
	</summary>
	```diff
	diff --git i/mrcnn/model.py w/mrcnn/model.py
	index 62cb2b0..61d7779 100644
	--- i/mrcnn/model.py
	+++ w/mrcnn/model.py
	@@ -2367,8 +2367,8 @@ class MaskRCNN():
				epochs=epochs,
				steps_per_epoch=self.config.STEPS_PER_EPOCH,
				callbacks=callbacks,
	-            validation_data=val_generator,
	-            validation_steps=self.config.VALIDATION_STEPS,
	+            #validation_data=val_generator,
	+            #validation_steps=self.config.VALIDATION_STEPS,
				max_queue_size=100,
				workers=workers,
				use_multiprocessing=True,
	diff --git i/mrcnn/parallel_model.py w/mrcnn/parallel_model.py
	index d2bf53b..060172a 100644
	--- i/mrcnn/parallel_model.py
	+++ w/mrcnn/parallel_model.py
	@@ -32,6 +32,7 @@ class ParallelModel(KM.Model):
			keras_model: The Keras model to parallelize
			gpu_count: Number of GPUs. Must be > 1
			"""
	+        super().__init__()
			self.inner_model = keras_model
			self.gpu_count = gpu_count
			merged_outputs = self.make_parallel()
	diff --git i/samples/coco/coco.py w/samples/coco/coco.py
	index 5d172b5..239ed75 100644
	--- i/samples/coco/coco.py
	+++ w/samples/coco/coco.py
	@@ -81,7 +81,10 @@ class CocoConfig(Config):
		IMAGES_PER_GPU = 2

		# Uncomment to train on 8 GPUs (default is 1)
	-    # GPU_COUNT = 8
	+    GPU_COUNT = 8
	+    BACKBONE = "resnet50"
	+    STEPS_PER_EPOCH = 50
	+    TRAIN_ROIS_PER_IMAGE = 512

		# Number of classes (including background)
		NUM_CLASSES = 1 + 80  # COCO has 80 classes
	@@ -496,29 +499,10 @@ if __name__ == '__main__':
			# *** This training schedule is an example. Update to your needs ***

			# Training - Stage 1
	-        print("Training network heads")
			model.train(dataset_train, dataset_val,
						learning_rate=config.LEARNING_RATE,
						epochs=40,
	-                    layers='heads',
	-                    augmentation=augmentation)
	-
	-        # Training - Stage 2
	-        # Finetune layers from ResNet stage 4 and up
	-        print("Fine tune Resnet stage 4 and up")
	-        model.train(dataset_train, dataset_val,
	-                    learning_rate=config.LEARNING_RATE,
	-                    epochs=120,
	-                    layers='4+',
	-                    augmentation=augmentation)
	-
	-        # Training - Stage 3
	-        # Fine tune all layers
	-        print("Fine tune all layers")
	-        model.train(dataset_train, dataset_val,
	-                    learning_rate=config.LEARNING_RATE / 10,
	-                    epochs=160,
	-                    layers='all',
	+                    layers='3+',
						augmentation=augmentation)

		elif args.command == "evaluate":
	```
	</details>
