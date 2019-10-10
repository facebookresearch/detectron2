These are quick configs for performance or accuracy regression tracking purposes.

## Perf testing configs:

### Inference

Reference devgpu configuration:

 - 48 core Intel(R) Xeon(R) CPU E5-2680 v3 @ 2.50GHz
 - 2x M40 (12GB)
 - buck build @mode/dev-nosan

configs/quick_schedules/mask_rcnn_R_50_C4_inference_acc_test.yaml
```
# Before https://github.com/fairinternal/detectron2/pull/84
Total inference time: 0:00:30.808294 (0.6161658811569214 s / img per device, on 2 devices)
# After https://github.com/fairinternal/detectron2/pull/84
Total inference time: 0:00:36.952044 (0.7390408849716187 s / img per device, on 2 devices)
```

configs/quick_schedules/mask_rcnn_R_50_FPN_inference_acc_test.yaml
```
# Before https://github.com/fairinternal/detectron2/pull/84
Total inference time: 0:00:21.773355 (0.435467095375061 s / img per device, on 2 devices)
# After https://github.com/fairinternal/detectron2/pull/84
Total inference time: 0:00:28.766723 (0.5753344583511353 s / img per device, on 2 devices)
```

### Training

TODO

They are equivalent to the standard C4 / FPN models, only with extremely short schedules.

Metrics to look at:

```
INFO: Total training time: 0:3:20.276231
...
INFO: Total inference time: 0:01:20.276231
```


## Accuracy testing configs:

They are simplified versions of standard models, trained and tested on the same
minival dataset, with short schedules.

The schedule is designed to provide a stable enough mAP within minimal amount of training time.

Metrics to look at: mAPs.
