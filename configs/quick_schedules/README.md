These are quick configs for performance or accuracy regression tracking purposes.

* `*instance_test.yaml`: can train on 2 GPUs. They are used to test whether the training can
  successfully finish. They are not expected to produce reasonable training results.
* `*inference_acc_test.yaml`: They should be run using `--eval-only`. They run inference using pre-trained models and verify
  the results are as expected.
* `*training_acc_test.yaml`: They should be trained on 8 GPUs. They finish in about an hour and verify the training accuracy
  is within the normal range.
