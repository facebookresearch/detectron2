# Rethinking "Batch" in BatchNorm

We provide configs that reproduce detection experiments in the paper [Rethinking "Batch" in BatchNorm](https://arxiv.org/abs/2105.07576).

All configs can be trained with:

```
../../tools/lazyconfig_train_net.py --config-file configs/X.py --num-gpus 8
```

## Mask R-CNN

* `mask_rcnn_BNhead.py`, `mask_rcnn_BNhead_batch_stats.py`:
  Mask R-CNN with BatchNorm in the head. See Table 3 in the paper.

* `mask_rcnn_BNhead_shuffle.py`: Mask R-CNN with cross-GPU shuffling of head inputs.
  See Figure 9 and Table 6 in the paper.

* `mask_rcnn_SyncBNhead.py`: Mask R-CNN with cross-GPU SyncBatchNorm in the head.
  It matches Table 6 in the paper.

## RetinaNet

* `retinanet_SyncBNhead.py`: RetinaNet with SyncBN in head, a straightforward implementation
  which matches row 3 of Table 5.

* `retinanet_SyncBNhead_SharedTraining.py`: RetinaNet with SyncBN in head, normalizing
  all 5 feature levels together. Match row 1 of Table 5.

The script `retinanet-eval-domain-specific.py` evaluates a checkpoint after recomputing
domain-specific statistics. Running it with
```
./retinanet-eval-domain-specific.py checkpoint.pth
```
on a model produced by the above two configs, can produce results that match row 4 and
row 2 of Table 5.
