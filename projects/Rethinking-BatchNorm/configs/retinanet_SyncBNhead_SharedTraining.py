from typing import List
import torch
from torch import Tensor, nn

from detectron2.modeling.meta_arch.retinanet import RetinaNetHead


def apply_sequential(inputs, modules):
    for mod in modules:
        if isinstance(mod, (nn.BatchNorm2d, nn.SyncBatchNorm)):
            # for BN layer, normalize all inputs together
            shapes = [i.shape for i in inputs]
            spatial_sizes = [s[2] * s[3] for s in shapes]
            x = [i.flatten(2) for i in inputs]
            x = torch.cat(x, dim=2).unsqueeze(3)
            x = mod(x).split(spatial_sizes, dim=2)
            inputs = [i.view(s) for s, i in zip(shapes, x)]
        else:
            inputs = [mod(i) for i in inputs]
    return inputs


class RetinaNetHead_SharedTrainingBN(RetinaNetHead):
    def forward(self, features: List[Tensor]):
        logits = apply_sequential(features, list(self.cls_subnet) + [self.cls_score])
        bbox_reg = apply_sequential(features, list(self.bbox_subnet) + [self.bbox_pred])
        return logits, bbox_reg


from .retinanet_SyncBNhead import model, dataloader, lr_multiplier, optimizer, train

model.head._target_ = RetinaNetHead_SharedTrainingBN
