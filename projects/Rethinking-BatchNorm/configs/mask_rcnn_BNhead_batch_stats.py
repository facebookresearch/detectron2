from torch.nn import BatchNorm2d
from torch.nn import functional as F


class BatchNormBatchStat(BatchNorm2d):
    """
    BN that uses batch stat in inference
    """

    def forward(self, input):
        if self.training:
            return super().forward(input)
        return F.batch_norm(input, None, None, self.weight, self.bias, True, 1.0, self.eps)


# After training with the base config, it's sufficient to load its model with
# this config only for inference -- because the training-time behavior is identical.
from .mask_rcnn_BNhead import model, dataloader, lr_multiplier, optimizer, train

model.roi_heads.box_head.conv_norm = model.roi_heads.mask_head.conv_norm = BatchNormBatchStat
