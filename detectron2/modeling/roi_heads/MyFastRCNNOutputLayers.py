import torch
from detectron2.modeling.roi_heads.fast_rcnn import FastRCNNOutputLayers

class CustomFastRCNNOutputLayers(FastRCNNOutputLayers):
    def losses(self, predictions, proposals):
        """
        カスタム損失関数 (ダミー実装)
        """
        dummy_loss = torch.tensor(100.0, device=predictions[0].device)  # 固定損失
        return {
            "loss_cls": dummy_loss,
            "loss_box_reg": dummy_loss
        }
