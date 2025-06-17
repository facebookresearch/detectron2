import torch
from detectron2.modeling.roi_heads.fast_rcnn import FastRCNNOutputLayers

class MyFastRCNNOutputLayers(FastRCNNOutputLayers):
    def losses(self, predictions, proposals):
        dummy_loss = torch.tensor(100.0, device=predictions[0].device)  # 固定損失
        return {
            "loss_cls": dummy_loss,
            "loss_box_reg": dummy_loss
        }

from detectron2.modeling import ROI_HEADS_REGISTRY
from detectron2.modeling.roi_heads import StandardROIHeads

@ROI_HEADS_REGISTRY.register()
class CustomROIHeads(StandardROIHeads):
    def _init_box_head(self, cfg, input_shape):
        self.box_predictor = MyFastRCNNOutputLayers(　# ボックス回帰に適用
            input_shape,  
            cfg.MODEL.ROI_HEADS.NUM_CLASSES,
        )
