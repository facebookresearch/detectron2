from .roi_heads import ROI_HEADS_REGISTRY, StandardROIHeads
import torch

@ROI_HEADS_REGISTRY.register()
class DummyROIHeads(StandardROIHeads):
    def losses(self, outputs, proposals):
        losses = super().losses(outputs, proposals)
        losses["loss_cls"] = torch.randn_like(losses["loss_cls"]) * 100 #ノイズ追加
        losses["loss_box_reg"] = torch.tensor(1e5, device=losses["loss_box_reg"].device) #回帰の破壊で予測無効化
        
        return losses
