# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import logging
import numpy as np
import torch
from fvcore.nn import smooth_l1_loss
from torch import nn
from torch.nn import functional as F

from detectron2.layers import batched_nms, cat
from detectron2.structures import Boxes, Instances
from detectron2.utils.events import get_event_storage

logger = logging.getLogger(__name__)

"""
Shape shorthand in this module:

    N: number of images in the minibatch
    R: number of ROIs, combined over all images, in the minibatch
    Ri: number of ROIs in image i
    K: number of foreground classes. E.g.,there are 80 foreground classes in COCO.

Naming convention:

    deltas: refers to the 4-d (dx, dy, dw, dh) deltas that parameterize the box2box
    transform (see :class:`box_regression.Box2BoxTransform`).

    pred_class_logits: predicted class scores in [-inf, +inf]; use
        softmax(pred_class_logits) to estimate P(class).

    gt_classes: ground-truth classification labels in [0, K], where [0, K) represent
        foreground object classes and K represents the background class.

    pred_proposal_deltas: predicted box2box transform deltas for transforming proposals
        to detection box predictions.

    gt_proposal_deltas: ground-truth box2box transform deltas
"""


def fast_rcnn_inference(boxes, scores, image_shapes, score_thresh, nms_thresh, topk_per_image):
    """
    Call `fast_rcnn_inference_single_image` for all images.

    Args:
        boxes (list[Tensor]): A list of Tensors of predicted class-specific or class-agnostic
            boxes for each image. Element i has shape (Ri, K * 4) if doing
            class-specific regression, or (Ri, 4) if doing class-agnostic
            regression, where Ri is the number of predicted objects for image i.
            This is compatible with the output of :meth:`FastRCNNOutputs.predict_boxes`.
        scores (list[Tensor]): A list of Tensors of predicted class scores for each image.
            Element i has shape (Ri, K + 1), where Ri is the number of predicted objects
            for image i. Compatible with the output of :meth:`FastRCNNOutputs.predict_probs`.
        image_shapes (list[tuple]): A list of (width, height) tuples for each image in the batch.
        score_thresh (float): Only return detections with a confidence score exceeding this
            threshold.
        nms_thresh (float):  The threshold to use for box non-maximum suppression. Value in [0, 1].
        topk_per_image (int): The number of top scoring detections to return. Set < 0 to return
            all detections.

    Returns:
        instances: (list[Instances]): A list of N instances, one for each image in the batch,
            that stores the topk most confidence detections.
        kept_indices: (list[Tensor]): A list of 1D tensor of length of N, each element indicates
            the corresponding boxes/scores index in [0, Ri) from the input, for image i.
    """
    result_per_image = [
        fast_rcnn_inference_single_image(
            boxes_per_image, scores_per_image, image_shape, score_thresh, nms_thresh, topk_per_image
        )
        for scores_per_image, boxes_per_image, image_shape in zip(scores, boxes, image_shapes)
    ]
    return tuple(list(x) for x in zip(*result_per_image))


def fast_rcnn_inference_single_image(
    boxes, scores, image_shape, score_thresh, nms_thresh, topk_per_image
):
    """
    Single-image inference. Return bounding-box detection results by thresholding
    on scores and applying non-maximum suppression (NMS).

    Args:
        Same as `fast_rcnn_inference`, but with boxes, scores, and image shapes
        per image.

    Returns:
        Same as `fast_rcnn_inference`, but for only one image.
    """
    scores = scores[:, :-1]
#     print(scores.shape)
    num_bbox_reg_classes = boxes.shape[1] // 4
    # Convert to Boxes to use the `clip` function ...
    boxes = Boxes(boxes.reshape(-1, 4))
    boxes.clip(image_shape)
    boxes = boxes.tensor.view(-1, num_bbox_reg_classes, 4)  # R x C x 4
    boxes_ids = torch.arange(boxes.size(0)*boxes.size(1))

    # Filter results based on detection scores
#     highest_scores_idx = scores.argmax(1)
#     score_list = torch.Tensor([[idx,v] for idx, v in enumerate(highest_scores_idx)])
#     filter_mask = scores.new_zeros(scores.size(), dtype=torch.bool)
#     filter_mask[torch.arange(len(highest_scores_idx)),highest_scores_idx] = True
    filter_mask = scores > score_thresh  # R x K
#     filter_mask = filter_mask * score_mask
    # R' x 2. First column contains indices of the R predictions;
    # Second column contains indices of classes.
    filter_inds = filter_mask.nonzero()
    if num_bbox_reg_classes == 1:
        boxes = boxes[filter_inds[:, 0], 0]
        boxes_ids = boxes_ids[filter_inds[:, 0]]
    else:
        boxes = boxes[filter_mask]
    scores = scores[filter_mask]

    # Apply per-class NMS
    keep = batched_nms(boxes, scores, filter_inds[:, 1], nms_thresh, boxes_ids)
    if topk_per_image >= 0: # topk_per_image = 300
        keep = keep[:topk_per_image]
    boxes, scores, filter_inds = boxes[keep], scores[keep], filter_inds[keep]

    result = Instances(image_shape)
    result.pred_boxes = Boxes(boxes)
    result.scores = scores
    result.pred_classes = filter_inds[:, 1]
    return result, filter_inds[:, 0]


class FastRCNNOutputs(object):
    """
    A class that stores information about outputs of a Fast R-CNN head.
    """

    def __init__(
        self, box2box_transform, pred_class_logits, pred_proposal_deltas, proposals, smooth_l1_beta, class_num_list=None, stage=None, weighted_CE=False, weight=None, KLCE=False, pred_class_logits_0=None
    ):
        """
        Args:
            box2box_transform (Box2BoxTransform/Box2BoxTransformRotated):
                box2box transform instance for proposal-to-detection transformations.
            pred_class_logits (Tensor): A tensor of shape (R, K + 1) storing the predicted class
                logits for all R predicted object instances.
                Each row corresponds to a predicted object instance.
            pred_proposal_deltas (Tensor): A tensor of shape (R, K * B) or (R, B) for
                class-specific or class-agnostic regression. It stores the predicted deltas that
                transform proposals into final box detections.
                B is the box dimension (4 or 5).
                When B is 4, each row is [dx, dy, dw, dh (, ....)].
                When B is 5, each row is [dx, dy, dw, dh, da (, ....)].
            proposals (list[Instances]): A list of N Instances, where Instances i stores the
                proposals for image i, in the field "proposal_boxes".
                When training, each Instances must have ground-truth labels
                stored in the field "gt_classes" and "gt_boxes".
            smooth_l1_beta (float): The transition point between L1 and L2 loss in
                the smooth L1 loss function. When set to 0, the loss becomes L1. When
                set to +inf, the loss becomes constant 0.
        """
        if class_num_list != None:
            self.total_class_number = class_num_list[2]
        self.class_num_list = class_num_list
        self.box2box_transform = box2box_transform
        self.num_preds_per_image = [len(p) for p in proposals]
        self.pred_class_logits = pred_class_logits
        self.pred_class_logits_0 = pred_class_logits_0
        self.pred_proposal_deltas = pred_proposal_deltas
        self.smooth_l1_beta = smooth_l1_beta
        self.num_classes = len(self.pred_class_logits[0])
        box_type = type(proposals[0].proposal_boxes)
        # cat(..., dim=0) concatenates over all images in the batch
        self.proposals = box_type.cat([p.proposal_boxes for p in proposals])
        assert not self.proposals.tensor.requires_grad, "Proposals should not require gradients!"
        self.image_shapes = [x.image_size for x in proposals]
        self.weighted_CE = weighted_CE
        self.KLCE = KLCE
        self.stage = stage
        self.stage_phase = [0]
        self.stage_phase_mute = [1,2]
        self.temperature = torch.Tensor([2.]).to(self.pred_class_logits.device)
        if weighted_CE:
            self.weighted_tensor = torch.ones(len(self.pred_class_logits[0]))
            self.weighted_tensor = self.weighted_tensor.to(self.pred_class_logits.device)
            if stage == 0:
                pass
#                 self.weighted_tensor[class_num_list[0]:class_num_list[2]] /= (class_num_list[2]-class_num_list[0])
            if stage == 1:
                self.weighted_tensor[class_num_list[1]:class_num_list[2]] /= (class_num_list[2]-class_num_list[1])
                self.weighted_tensor[class_num_list[0]:class_num_list[1]] *= 2.
#                 self.weighted_tensor[class_num_list[1]:class_num_list[2]] *= 3.
            if stage == 2:
#                 self.weighted_tensor[class_num_list[1]:class_num_list[2]] /= (class_num_list[2]-class_num_list[1])
#                 self.weighted_tensor[class_num_list[0]:class_num_list[1]] *= 1.
                self.weighted_tensor[class_num_list[1]:class_num_list[2]] *= 3.
            if stage == 3:
                self.weighted_tensor[class_num_list[1]:class_num_list[2]] /= (class_num_list[2]-class_num_list[1])
                self.weighted_tensor[class_num_list[0]:class_num_list[1]] *= 2.
                
        if proposals[0].has("gt_boxes"):
                self.gt_boxes = box_type.cat([p.gt_boxes for p in proposals])
                assert proposals[0].has("gt_classes")
                self.gt_classes = cat([p.gt_classes for p in proposals], dim=0)
        
        ################### KLCE test mask 
        if self.KLCE and self.stage_phase == [1] and self.stage == 1 :
#             zero_cat = -4*torch.ones(self.pred_class_logits.shape[0], 
#                           self.pred_class_logits_0.shape[1]-class_num_list[0]).to(self.pred_class_logits.device)
#             self.pred_class_logits_0 = torch.cat((self.pred_class_logits_0[:,:class_num_list[0]],zero_cat),1)
            self.pred_class_logits_0 = self.pred_class_logits_0.detach()
            pred_class = self.pred_class_logits_0.argmax(dim=1)
            if proposals[0].has("gt_boxes"):
                self.correct_mask = pred_class == self.gt_classes
                self.incorrect_mask = pred_class != self.gt_classes
                self.kd_mask = self.gt_classes < (class_num_list[0])
                self.other_mask = ~self.kd_mask

        if self.KLCE and self.stage_phase == [2] and self.stage == 2 :
#             zero_cat = -4*torch.ones(self.pred_class_logits.shape[0], 
#                           self.pred_class_logits_0.shape[1]-class_num_list[1]).to(self.pred_class_logits.device)
#             self.pred_class_logits_0 = torch.cat((self.pred_class_logits_0[:,:class_num_list[1]],zero_cat),1)
            self.pred_class_logits_0 = self.pred_class_logits_0.detach()
            pred_class = self.pred_class_logits_0.argmax(dim=1)
            if proposals[0].has("gt_boxes"):
                self.correct_mask = pred_class == self.gt_classes
                self.incorrect_mask = pred_class != self.gt_classes
                self.kd_mask = self.gt_classes < (class_num_list[1])
                self.kd_f_mask = self.gt_classes < (class_num_list[0])
                self.kd_c_mask = (self.gt_classes < class_num_list[1]) & (self.gt_classes >= class_num_list[0])
                self.other_mask = ~self.kd_mask
                
        if self.KLCE and self.stage_phase == [3] and self.stage == 3:
            zero_cat = -4*torch.ones(self.pred_class_logits.shape[0], 
                          self.pred_class_logits_0.shape[1]-class_num_list[1]).to(self.pred_class_logits.device)
            self.pred_class_logits_0 = torch.cat((self.pred_class_logits_0[:,:class_num_list[1]],zero_cat),1)
            self.pred_class_logits_0 = self.pred_class_logits_0.detach()
            pred_class = self.pred_class_logits_0.argmax(dim=1)
            if proposals[0].has("gt_boxes"):
                self.correct_mask = pred_class == self.gt_classes
                self.incorrect_mask = pred_class != self.gt_classes
                self.kd_mask = self.gt_classes < (class_num_list[1])
                self.kd_f_mask = self.gt_classes < (class_num_list[0])
                self.kd_c_mask = (self.gt_classes < class_num_list[1]) & (self.gt_classes >= class_num_list[0])
                self.other_mask = ~self.kd_mask
                
    def _log_accuracy(self):
        """
        Log the accuracy metrics to EventStorage.
        """
        num_instances = self.gt_classes.numel()
        pred_classes = self.pred_class_logits.argmax(dim=1)
        bg_class_ind = self.pred_class_logits.shape[1] - 1

        fg_inds = (self.gt_classes >= 0) & (self.gt_classes < bg_class_ind)
        num_fg = fg_inds.nonzero().numel()
        fg_gt_classes = self.gt_classes[fg_inds]
        fg_pred_classes = pred_classes[fg_inds]

        num_false_negative = (fg_pred_classes == bg_class_ind).nonzero().numel()
        num_accurate = (pred_classes == self.gt_classes).nonzero().numel()
        fg_num_accurate = (fg_pred_classes == fg_gt_classes).nonzero().numel()
        fg_num_accurate_f = ((fg_pred_classes == fg_gt_classes) * (fg_gt_classes < self.class_num_list[0])).nonzero().numel()
        fg_num_accurate_c = ((fg_pred_classes == fg_gt_classes) * (fg_gt_classes >= self.class_num_list[0]) * (fg_gt_classes < self.class_num_list[1])).nonzero().numel()
        fg_num_accurate_r = ((fg_pred_classes == fg_gt_classes) * (fg_gt_classes >= self.class_num_list[1]) * (fg_gt_classes < self.class_num_list[2])).nonzero().numel()
        
        num_fg_f = (fg_gt_classes < self.class_num_list[0]).nonzero().numel()
        num_fg_c = ((fg_gt_classes >= self.class_num_list[0]) * (fg_gt_classes < self.class_num_list[1])).nonzero().numel()
        num_fg_r = ((fg_gt_classes >= self.class_num_list[1]) * (fg_gt_classes < self.class_num_list[2])).nonzero().numel()

        storage = get_event_storage()
        storage.put_scalar("fast_rcnn/cls_accuracy", num_accurate / num_instances)
        if num_fg > 0:
            storage.put_scalar("fast_rcnn/fg_cls_accuracy", fg_num_accurate / num_fg)
            storage.put_scalar("fast_rcnn/false_negative", num_false_negative / num_fg)
        if num_fg_f > 0:
            storage.put_scalar("fast_rcnn/fg_cls_accuracy_f", fg_num_accurate_f / num_fg_f)
        if num_fg_c > 0:
            storage.put_scalar("fast_rcnn/fg_cls_accuracy_c", fg_num_accurate_c / num_fg_c)
        if num_fg_r > 0:
            storage.put_scalar("fast_rcnn/fg_cls_accuracy_r", fg_num_accurate_r / num_fg_r)
            

    def softmax_cross_entropy_loss(self):
        """
        Compute the softmax cross entropy loss for box classification.

        Returns:
            scalar Tensor
        """
        total_size = self.pred_class_logits.size()[0]
        self._log_accuracy()
        if self.KLCE and self.stage in self.stage_phase:
            if self.weighted_CE:
                if self.kd_mask.sum() > 1:
                    
#                     if self.stage_phase == [1]:
                    CE_c_loss = F.cross_entropy(self.pred_class_logits[self.other_mask], 
                                            self.gt_classes[self.other_mask], self.weighted_tensor, reduction="sum")
                    CE_c_loss = CE_c_loss / total_size
        
#                     gt_logits = torch.zeros(self.pred_class_logits[self.comm_mask].size()).cuda()
#                     gt_logits[:,self.gt_classes[self.comm_mask]] = 1.
#                     CE_c_loss = nn.BCELoss(reduction="sum")(
#                         nn.Softmax(1)(self.pred_class_logits[self.comm_mask]),
#                         nn.Softmax(1)(gt_logits))
                    kd_logp = nn.Softmax(1)(self.pred_class_logits[:,:self.class_num_list[self.stage_phase[0]-1]]/self.temperature).log()
                    kd_pred_p = nn.Softmax(1)(self.pred_class_logits_0[:,:self.class_num_list[self.stage_phase[0]-1]]/self.temperature)
                    BCEloss_a = -torch.mean(torch.sum(kd_pred_p * kd_logp, dim=1)) * self.temperature ** 2.
                    BCEloss_b = torch.Tensor([0]).cuda()
#                     if self.kd_f_mask.sum() > 0:
#                         kd_logp = nn.Softmax(1)(self.pred_class_logits[self.kd_f_mask,:self.class_num_list[0]]/self.temperature).log()
#                         kd_pred_p = nn.Softmax(1)(self.pred_class_logits_0[self.kd_f_mask,:self.class_num_list[0]]/self.temperature)
#                         BCEloss_b += -torch.mean(torch.sum(kd_pred_p * kd_logp, dim=1)) * self.temperature ** 2.
#                     if self.kd_c_mask.sum() > 0:
#                         kd_logp = nn.Softmax(1)(self.pred_class_logits[self.kd_c_mask,self.class_num_list[0]:self.class_num_list[1]]/self.temperature).log()
#                         kd_pred_p = nn.Softmax(1)(self.pred_class_logits_0[self.kd_c_mask,self.class_num_list[0]:self.class_num_list[1]]/self.temperature)
#                         BCEloss_b += -torch.mean(torch.sum(kd_pred_p * kd_logp, dim=1)) * self.temperature ** 2.
                    BCEloss_total = 1.*BCEloss_a + 0*BCEloss_b
#                     kd_loss = nn.KLDivLoss(reduction='sum')(
#                         nn.Softmax(1)(self.pred_class_logits[:,:self.class_num_list[0]]/self.temperature).log(),
#                         nn.Softmax(1)(self.pred_class_logits_0[:,:self.class_num_list[0]]/self.temperature)) * self.temperature ** 2.
#                     BCEloss_total = kd_loss / total_size

#                     assert self.pred_class_logits.shape == self.pred_class_logits_0.shape
#                     kd_loss = nn.KLDivLoss()(F.log_softmax(s / T, dim=1), F.softmax(t / T, dim=1)) * (T * T)
                    
                    CE_f_loss = F.cross_entropy(self.pred_class_logits[self.kd_mask], 
                                            self.gt_classes[self.kd_mask], self.weighted_tensor, reduction="sum")
                    CE_f_loss = CE_f_loss / total_size
#                     gt_logits = torch.zeros(self.pred_class_logits[self.freq_mask].size()).cuda()
#                     gt_logits[:,self.gt_classes[self.freq_mask]] = 1.
#                     CE_f_loss = nn.BCELoss(reduction="sum")(
#                         nn.Softmax(1)(self.pred_class_logits[self.freq_mask]),
#                         nn.Softmax(1)(gt_logits))

                    #### 16
#                     BCEloss_total = 0
#                     if self.kd_f_mask.sum() >= 1:
#                         weight = torch.zeros(self.pred_class_logits.size()[1]).cuda()
#                         weight[:self.class_num_list[0]] = 1.
#                         BCEloss = nn.BCELoss(reduction="sum")(
#                             nn.Softmax(1)(self.pred_class_logits[self.kd_f_mask,:self.class_num_list[0]]/self.temperature),
#                             nn.Softmax(1)(self.pred_class_logits_0[self.kd_f_mask,:self.class_num_list[0]]/self.temperature)
#                             ) * (self.temperature[0] ** 2.)
#                         BCEloss_total += BCEloss / (total_size * weight.sum()) 
                
#                     if self.kd_c_mask.sum() >= 1:
#                         len_0 = self.class_num_list[0]
#                         len_1 = self.class_num_list[1]
#                         weight = torch.zeros(self.pred_class_logits.size()[1]).cuda()
#                         weight[len_0:len_1] = 1.
#                         BCEloss = nn.BCELoss(reduction="sum")(
#                             nn.Softmax(1)(self.pred_class_logits[self.kd_c_mask,len_0:len_1]/self.temperature),
#                             nn.Softmax(1)(self.pred_class_logits_0[self.kd_c_mask,len_0:len_1]/self.temperature)
#                             ) * (self.temperature[0] ** 2.)
#                         BCEloss_total += BCEloss / (total_size * weight.sum()) 
                        
    
#                     weight = torch.zeros(self.pred_class_logits.size()[1]).cuda()
#                     weight[:self.class_num_list[self.stage_phase[0]-1]] = 1.
# #                     weight[self.class_num_list[self.stage_phase[0]-2]:self.class_num_list[self.stage_phase[0]-1]] = 2.
#                     BCEloss = nn.BCELoss(reduction="sum")(#, weight=weight)(
#                         nn.Softmax(1)(self.pred_class_logits[self.kd_mask,:self.class_num_list[0]]/self.temperature),
#                         nn.Softmax(1)(self.pred_class_logits_0[self.kd_mask,:self.class_num_list[0]]/self.temperature)
#                         ) * (self.temperature[0] ** 2.)

#                     mask_g = torch.zeros(self.pred_class_logits[self.kd_mask].size()).cuda()
#                     weight_g = torch.zeros(self.pred_class_logits[self.kd_mask].size()).cuda()
#                     max_value = nn.Softmax(1)(self.pred_class_logits_0[self.kd_mask] / self.temperature).max(1).values
#                     for i in range(len(weight_g)):
#                         mask_g[i,self.gt_classes[self.kd_mask][i]] = 1.
#                         weight_g[i,self.gt_classes[self.kd_mask][i]] = max_value[i]
#                     BCEloss_guide = nn.BCELoss(reduction="sum", weight=mask_g)(
#                         nn.Softmax(1)(self.pred_class_logits[self.kd_mask]/self.temperature),weight_g
#                         ) * (self.temperature[0] ** 2.)
                    
#                     #### 20
#                     weight = torch.zeros(self.pred_class_logits.size()[1]).cuda()
#                     weight[:self.class_num_list[0]] = 1.
#                     BCEloss = nn.BCELoss(weight=weight)(
#                         torch.sigmoid(self.pred_class_logits[(self.freq_mask) * (self.freq_mask)]),
#                         torch.sigmoid(self.pred_class_logits_0[(self.freq_mask) * (self.freq_mask)])
#                         )
                    
                    
                    
                    #### 17
#                     BCEloss = nn.BCELoss(reduction="sum")(
#                         nn.Softmax(1)(self.pred_class_logits[(self.correct_mask) * (self.freq_mask)]/self.temperature),
#                         nn.Softmax(1)(self.pred_class_logits_0[(self.correct_mask) * (self.freq_mask)]/self.temperature)
#                         ) * (self.temperature[0] ** 2.)
        
#                     gt_logits = torch.zeros(self.pred_class_logits[(self.incorrect_mask) * (self.freq_mask)].size()).cuda()
#                     for i in range(len(self.gt_classes[(self.incorrect_mask) * (self.freq_mask)])):
#                         gt_logits[i,self.gt_classes[(self.incorrect_mask) * (self.freq_mask)][i]] = 1.
#                     if ((self.incorrect_mask) * (self.freq_mask)).nonzero().numel():
#                         BCEloss += nn.BCELoss(reduction="sum")(
#                         nn.Softmax(1)(self.pred_class_logits[(self.incorrect_mask) * (self.freq_mask)]/self.temperature),
#                         nn.Softmax(1)(gt_logits/self.temperature)
#                         ) * (self.temperature[0] ** 2.)

                    
#                     BCEloss_guide = BCEloss_guide / total_size

#                     cr_mask = (self.gt_classes >= self.class_num_list[1]) & (self.gt_classes < self.class_num_list[2])
#                     cr_loss = 0
#                     if cr_mask.sum() > 0:
#                         cr_loss = torch.sum(-1*torch.log(1-nn.Softmax(1)(self.pred_class_logits[cr_mask])[:,:self.class_num_list[1]]))/cr_mask.sum()
    
                    weight_d = float(self.class_num_list[self.stage_phase[0]-1])/self.class_num_list[self.stage_phase[0]]
                    weight_c = 1. - weight_d
                    return {'loss_cls_c':  weight_c * CE_c_loss,
                            'loss_cls_f':  weight_c * CE_f_loss, 
                            'loss_cls_BCE':weight_d * BCEloss_total, 
                            'loss_cr': weight_c * 0}
                else:
#                     cr_mask = (self.gt_classes >= self.class_num_list[1]) & (self.gt_classes < self.class_num_list[2])
#                     cr_loss = 0
#                     if cr_mask.sum() > 0:
#                         cr_loss = torch.sum(-1*torch.log(1-nn.Softmax(1)(self.pred_class_logits[cr_mask])[:,:self.class_num_list[1]]))/cr_mask.sum()
                        
                    weight_d = float(self.class_num_list[self.stage_phase[0]-1])/self.class_num_list[self.stage_phase[0]]
                    weight_c = 1. - weight_d
                    loss =  F.cross_entropy(self.pred_class_logits[self.other_mask], self.gt_classes[self.other_mask], self.weighted_tensor, reduction="mean") 
                    return {'loss_cls_c': weight_c*loss, 
                            'loss_cls_f': 0, 'loss_cls_BCE': 0, 'loss_cr': weight_c*0}
            else:
                loss =  F.cross_entropy(self.pred_class_logits[self.other_mask], self.gt_classes[self.other_mask], reduction="mean") 
                loss += nn.KLDivLoss(reduction='batchmean')(self.pred_class_logits[self.correct_mask], self.pred_class_logits_0[self.correct_mask])
                return loss

        else:
            if self.weighted_CE:
#                 nn.Softmax(1)(self.pred_class_logits/self.temperature)
#                 cr_mask = (self.gt_classes >= self.class_num_list[0]) & (self.gt_classes < self.class_num_list[2])
#                 cr_loss = 0
#                 if cr_mask.sum() > 0:
# #                     gt_logits = torch.zeros(self.pred_class_logits[cr_mask].size()).cuda()
# #                     gt_logits[torch.arange(gt_logits.size(0)),self.gt_classes[cr_mask]] = 1.
# #                     cr_loss = nn.KLDivLoss(reduction='sum')(
# #                         nn.Softmax(1)(self.pred_class_logits[cr_mask]/self.temperature).log()[:,:self.class_num_list[0]],
# #                         nn.Softmax(1)(gt_logits/self.temperature)[:,:self.class_num_list[0]]) * self.temperature ** 2.
# #                     cr_loss = cr_loss/cr_mask.sum()
#                     cr_loss = torch.sum(-1*torch.log(1-nn.Softmax(1)(self.pred_class_logits[cr_mask])[:,:self.class_num_list[0]]))/cr_mask.sum()
                ce_loss = F.cross_entropy(self.pred_class_logits,
                                       self.gt_classes, self.weighted_tensor, reduction="mean") 
                
                return {'loss_cls': 1.*ce_loss, 'loss_cr': 0}
            else:
                ce_loss = F.cross_entropy(self.pred_class_logits, self.gt_classes, reduction="mean")
                return {'loss_cls': ce_loss}

    def smooth_l1_loss(self):
        """
        Compute the smooth L1 loss for box regression.
        Returns:
            scalar Tensor
        """
        gt_proposal_deltas = self.box2box_transform.get_deltas(
            self.proposals.tensor, self.gt_boxes.tensor
        )
        box_dim = gt_proposal_deltas.size(1)  # 4 or 5
        cls_agnostic_bbox_reg = self.pred_proposal_deltas.size(1) == box_dim
        device = self.pred_proposal_deltas.device

        bg_class_ind = self.pred_class_logits.shape[1] - 1

        # Box delta loss is only computed between the prediction for the gt class k
        # (if 0 <= k < bg_class_ind) and the target; there is no loss defined on predictions
        # for non-gt classes and background.
        # Empty fg_inds produces a valid loss of zero as long as the size_average
        # arg to smooth_l1_loss is False (otherwise it uses torch.mean internally
        # and would produce a nan loss).
        fg_inds = torch.nonzero((self.gt_classes >= 0) & (self.gt_classes < bg_class_ind)).squeeze(
            1
        )
        if cls_agnostic_bbox_reg:
            # pred_proposal_deltas only corresponds to foreground class for agnostic
            gt_class_cols = torch.arange(box_dim, device=device)
        else:
            fg_gt_classes = self.gt_classes[fg_inds]
            # pred_proposal_deltas for class k are located in columns [b * k : b * k + b],
            # where b is the dimension of box representation (4 or 5)
            # Note that compared to Detectron1,
            # we do not perform bounding box regression for background classes.
            gt_class_cols = box_dim * fg_gt_classes[:, None] + torch.arange(box_dim, device=device)

        loss_box_reg = smooth_l1_loss(
            self.pred_proposal_deltas[fg_inds[:, None], gt_class_cols],
            gt_proposal_deltas[fg_inds],
            self.smooth_l1_beta,
            reduction="sum",
        )
        # The loss is normalized using the total number of regions (R), not the number
        # of foreground regions even though the box regression loss is only defined on
        # foreground regions. Why? Because doing so gives equal training influence to
        # each foreground example. To see how, consider two different minibatches:
        #  (1) Contains a single foreground region
        #  (2) Contains 100 foreground regions
        # If we normalize by the number of foreground regions, the single example in
        # minibatch (1) will be given 100 times as much influence as each foreground
        # example in minibatch (2). Normalizing by the total number of regions, R,
        # means that the single example in minibatch (1) and each of the 100 examples
        # in minibatch (2) are given equal influence.
        loss_box_reg = loss_box_reg / self.gt_classes.numel()
        return loss_box_reg

    def losses(self, is_global=False):
        """
        Compute the default losses for box head in Fast(er) R-CNN,
        with softmax cross entropy loss and smooth L1 loss.

        Returns:
            A dict of losses (scalar tensors) containing keys "loss_cls" and "loss_box_reg".
        """
        if self.stage in self.stage_phase_mute:
            return {}
        if self.stage in self.stage_phase:
            out = dict()
            for name, item in self.softmax_cross_entropy_loss().items():
                out[name] = item
            out['loss_box_reg'] = self.smooth_l1_loss()
            return out
#                 return {
#                     "loss_cls": self.softmax_cross_entropy_loss(),
#                     "loss_box_reg": self.smooth_l1_loss(),
#                 }
        else:
            return {
                "loss_box_reg": self.smooth_l1_loss(),
            }
            
    def predict_boxes(self):
        """
        Returns:
            list[Tensor]: A list of Tensors of predicted class-specific or class-agnostic boxes
                for each image. Element i has shape (Ri, K * B) or (Ri, B), where Ri is
                the number of predicted objects for image i and B is the box dimension (4 or 5)
        """
        num_pred = len(self.proposals)
        B = self.proposals.tensor.shape[1]
        K = self.pred_proposal_deltas.shape[1] // B
        boxes = self.box2box_transform.apply_deltas(
            self.pred_proposal_deltas.view(num_pred * K, B),
            self.proposals.tensor.unsqueeze(1).expand(num_pred, K, B).reshape(-1, B),
        )
        return boxes.view(num_pred, K * B).split(self.num_preds_per_image, dim=0)

    def predict_probs(self):
        """
        Returns:
            list[Tensor]: A list of Tensors of predicted class probabilities for each image.
                Element i has shape (Ri, K + 1), where Ri is the number of predicted objects
                for image i.
        """
        probs = F.softmax(self.pred_class_logits, dim=-1)
        return probs.split(self.num_preds_per_image, dim=0)

    def inference(self, score_thresh, nms_thresh, topk_per_image):
        """
        Args:
            score_thresh (float): same as fast_rcnn_inference.
            nms_thresh (float): same as fast_rcnn_inference.
            topk_per_image (int): same as fast_rcnn_inference.
        Returns:
            list[Instances]: same as fast_rcnn_inference.
            list[Tensor]: same as fast_rcnn_inference.
        """
        boxes = self.predict_boxes()
        scores = self.predict_probs()
        image_shapes = self.image_shapes

        return fast_rcnn_inference(
            boxes, scores, image_shapes, score_thresh, nms_thresh, topk_per_image
        )

class FastRCNNOutputLayers(nn.Module):
    """
    Two linear layers for predicting Fast R-CNN outputs:
      (1) proposal-to-detection box regression deltas
      (2) classification scores
    """

    def __init__(self, input_size, num_classes, cls_agnostic_bbox_reg, box_dim=4):
        """
        Args:
            input_size (int): channels, or (channels, height, width)
            num_classes (int): number of foreground classes
            cls_agnostic_bbox_reg (bool): whether to use class agnostic for bbox regression
            box_dim (int): the dimension of bounding boxes.
                Example box dimensions: 4 for regular XYXY boxes and 5 for rotated XYWHA boxes
        """
        super(FastRCNNOutputLayers, self).__init__()
        
        if not isinstance(input_size, int):
            input_size = np.prod(input_size)

        # The prediction layer for num_classes foreground classes and one background class
        # (hence + 1)

        num_bbox_reg_classes = 1 if cls_agnostic_bbox_reg else num_classes
        self.bbox_pred = nn.Linear(input_size, num_bbox_reg_classes * box_dim)
        nn.init.normal_(self.bbox_pred.weight, std=0.001)
        nn.init.constant_(self.bbox_pred.bias, 0)
        
        self.cls_score = nn.Linear(input_size, num_classes + 1)
        nn.init.normal_(self.cls_score.weight, std=0.01)
        for l in [self.cls_score]:
            nn.init.constant_(l.bias, 0)

    def forward(self, x):

        if x.dim() > 2:
            x = torch.flatten(x, start_dim=1)
        scores = self.cls_score(x)
        proposal_deltas = self.bbox_pred(x)
        return scores, proposal_deltas

class FastRCNNOutputLayers_box(nn.Module):
    """
    Two linear layers for predicting Fast R-CNN outputs:
      (1) proposal-to-detection box regression deltas
      (2) classification scores
    """

    def __init__(self, input_size, num_classes, cls_agnostic_bbox_reg, box_dim=4):
        """
        Args:
            input_size (int): channels, or (channels, height, width)
            num_classes (int): number of foreground classes
            cls_agnostic_bbox_reg (bool): whether to use class agnostic for bbox regression
            box_dim (int): the dimension of bounding boxes.
                Example box dimensions: 4 for regular XYXY boxes and 5 for rotated XYWHA boxes
        """
        super(FastRCNNOutputLayers_box, self).__init__()
        
        if not isinstance(input_size, int):
            input_size = np.prod(input_size)

        # The prediction layer for num_classes foreground classes and one background class
        # (hence + 1)

        num_bbox_reg_classes = 1 if cls_agnostic_bbox_reg else num_classes
        self.bbox_pred = nn.Linear(input_size, num_bbox_reg_classes * box_dim)
        nn.init.normal_(self.bbox_pred.weight, std=0.001)
        nn.init.constant_(self.bbox_pred.bias, 0)
            

    def forward(self, x):

        if x.dim() > 2:
            x = torch.flatten(x, start_dim=1)
#         scores = self.cls_score(x)
        proposal_deltas = self.bbox_pred(x)
        return proposal_deltas
    
class FastRCNNOutputLayers_class(nn.Module):
    """
    Two linear layers for predicting Fast R-CNN outputs:
      (1) proposal-to-detection box regression deltas
      (2) classification scores
    """

    def __init__(self, input_size, num_classes, cls_agnostic_bbox_reg, box_dim=4):
        """
        Args:
            input_size (int): channels, or (channels, height, width)
            num_classes (int): number of foreground classes
            cls_agnostic_bbox_reg (bool): whether to use class agnostic for bbox regression
            box_dim (int): the dimension of bounding boxes.
                Example box dimensions: 4 for regular XYXY boxes and 5 for rotated XYWHA boxes
        """
        super(FastRCNNOutputLayers_class, self).__init__()
        
        if not isinstance(input_size, int):
            input_size = np.prod(input_size)

        # The prediction layer for num_classes foreground classes and one background class
        # (hence + 1)


        self.cls_score = nn.Linear(input_size, num_classes + 1)
        nn.init.normal_(self.cls_score.weight, std=0.01)
        for l in [self.cls_score]:
            nn.init.constant_(l.bias, 0)

    def forward(self, x):

        if x.dim() > 2:
            x = torch.flatten(x, start_dim=1)
        scores = self.cls_score(x)
#         proposal_deltas = self.bbox_pred(x)
        return scores
