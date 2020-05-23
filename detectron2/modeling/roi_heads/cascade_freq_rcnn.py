# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch
from torch import nn
from torch.autograd.function import Function
import logging
import pickle
import json

from detectron2.layers import ShapeSpec, SELayer
from detectron2.structures import Boxes, Instances, pairwise_iou
from detectron2.utils.events import get_event_storage
from detectron2.data.datasets.lvis_categories_mapper import *

from ..box_regression import Box2BoxTransform
from ..matcher import Matcher
from ..poolers import ROIPooler
from .box_head import build_box_head
from .fast_rcnn import FastRCNNOutputLayers_class, FastRCNNOutputLayers_box, FastRCNNOutputs, fast_rcnn_inference
from .roi_heads import ROI_HEADS_REGISTRY, StandardROIHeads

###########
# Original Cascade RCNN supported
# please trun on the original switch
# and change the 

class _ScaleGradient(Function):
    @staticmethod
    def forward(ctx, input, scale):
        ctx.scale = scale
        return input

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output * ctx.scale, None

logger = logging.getLogger(__name__)

@ROI_HEADS_REGISTRY.register()
class CascadeFREQROIHeads(StandardROIHeads):
    def _init_box_head(self, cfg):
        # fmt: off
        self.index_list = cate_id_list()
        pooler_resolution        = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pooler_scales            = tuple(1.0 / self.feature_strides[k] for k in self.in_features)
        sampling_ratio           = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler_type              = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE
        cascade_bbox_reg_weights = cfg.MODEL.ATTENTION_ROI_HEAD.BBOX_REG_WEIGHTS
        cascade_ious             = cfg.MODEL.ATTENTION_ROI_HEAD.IOUS
        self.num_cascade_stages  = len(cascade_ious)
        
        self.num_f_classes = cfg.DATASETS.NUM_CLASSES_F
        self.num_c_classes = cfg.DATASETS.NUM_CLASSES_C
        self.num_r_classes = cfg.DATASETS.NUM_CLASSES_R
        
        self.test_keep_anns = cfg.TEST.KEEP_ANNS
        self.test_cal_acc = cfg.TEST.CAL_ACC
        if self.test_cal_acc:
            self.acc_data_path = './LVIS_acc_data_x_teacher.json'
            assert self.test_keep_anns, "[Test] Calculate acc on. Please ture on TEST_KEEP_ANNS."
            self.acc_data = {'f':0,'f_gt':0,'c':0,'c_gt':0,'r':0,'r_gt':0,'c_fgt':0,'f_cgt':0,
                             'f_rgt':0,'c_rgt':0,'r_fgt':0,'r_cgt':0,'pred_f':0,'pred_c':0,'pred_r':0}
            with open(self.acc_data_path, 'w') as outfile:
                json.dump(self.acc_data, outfile)
        
        ############# my switch
        self.shared_weight = cfg.MODEL.ATTENTION_ROI_HEAD.SHARED_WEIGHT
        self.weighted_CE = cfg.MODEL.ATTENTION_ROI_HEAD.WEIGHTED_CE
        self.learn_weighted_CE = cfg.MODEL.ATTENTION_ROI_HEAD.LEARN_WEIGHTED_CE
        self.KLCE = cfg.MODEL.ATTENTION_ROI_HEAD.KLCE
        self.has_selayer = cfg.MODEL.ATTENTION_ROI_HEAD.SELAYER
        self.stage_phase = 0
        self.multi_head = False
        ############# attention part
        self.self_attentiion = cfg.MODEL.ATTENTION_ROI_HEAD.ATTENTION
        self.enhance_size = cfg.MODEL.ATTENTION_ROI_HEAD.CHANNEL_OF_ENHANCED_FEATURE
        self.relu = nn.ReLU(inplace=True)
    
        ############# weighted ce part
        self.weight = None
        if self.learn_weighted_CE:
            self.comm_weight = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
            self.rare_weight = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
            self.comm_weight.data.fill_(2.)
            self.rare_weight.data.fill_(3.)
            self.weight = [self.comm_weight, self.rare_weight]
        
        
        assert len(cascade_bbox_reg_weights) == self.num_cascade_stages
#         assert self.num_cascade_stages == 3, "CascadeFREQROIHeads only support 3 stages now!"
        assert cfg.MODEL.ROI_BOX_HEAD.CLS_AGNOSTIC_BBOX_REG,  \
            "CascadeFREQROIHeads only support class-agnostic regression now!"
        assert cascade_ious[0] == cfg.MODEL.ROI_HEADS.IOU_THRESHOLDS[0]
        # fmt: on
        in_channels = [self.feature_channels[f] for f in self.in_features]

        # Check all channel counts are equal
        assert len(set(in_channels)) == 1, in_channels
        in_channels = in_channels[0]

        
        self.box_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )

        pooled_shape = ShapeSpec(
            channels=in_channels, width=pooler_resolution, height=pooler_resolution
        )
        
        if self.has_selayer:
            self.se_layer = nn.ModuleList()
            for k in range(self.num_cascade_stages):
                self.se_layer.append(SELayer(channel=in_channels).cuda())
                
        self.box_head = nn.ModuleList()

        self.box_predictor = nn.ModuleList()
        self.class_predictor = nn.ModuleList()
        self.box2box_transform = []
        self.proposal_matchers = []
        self.W_G = []
        ## self.num_classes_list
        # first one: frequent classes + 1 (common or rare), second one: frequent and common classes + 1 (rare)
        # third one: frequent and common and rare classes.
        self.num_classes_list = [self.num_f_classes, self.num_f_classes + self.num_c_classes, self.num_f_classes + self.num_c_classes + self.num_r_classes]
        self.num_classes_list_with_bg = []
#         if self.multi_head: self.num_cascade_stages += 1
        for k in range(self.num_cascade_stages):
            ## Only full classes here
            num_classes = self.num_classes_list[2]
            
#             num_classes = self.num_classes_list[k]
#             if k != self.num_cascade_stages - 1:
#                 num_classes += 1
            self.num_classes_list_with_bg.append(num_classes)
            box_head = build_box_head(cfg, pooled_shape)
            self.box_head.append(box_head)
            
            if self.self_attentiion and k != 0:
                self.box_predictor.append(
                    FastRCNNOutputLayers_box(
                        box_head.output_size + self.enhance_size, num_classes, cls_agnostic_bbox_reg=True
                    )
                )
                self.class_predictor.append(
                    FastRCNNOutputLayers_class(
                        box_head.output_size + self.enhance_size, num_classes, cls_agnostic_bbox_reg=True
                    )
                )
                self.W_G.append(
                    nn.Linear(self.class_predictor[k-1].cls_score.weight.shape[1] + 1, self.enhance_size).cuda()
                )
            else:
                self.box_predictor.append(
                    FastRCNNOutputLayers_box(
                        box_head.output_size, num_classes, cls_agnostic_bbox_reg=True
                    )
                )
                self.class_predictor.append(
                    FastRCNNOutputLayers_class(
                        box_head.output_size, num_classes, cls_agnostic_bbox_reg=True
                    )
                )
            self.box2box_transform.append(Box2BoxTransform(weights=cascade_bbox_reg_weights[k]))

            if k == 0:
                # The first matching is done by the matcher of ROIHeads (self.proposal_matcher).
                self.proposal_matchers.append(None)
            else:
                self.proposal_matchers.append(
                    Matcher([cascade_ious[k]], [0, 1], allow_low_quality_matches=False)
                )
                
        self.mute_loss_stage = []
        
        if self.stage_phase == 0:
            logger.info("Parameters in self.class_predictor[1] are fixed!!")
            for param in self.class_predictor[1].parameters():
                param.requires_grad = False
            logger.info("Parameters in self.box_predictor[1] are fixed!!")
            for param in self.box_predictor[1].parameters():
                param.requires_grad = False
            ##### stage 3
#             logger.info("Parameters in self.class_predictor[2] are fixed!!")
#             for param in self.class_predictor[2].parameters():
#                 param.requires_grad = False
#             logger.info("Parameters in self.box_predictor[2] are fixed!!")
#             for param in self.box_predictor[2].parameters():
#                 param.requires_grad = False
            if self.multi_head:
                logger.info("Parameters in self.box_head[3] are fixed!!")
                for param in self.box_head[3].parameters():
                    param.requires_grad = False
                logger.info("Parameters in self.box_predictor[3] are fixed!!")
                for param in self.box_predictor[3].parameters():
                    param.requires_grad = False
                logger.info("Parameters in self.class_predictor[3] are fixed!!")
                for param in self.class_predictor[3].parameters():
                    param.requires_grad = False
            
        
        if self.KLCE and self.stage_phase == 1:
            self.mute_loss_stage = [0, 3]
            logger.info("Parameters in self.box_head[0] are fixed!!")
            for param in self.box_head[0].parameters():
                param.requires_grad = False
            logger.info("Parameters in self.box_predictor[0] are fixed!!")
            for param in self.box_predictor[0].parameters():
                param.requires_grad = False
            logger.info("Parameters in self.class_predictor[0] are fixed!!")
            for param in self.class_predictor[0].parameters():
                param.requires_grad = False
            logger.info("Parameters in self.class_predictor[2] are fixed!!")
            for param in self.class_predictor[2].parameters():
                param.requires_grad = False
            if self.multi_head:
                logger.info("Parameters in self.box_head[3] are fixed!!")
                for param in self.box_head[3].parameters():
                    param.requires_grad = False
                logger.info("Parameters in self.box_predictor[3] are fixed!!")
                for param in self.box_predictor[3].parameters():
                    param.requires_grad = False
                logger.info("Parameters in self.class_predictor[3] are fixed!!")
                for param in self.class_predictor[3].parameters():
                    param.requires_grad = False
            if self.has_selayer:
                for param in self.se_layer[0].parameters():
                    param.requires_grad = False

        if self.KLCE and self.stage_phase == 2:
            self.mute_loss_stage = [0, 1]
            logger.info("Parameters in self.box_head[0] are fixed!!")
            for param in self.box_head[0].parameters():
                param.requires_grad = False
            logger.info("Parameters in self.box_predictor[0] are fixed!!")
            for param in self.box_predictor[0].parameters():
                param.requires_grad = False
            logger.info("Parameters in self.class_predictor[0] are fixed!!")
            for param in self.class_predictor[0].parameters():
                param.requires_grad = False
            logger.info("Parameters in self.box_head[1] are fixed!!")
            for param in self.box_head[1].parameters():
                param.requires_grad = False
            logger.info("Parameters in self.box_predictor[1] are fixed!!")
            for param in self.box_predictor[1].parameters():
                param.requires_grad = False
            logger.info("Parameters in self.class_predictor[1] are fixed!!")
            for param in self.class_predictor[1].parameters():
                param.requires_grad = False
            if self.has_selayer:
                for param in self.se_layer[0].parameters():
                    param.requires_grad = False
                for param in self.se_layer[1].parameters():
                    param.requires_grad = False
        if self.KLCE and self.stage_phase == 3:
            self.mute_loss_stage = [0, 1, 2]
            logger.info("Parameters in self.box_head[0] are fixed!!")
            for param in self.box_head[0].parameters():
                param.requires_grad = False
            logger.info("Parameters in self.box_predictor[0] are fixed!!")
            for param in self.box_predictor[0].parameters():
                param.requires_grad = False
            logger.info("Parameters in self.class_predictor[0] are fixed!!")
            for param in self.class_predictor[0].parameters():
                param.requires_grad = False
            logger.info("Parameters in self.box_head[1] are fixed!!")
            for param in self.box_head[1].parameters():
                param.requires_grad = False
            logger.info("Parameters in self.box_predictor[1] are fixed!!")
            for param in self.box_predictor[1].parameters():
                param.requires_grad = False
            logger.info("Parameters in self.class_predictor[1] are fixed!!")
            for param in self.class_predictor[1].parameters():
                param.requires_grad = False
            logger.info("Parameters in self.box_head[2] are fixed!!")
            for param in self.box_head[2].parameters():
                param.requires_grad = False
            logger.info("Parameters in self.box_predictor[2] are fixed!!")
            for param in self.box_predictor[2].parameters():
                param.requires_grad = False
            logger.info("Parameters in self.class_predictor[2] are fixed!!")
            for param in self.class_predictor[2].parameters():
                param.requires_grad = False
            ## mask in roi_head.py
            if self.has_selayer:
                for param in self.se_layer[0].parameters():
                    param.requires_grad = False
                for param in self.se_layer[1].parameters():
                    param.requires_grad = False


    def forward(self, images, features, proposals, targets=None):
        del images
        # reIndex the targets
        if self.training or self.test_keep_anns:
#             print("targets:",targets)
            targets_list = self.target_gt_classes_transform(targets, self.num_classes_list, self.num_classes_list_with_bg)
            assert len(targets_list[0]) == len(targets_list[1]) == len(targets_list[2])
                      
        if self.learn_weighted_CE:
            logger.info("Learning weight CE comm, rare = {}, {}".format(self.comm_weight,self.rare_weight))

        if self.training or self.test_keep_anns:

            proposals_mask = self.label_and_sample_proposals(proposals, targets_list[2], bg_classes = self.num_classes_list_with_bg[2])
            proposals = self.label_and_sample_proposals(proposals, targets_list[0], bg_classes = self.num_classes_list_with_bg[0])
            ### analyze the proposal boxes 
#             p_bboxes = {'gt':{'f':0,'c':0,'r':0}, 'prop':{'f':0,'c':0,'r':0}}

#             with open('./chris_output/LVIS_proposal_analysis.pkl', 'rb') as f:
#                 p_bboxes = f.read()
#                 p_bboxes = pickle.loads(p_bboxes, encoding='latin1')
#                 f.close()
#             for i in range(len(proposals)):
#                 p_bboxes['gt']['f'] += (targets_list[0][i].get('gt_frequencys') == 0).sum().item()
#                 p_bboxes['gt']['c'] += (targets_list[0][i].get('gt_frequencys') == 1).sum().item()
#                 p_bboxes['gt']['r'] += (targets_list[0][i].get('gt_frequencys') == 2).sum().item()
#                 foreground_mask = proposals[0].get('gt_classes') != 1230
#                 p_bboxes['prop']['f'] += (proposals[i].get('gt_frequencys')[foreground_mask] == 0).sum().item()
#                 p_bboxes['prop']['c'] += (proposals[i].get('gt_frequencys')[foreground_mask] == 1).sum().item()
#                 p_bboxes['prop']['r'] += (proposals[i].get('gt_frequencys')[foreground_mask] == 2).sum().item()
#             with open('./chris_output/LVIS_proposal_analysis.pkl', 'wb') as f:
#                 pickle.dump(p_bboxes, f)
#                 f.close()
        
        features_list = [features[f] for f in self.in_features]
        
        ###################
        # Start forward
        if self.training:
            losses = self._forward_box(features_list, proposals, targets_list)
#             if self.stage_phase != 3:
            losses.update(self._forward_mask(features_list, proposals_mask))
            losses.update(self._forward_keypoint(features_list, proposals))
            return proposals, losses
        elif not self.training and self.test_keep_anns:
            pred_instances = self._forward_box(features_list, proposals, targets_list)
            pred_instances = self.forward_with_given_boxes(features, pred_instances)
            pred_instances = self.reindex_result(pred_instances)
            return pred_instances, {}
        else:
            pred_instances = self._forward_box(features_list, proposals)
            pred_instances = self.forward_with_given_boxes(features, pred_instances)
            pred_instances = self.reindex_result(pred_instances)
            return pred_instances, {}

    def _forward_box(self, features, proposals, targets_list=None, enhanced_feature=[]):
        # targets_list = [gt_instances_f, gt_instances_fc, gt_instances_fcr]
        
        head_outputs = []
        if self.test_keep_anns:
            head_outputs_classes = []
            head_outputs_classes_freq = []
        image_sizes = [x.image_size for x in proposals]
        
        for k in range(self.num_cascade_stages):            
#             print('len(proposals):',len(proposals)) # 2
#             print('proposals:',proposals[0]) # 512
            if k > 0 :
                if not self.multi_head:
                    # The output boxes of the previous stage are the input proposals of the next stage
                    pred_boxes = head_outputs[-1].predict_boxes()

                    proposals = self._create_proposals_from_boxes(
                        pred_boxes, image_sizes
                    )
                    if self.training or self.test_keep_anns:
                        proposals = self._match_and_label_boxes(proposals, k, targets_list[k])
                else:
                    if k < self.num_cascade_stages - 1:
                        # The output boxes of the previous stage are the input proposals of the next stage
                        pred_boxes = head_outputs[-1].predict_boxes()

                        proposals = self._create_proposals_from_boxes(
                            pred_boxes, image_sizes
                        )
                        if self.training or self.test_keep_anns:
                            proposals = self._match_and_label_boxes(proposals, k, targets_list[k])
                    else: 
                        pred_boxes = head_outputs[0].predict_boxes()

                        proposals = self._create_proposals_from_boxes(
                            pred_boxes, image_sizes
                        )
                        if self.training or self.test_keep_anns:
                            proposals = self._match_and_label_boxes(proposals, 1, targets_list[1])

            outputs, enhanced_feature = self._run_stage(features, proposals, enhanced_feature, k)
            head_outputs.append(outputs)
            if self.test_keep_anns:
                assert len(proposals) == 1, "keep anns inference only support 1 image per GPU."
                head_outputs_classes.append(proposals[0].get_fields()['gt_classes'])
                head_outputs_classes_freq.append(proposals[0].get_fields()['gt_frequencys'])
            
        if self.training:
            losses = {}
            storage = get_event_storage()
            for stage, output in enumerate(head_outputs):
                if stage not in self.mute_loss_stage:
                    with storage.name_scope("stage{}".format(stage)):
                        stage_losses = output.losses()
                    losses.update({k + "_stage{}".format(stage): v for k, v in stage_losses.items()})
            return losses
        else:
            # Each is a list[Tensor] of length #image. Each tensor is Ri x (K+1)
            scores_per_stage = [h.predict_probs() for h in head_outputs]
            if self.test_keep_anns:
                gt_class_per_stage = [h.gt_classes for h in head_outputs]
            scores = []
            for idx, scores_per_image in enumerate(zip(*scores_per_stage)):
                scores_per_image = list(scores_per_image)
#                 print(scores_per_image[0].shape)
#                 print(scores_per_image[1].shape)
                len_0 = self.num_classes_list[0] # 317
                len_1 = self.num_classes_list[1] # 778
                len_2 = self.num_classes_list[2] # 1232
#                 scores_per_image[0] = torch.cat((scores_per_image[0][:,0:len_0-2],scores_per_image[1][:,len_0:]),1)
#                 scores_per_image[0] = torch.cat((scores_per_image[0],scores_per_image[2][:,len_1:]),1)
#                 scores_per_image[1] = torch.cat((scores_per_image[1],scores_per_image[2][:,len_1:]),1)
#                 print(scores_per_image[2].shape)
#                 print(sum(list(scores_per_image)).shape) # 1000,1231
#                 tmp = torch.cat((scores_per_image[0][:,:len_0],scores_per_image[1][:,(len_0):(len_1)],scores_per_image[2][:,(len_1-2):(len_2-1)]),1)

                # type 1
#                 weight = torch.ones(scores_per_image[0].size()[0]).cuda()/scores_per_image[0][:,:len_0].sum(1)
#                 feq_mask = scores_per_image[3].argmax(1) < len_0
#                 scores_per_image[3][feq_mask,:len_0] = (scores_per_image[0][feq_mask,:len_0] + scores_per_image[3][feq_mask,:len_0]) / 2.
#                 tmp = scores_per_image[3]
                  
                # type 2
#                 tmp = torch.zeros(scores_per_image[0].size()).cuda()
#                 for i in range(scores_per_image[0].shape[0]):
#                     if scores_per_image[0][i,:].argmax() <= len_0 and scores_per_image[0][i,:].max() >= 0.7:
#                         tmp[i,:] = scores_per_image[0][i,:]
#                     else:
#                         tmp[i,:] = scores_per_image[1][i,:]
                
                # type 2;
#                 tmp = torch.zeros(scores_per_image[0].size()).cuda()
#                 for i in range(scores_per_image[0].shape[0]):
#                     if scores_per_image[0][i,:].argmax() <= len_0:# and scores_per_image[0][i,:].max() >= 0.4:
#                         weight = scores_per_image[1][i,:len_0].sum()/scores_per_image[0][i,:len_0].sum()
#                         freq_score = (scores_per_image[0][i,:len_0] * weight + scores_per_image[1][i,:len_0]) / 2.

#                         tmp[i,:] = torch.cat((freq_score, scores_per_image[1][i,len_0:]),0)
#                     else:
#                         tmp[i,:] = scores_per_image[1][i,:]

                # type 3
#                 weight = scores_per_image[1][:,:len_0].sum(1)/scores_per_image[0][:,:len_0].sum(1)
#                 freq_score = ((scores_per_image[0][:,:len_0].T * weight).T * scores_per_image[1][:,:len_0]) /2.
#                 tmp = torch.cat((freq_score, scores_per_image[1][:,len_0:]),1)
            
#                 tmp = scores_per_image[1]  
#                 scores.append(sum(list(scores_per_image)) * (1.0 / self.num_cascade_stages))
#                 if self.stage_phase == 2:
#                     tmp = scores_per_image[1]  
#                     tmp = (scores_per_image[1] + scores_per_image[2]) / 2.

                # keep annotations
#                 if self.test_keep_anns:
#                     tmp = torch.zeros(scores_per_image[0].size()).cuda()
#                     for i in range(scores_per_image[0].shape[0]):
#                         freq_mask = gt_class_per_stage[-1] < len_0
#                         other_mask = ~freq_mask
# #                         w0 = nn.Softmax(1)(scores_per_image[0][freq_mask])[:len_0].sum(1)
# #                         w1 = nn.Softmax(1)(scores_per_image[1][freq_mask])[:len_0].sum(1)
#                         tmp[freq_mask,:len_0] = (scores_per_image[0][freq_mask,:len_0] + scores_per_image[1][freq_mask,:len_0]) / 2.
#                         tmp[other_mask, len_0:len_1] = scores_per_image[1][other_mask, len_0:len_1]
                ##########
#                 alpha = 0.4
#                 feq_mask = (scores_per_image[3].argmax(1) < len_0) & (scores_per_image[3].max(1).values >= alpha)
# #                 other_mask = ~feq_mask
#                 comm_mask = (scores_per_image[3].argmax(1) >= len_0) & (scores_per_image[3].argmax(1) < len_1) & (scores_per_image[3].max(1).values >= alpha)
#                 scores_per_image[3][feq_mask,len_0:] = 0.
#                 scores_per_image[3][comm_mask,:len_0] = 0.
                ##########
                k = 1
                head = 0
#                 mask = torch.ones(scores_per_image[head].size(), dtype=torch.bool)
#                 topk_val, topk_idx = torch.topk(scores_per_image[head], k, dim=1)
#                 for i in range(k):
#                     mask[torch.arange(len(mask)), topk_idx[:,i]] = False
#                 scores_per_image[head][mask] = 0.
                tmp = scores_per_image[head]
                
                scores.append(tmp)
                
                if self.test_cal_acc:
                    with open(self.acc_data_path) as json_file:
                        acc_dict = json.load(json_file)
                    {'f':0,'f_gt':0,'c':0,'c_gt':0,'r':0,'r_gt':0,'c_fgt':0,'f_cgt':0,
                             'f_rgt':0,'c_rgt':0,'r_fgt':0,'r_cgt':0,'pred_f':0,'pred_c':0,'pred_r':0}
                    
                    acc_dict['f']+=int(((gt_class_per_stage[-1] == tmp.argmax(1)) & (gt_class_per_stage[-1] < len_0)).sum())
                    acc_dict['f_gt']+=int((gt_class_per_stage[-1] < len_0).sum())
                    acc_dict['c']+=int(((gt_class_per_stage[-1] == tmp.argmax(1)) & (gt_class_per_stage[-1] < len_1) & (gt_class_per_stage[-1] >= len_0)).sum())
                    acc_dict['c_gt']+=int(((gt_class_per_stage[-1] < len_1) & (gt_class_per_stage[-1] >= len_0)).sum())
                    acc_dict['r']+=int(((gt_class_per_stage[-1] == tmp.argmax(1)) & (gt_class_per_stage[-1] < len_2) & (gt_class_per_stage[-1] >= len_1)).sum())
                    acc_dict['r_gt']+=int(((gt_class_per_stage[-1] < len_2) & (gt_class_per_stage[-1] >= len_1)).sum())
                    acc_dict['f_cgt']+=int(((gt_class_per_stage[-1] < len_1) & (gt_class_per_stage[-1] >= len_0) & (tmp.argmax(1) < len_0)).sum())
                    acc_dict['c_fgt']+=int(((gt_class_per_stage[-1] < len_0) & (tmp.argmax(1) >= len_0) & (tmp.argmax(1) < len_1)).sum())
                    acc_dict['f_rgt']+=int(((gt_class_per_stage[-1] < len_2) & (gt_class_per_stage[-1] >= len_1) & (tmp.argmax(1) < len_0)).sum())
                    acc_dict['c_rgt']+=int(((gt_class_per_stage[-1] < len_2) & (gt_class_per_stage[-1] >= len_1) & (tmp.argmax(1) >= len_0) & (tmp.argmax(1) < len_1)).sum())
                    acc_dict['r_fgt']+=int(((gt_class_per_stage[-1] < len_0) & (tmp.argmax(1) >= len_1) & (tmp.argmax(1) < len_2)).sum())
                    acc_dict['r_cgt']+=int(((gt_class_per_stage[-1] < len_1) & (gt_class_per_stage[-1] >= len_0) & (tmp.argmax(1) >= len_1) & (tmp.argmax(1) < len_2)).sum())
                    acc_dict['pred_f']+=int((tmp.argmax(1) < len_0).sum())
                    acc_dict['pred_c']+=int(((tmp.argmax(1) >= len_0) & (tmp.argmax(1) < len_1)).sum())
                    acc_dict['pred_r']+=int(((tmp.argmax(1) >= len_1) & (tmp.argmax(1) < len_2)).sum())
                    with open(self.acc_data_path, 'w') as outfile:
                        json.dump(acc_dict, outfile)
                    
            
            # Average the scores across heads
            
            # Use the boxes of the last head
            boxes = head_outputs[0].predict_boxes()
            
            pred_instances, _ = fast_rcnn_inference(
                boxes,
                scores,
                image_sizes,
                self.test_score_thresh,
                self.test_nms_thresh,
                self.test_detections_per_img,
            )
            return pred_instances
    
#     def _forward_mask(self, features, instances):
#         """
#         Forward logic of the mask prediction branch.

#         Args:
#             features (list[Tensor]): #level input features for mask prediction
#             instances (list[Instances]): the per-image instances to train/predict masks.
#                 In training, they can be the proposals.
#                 In inference, they can be the predicted boxes.

#         Returns:
#             In training, a dict of losses.
#             In inference, update `instances` with new fields "pred_masks" and return it.
#         """
#         if not self.mask_on:
#             return {} if self.training else instances

#         if self.training:
#             # The loss is only defined on positive proposals.
#             proposals, _ = select_foreground_proposals(instances, self.num_classes)
#             proposal_boxes = [x.proposal_boxes for x in proposals]
#             mask_features = self.mask_pooler(features, proposal_boxes)
#             mask_logits = self.mask_head(mask_features)
#             return {"loss_mask": mask_rcnn_loss(mask_logits, proposals)}
#         else:
#             pred_boxes = [x.pred_boxes for x in instances]
#             mask_features = self.mask_pooler(features, pred_boxes)
#             mask_logits = self.mask_head(mask_features)
#             mask_rcnn_inference(mask_logits, instances)
#             return instances
        
    @torch.no_grad()
    def _match_and_label_boxes(self, proposals, stage, targets):
        """
        Match proposals with groundtruth using the matcher at the given stage.
        Label the proposals as foreground or background based on the match.

        Args:
            proposals (list[Instances]): One Instances for each image, with
                the field "proposal_boxes".
            stage (int): the current stage
            targets (list[Instances]): the ground truth instances

        Returns:
            list[Instances]: the same proposals, but with fields "gt_classes" and "gt_boxes"
        """
        num_fg_samples, num_bg_samples = [], []
        for proposals_per_image, targets_per_image in zip(proposals, targets):
            match_quality_matrix = pairwise_iou(
                targets_per_image.gt_boxes, proposals_per_image.proposal_boxes
            )
            # proposal_labels are 0 or 1
            matched_idxs, proposal_labels = self.proposal_matchers[stage](match_quality_matrix)
            if len(targets_per_image) > 0:
                gt_classes = targets_per_image.gt_classes[matched_idxs]
                # Label unmatched proposals (0 label from matcher) as background (label=num_classes)
                gt_classes[proposal_labels == 0] = self.num_classes_list_with_bg[stage] #self.num_classes
                gt_boxes = targets_per_image.gt_boxes[matched_idxs]
                gt_frequencys = targets_per_image.gt_frequencys[matched_idxs]
            else:
                gt_classes = torch.zeros_like(matched_idxs) + self.num_classes_list_with_bg[stage] # self.num_classes
                gt_frequencys = torch.zeros_like(matched_idxs) + self.num_classes_list_with_bg[stage]
                gt_boxes = Boxes(
                    targets_per_image.gt_boxes.tensor.new_zeros((len(proposals_per_image), 4))
                )
            proposals_per_image.gt_classes = gt_classes
            proposals_per_image.gt_frequencys = gt_frequencys
            proposals_per_image.gt_boxes = gt_boxes

            num_fg_samples.append((proposal_labels == 1).sum().item())
            num_bg_samples.append(proposal_labels.numel() - num_fg_samples[-1])

        # Log the number of fg/bg samples in each stage
        if not self.test_keep_anns:
            storage = get_event_storage()
            storage.put_scalar(
                "stage{}/roi_head/num_fg_samples".format(stage),
                sum(num_fg_samples) / len(num_fg_samples),
            )
            storage.put_scalar(
                "stage{}/roi_head/num_bg_samples".format(stage),
                sum(num_bg_samples) / len(num_bg_samples),
            )
        return proposals

    def _run_stage(self, features, proposals, enhanced_feature, stage):
        
        """
        Args:
            features (list[Tensor]): #lvl input features to ROIHeads
            proposals (list[Instances]): #image Instances, with the field "proposal_boxes"
            stage (int): the current stage

        Returns:
            FastRCNNOutputs: the output of this stage
        """
        ##### stage 3
#         if self.shared_weight:
#             if stage == 1 and not self.KLCE:
#                 ## F -> C
#                 shared_len = self.num_classes_list[0] #self.box_predictor[stage-1].cls_score.weight.shape[0]-2
#                 shared_wid = self.class_predictor[stage-1].cls_score.weight.shape[1]
#                 self.class_predictor[stage].cls_score.weight.data[:shared_len,:shared_wid] = self.class_predictor[stage-1].cls_score.weight.data[:shared_len,:shared_wid]
                
#             if stage == 2:
#                 ## C -> R
#                 shared_len = self.num_classes_list[1] #self.box_predictor[stage-1].cls_score.weight.shape[0]-2
#                 shared_wid = self.class_predictor[stage-1].cls_score.weight.shape[1]
#                 self.class_predictor[stage].cls_score.weight.data[:shared_len,:shared_wid] = self.class_predictor[stage-1].cls_score.weight.data[:shared_len,:shared_wid]
                
#                 ## R -> F
#                 shared_len = self.box_predictor[0].cls_score.weight.shape[0]-2
#                 shared_wid = self.box_predictor[0].cls_score.weight.shape[1]
#                 self.box_predictor[0].cls_score.weight.data[:shared_len,:shared_wid] = self.box_predictor[2].cls_score.weight.data[:shared_len,:shared_wid]
#                 ## R -> C
#                 shared_len_2 = self.box_predictor[1].cls_score.weight.shape[0]-2
#                 shared_wid_2 = self.box_predictor[1].cls_score.weight.shape[1]
#                 self.box_predictor[1].cls_score.weight.data[(shared_len+2):shared_len_2,:shared_wid_2] = self.box_predictor[2].cls_score.weight.data[(shared_len+2):shared_len_2,:shared_wid_2]
            
        
        if self.has_selayer:
            new_feature = []
            for f in features:
                new_feature.append(self.se_layer[stage](f))
            features = new_feature
            
        box_features = self.box_pooler(features, [x.proposal_boxes for x in proposals])
        if self.test_keep_anns or self.training:
            box_gt_classes = torch.cat([x.gt_classes for x in proposals],0)
        
        # The original implementation averages the losses among heads,
        # but scale up the parameter gradients of the heads.
        # This is equivalent to adding the losses among heads,
        # but scale down the gradients on features.
        box_features = _ScaleGradient.apply(box_features, 1.0 / 3.0)
        
        if self.KLCE:
            if self.test_keep_anns or self.training:
                if stage == 1 or stage == 0 :
                    box_features_0 = self.box_head[0](box_features)
                    pred_class_logits_0 = self.class_predictor[0](box_features_0)
                elif stage == 2:
                    box_features_0 = self.box_head[1](box_features)
                    pred_class_logits_0 = self.class_predictor[1](box_features_0)
                else:
                    ##### stage 3
                    len_0 = self.num_classes_list[0] # 317
                    len_1 = self.num_classes_list[1] # 778
                    len_2 = self.num_classes_list[2] # 1232
                    box_features_0 = self.box_head[0](box_features)
                    pred_class_logits_0 = self.class_predictor[0](box_features_0)

                    box_features_1 = self.box_head[1](box_features)
                    pred_class_logits_1 = self.class_predictor[1](box_features_1)

                    tmp = torch.zeros(pred_class_logits_1.size()).cuda() # (1024, 1231)
                    freq_mask = box_gt_classes < len_0
    #                 print(freq_mask.shape)
                    other_mask = ~freq_mask
                    tmp[freq_mask,:len_0] = (pred_class_logits_0[freq_mask,:len_0] + pred_class_logits_1[freq_mask,:len_0]) / 2.
                    tmp[other_mask, len_0:len_1] = pred_class_logits_1[other_mask, len_0:len_1]
                    pred_class_logits_0 = tmp

                del box_features_0
            else:
                box_features_0 = self.box_head[0](box_features)
                pred_class_logits_0 = self.class_predictor[0](box_features_0)
            
        box_features = self.box_head[stage](box_features)
        if self.self_attentiion and len(enhanced_feature) != 0:
            enhanced_feature = enhanced_feature[0].repeat(len(box_features),1)
            assert len(enhanced_feature) == len(box_features)
            box_features = torch.cat((box_features, enhanced_feature), 1)
        pred_proposal_deltas = self.box_predictor[stage](box_features)
        pred_class_logits = self.class_predictor[stage](box_features)

        del box_features
        if self.KLCE:
            outputs = FastRCNNOutputs(
                self.box2box_transform[stage],
                pred_class_logits,
                pred_proposal_deltas,
                proposals,
                self.smooth_l1_beta,
                self.num_classes_list,
                stage,
                self.weighted_CE,
                self.weight,
                self.KLCE,
                pred_class_logits_0,
            )
        else:
            outputs = FastRCNNOutputs(
                self.box2box_transform[stage],
                pred_class_logits,
                pred_proposal_deltas,
                proposals,
                self.smooth_l1_beta,
                self.num_classes_list,
                stage,
                self.weighted_CE,
                self.weight,
                self.KLCE,
            )
            
        if self.self_attentiion and stage < 2:
            # my defined of attention
            # all_possible_classes_logits = [num_class + 1 - 2]
            all_possible_classes_logits = pred_class_logits[:,:-1].max(0).values
            all_possible_classes_logits = nn.Softmax(0)(all_possible_classes_logits)

            # global_semantic_features = [num_class + 1 - 2, 1025]
            global_semantic_features = torch.cat((self.class_predictor[stage].cls_score.weight[:-1,:],
                                                      self.class_predictor[stage].cls_score.bias[:-1].unsqueeze(1)), 1).detach()
            # img_wise_semantic_pool = [1, 1025]
            img_wise_semantic_pool = torch.mm(all_possible_classes_logits.unsqueeze(0),global_semantic_features)

            tmp_feature = img_wise_semantic_pool.squeeze(0).repeat(len(pred_class_logits),1)

            enhanced_feature = self.W_G[stage](tmp_feature.to(features[0].device))
            
        return outputs, enhanced_feature

    def _create_proposals_from_boxes(self, boxes, image_sizes):
        """
        Args:
            boxes (list[Tensor]): per-image predicted boxes, each of shape Ri x 4
            image_sizes (list[tuple]): list of image shapes in (h, w)

        Returns:
            list[Instances]: per-image proposals with the given boxes.
        """
        # Just like RPN, the proposals should not have gradients

        boxes = [Boxes(b.detach()) for b in boxes]
        proposals = []
        for boxes_per_image, image_size in zip(boxes, image_sizes):
            boxes_per_image.clip(image_size)
            if self.training:
                # do not filter empty boxes at inference time,
                # because the scores from each stage need to be aligned and added later
                boxes_per_image = boxes_per_image[boxes_per_image.nonempty()]
            prop = Instances(image_size)
            prop.proposal_boxes = boxes_per_image
            proposals.append(prop)
        return proposals
    
    def target_gt_classes_transform(self, target, num_classes_list, num_classes_list_with_bg):
        targets_stage0 = []
        targets_stage1 = []
        targets_stage2 = []
        for idx_batch, targ_per_image in enumerate(target):
            fields = targ_per_image.get_fields()
            gt_classes = fields['gt_classes']
            frequencys = fields['gt_frequencys']
            freq_mask = fields['gt_frequencys'] == 0
            comm_mask = fields['gt_frequencys'] == 1
            rare_mask = fields['gt_frequencys'] == 2
            all_mask = freq_mask + comm_mask + rare_mask
            gt_classes[comm_mask] += num_classes_list[0]
            gt_classes[rare_mask] += num_classes_list[1]
            targets_stage2.append(targ_per_image[all_mask])
#             gt_classes[rare_mask] = num_classes_list_with_bg[1] - 1 
            targets_stage1.append(targ_per_image[all_mask])
#             gt_classes[comm_mask] = num_classes_list_with_bg[0] - 1
#             gt_classes[rare_mask] = num_classes_list_with_bg[0] - 1
            targets_stage0.append(targ_per_image[all_mask])
            
#             for i, freq in enumerate(frequencys):
#                 if freq == 0:
#                     gt_classes[i] = gt_classes[i]
#                 if freq == 1:
#                     gt_classes[i] = gt_classes[i] + num_classes_list[0]
#                 if freq == 2:
#                     gt_classes[i] = gt_classes[i] + num_classes_list[1]
        return targets_stage0, targets_stage1, targets_stage2
        
        
    def split_input_to_freq(self, batched_inputs):
        f = []; c = []; r = []
        for idx, Int in enumerate(batched_inputs):
            Int_split = Int.frequency_split()
            f.append(Int_split[0])
            c.append(Int_split[1])
            r.append(Int_split[2])
        return [f, c, r]
    
    def Instance_shifter(self, instance_list, shift_number, training = True):
        for I in instance_list:
            I.classes_shifter(shift_number, train = training)
        return instance_list
           
    def Instance_combinator(self, instance_list):
        I_type = type(instance_list[0][0])
        num_batchs = len(instance_list[0])
        results = []
        for idx_batch in range(num_batchs):
            tmp_list = []
            for result in instance_list:
                Inst = result[idx_batch]
                tmp_list.append(Inst)
            new_instance = I_type.cat(tmp_list)
            results.append(new_instance)
        return results
    
    def reindex_result(self, result_list):
        """
        Combine the result.
        Args:
            result_list (list[result]): same as in :meth:`forward`
        Returns:
        """
        assert len(result_list) > 0
        num_batchs = len(result_list)
        results = []
        I_type = type(result_list[0])
        for idx_batch in range(num_batchs):
            Inst = result_list[idx_batch]

            Inst.classes_fcr_reindex(self.index_list, self.num_classes_list)

            results.append(Inst)        

        return results
