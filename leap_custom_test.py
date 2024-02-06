import tensorflow as tf
import numpy as np
import onnxruntime as rt

from effizency.metrics.detectron2_loss import calc_detectron2_loss
from effizency.metrics.metrics import calc_mean_mask_iou
from effizency.utils.visualization_utils import draw_image_with_masks, draw_image_with_boxes
from effizency.visualizers import bb_gt_visualizer, prediction_bb_visualizer, gt_mask_visualizer, pred_mask_visualizer
from leap_binder import preprocess_func, input_encoder, bbox_gt_encoder, polygons_gt_encoder, \
    original_image_shape_encoder, raw_image_encoder, masks_gt_encoder, get_metadata_dict

if __name__ == '__main__':
    sess = rt.InferenceSession('model/modified_model.onnx')
    response = preprocess_func()[0]

    for idx in range(8):
        input_image = input_encoder(idx, response)
        raw_input_image = np.expand_dims(raw_image_encoder(idx, response), 0)
        bbox_gt = np.expand_dims(bbox_gt_encoder(idx, response), 0)
        polygons = np.expand_dims(polygons_gt_encoder(idx, response), 0)
        mask = np.expand_dims(masks_gt_encoder(idx, response), 0)
        batched_input = np.expand_dims(input_image, 0)
        original_image_shape = original_image_shape_encoder(idx, response)

        metadata = get_metadata_dict(idx, response)
        # Run inference
        input_name = sess.get_inputs()[0].name  # Get input name from model
        output_names = [output.name for output in sess.get_outputs()]  # Get output name from model

        onnx_result = sess.run(output_names, {input_name: batched_input.transpose((0, 3, 1, 2))})

        results_shapes = dict(zip(output_names, [r.shape for r in onnx_result]))
        tf_result = [tf.convert_to_tensor(r) for r in onnx_result]

        # Test Visualizer
        bb_gt_vis = bb_gt_visualizer(image=raw_input_image[0], bboxes=bbox_gt[0, ...])
        draw_image_with_boxes(bb_gt_vis.data / 255., bb_gt_vis.bounding_boxes)

        pred_bb_vis = prediction_bb_visualizer(image=raw_input_image[0],
                                               bboxes=tf_result[0].numpy(),
                                               scores=tf_result[1].numpy(),
                                               class_ids=tf_result[2].numpy())
        draw_image_with_boxes(pred_bb_vis.data / 255., pred_bb_vis.bounding_boxes)

        gt_mask_vis = gt_mask_visualizer(image=raw_input_image[0], mask=mask[0])
        draw_image_with_masks(gt_mask_vis.image / 255., gt_mask_vis.mask, gt_mask_vis.labels)

        pred_mask_vis = pred_mask_visualizer(image=raw_input_image[0], masks=tf_result[3].numpy(),
                                             bboxes=tf_result[0].numpy())
        draw_image_with_masks(pred_mask_vis.image / 255., pred_mask_vis.mask, pred_mask_vis.labels)

        # Calculate metrics
        mask_iou = calc_mean_mask_iou(gt_masks=tf.convert_to_tensor(mask), pred_masks=tf_result[3],
                                      pred_bboxes=tf_result[0])

        # Calculate loss
        loss = calc_detectron2_loss(gt_boxes=tf.convert_to_tensor(bbox_gt), anchors=tf_result[4],
                                    pred_objectness_logits=tf_result[5], pred_anchor_deltas=tf_result[6],
                                    cls_loss_predictions=tf_result[7], box_loss_predictions=tf_result[8],
                                    proposal_boxes=tf_result[10], proposal_logits=tf_result[11],
                                    gt_polygons=tf.convert_to_tensor(polygons),
                                    mask_features=tf_result[9])
