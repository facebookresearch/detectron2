from detectron2.data.detection_utils import create_keypoint_hflip_indices

from .coco import dataloader

dataloader.train.dataset.min_keypoints = 1
dataloader.train.dataset.names = "keypoints_coco_2017_train"
dataloader.test.dataset.names = "keypoints_coco_2017_val"

dataloader.train.mapper.update(
    use_instance_mask=False,
    use_keypoint=True,
    keypoint_hflip_indices=create_keypoint_hflip_indices(dataloader.train.dataset.names),
)
