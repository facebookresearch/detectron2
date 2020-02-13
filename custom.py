from detectron2.data import build_detection_train_loader, build_detection_test_loader
from detectron2.engine import DefaultTrainer
from detectron2.data import transforms as T
from detectron2.data import detection_utils as utils
from detectron2.structures import BoxMode
import copy
import numpy as np
import torchvision
import torch


def apply_mask(image, mask):
    mask = np.concatenate([np.expand_dims(mask, -1)] * 3, axis=-1)
    mask = np.int32(mask)
    return image * mask


def cutout(n_mask, size, length, p):
    """
    n_mask: the number of mask per section, int
    size: the size of section, int
    length: length of mask. Int.
    p: probability of mask, Float.
    
    Return
        _cutout: Function for cutout
    """

    def _cutout(image):
        h, w, _ = image.shape
        mask = np.ones((h, w), np.float32)
        for sh in range(h // size):
            for sw in range(w // size):
                for _ in range(n_mask):
                    if np.random.random() > p:
                        continue
                    y = np.random.randint(sh * size, (sh + 1) * size)
                    x = np.random.randint(sw * size, (sw + 1) * size)
                    y1 = np.clip(y - length // 2, 0, h)
                    y2 = np.clip(y + length // 2, 0, h)
                    x1 = np.clip(x - length // 2, 0, w)
                    x2 = np.clip(x + length // 2, 0, w)
                    mask[y1:y2, x1:x2] = 0.0
        mask = np.concatenate([np.expand_dims(mask, -1)] * 3, axis=-1)
        mask = np.int32(mask)
        return image * mask

    return _cutout


# custome ransform_instance_annotations for Rotated bbox
def transform_instance_annotations(annotation, transforms, image_size):
    bbox = np.asarray([annotation["bbox"]])
    annotation["bbox"] = transforms.apply_rotated_box(bbox)[0]
    annotation["bbox_mode"] = BoxMode.XYWHA_ABS
    return annotation


class CDatasetMapper:
    """
    A callable which takes a dataset dict in Detectron2 Dataset format,
    and map it into a format used by the model.

    This is the default callable to be used to map your dataset dict into training data.
    You may need to follow it to implement your own one for customized logic.

    The callable currently does the following:

    1. Read the image from "file_name"
    2. Applies cropping/geometric transforms to the image and annotations
    3. Prepare data and annotations to Tensor and :class:`Instances`
    """

    def __init__(self, cfg, is_train=True):

        self.tfm_gens = utils.build_transform_gen(cfg, is_train)
        self.cutout_trf = torchvision.transforms.Compose(
            [
                cutout(
                    20, 200, 20, 0.8
                )
            ]
        )

        # fmt: off
        self.img_format     = cfg.INPUT.FORMAT
        self.mask_on        = cfg.MODEL.MASK_ON
        self.mask_format    = cfg.INPUT.MASK_FORMAT
        self.keypoint_on    = cfg.MODEL.KEYPOINT_ON
        self.load_proposals = cfg.MODEL.LOAD_PROPOSALS
        # fmt: on
        if self.keypoint_on and is_train:
            # Flip only makes sense in training
            self.keypoint_hflip_indices = utils.create_keypoint_hflip_indices(
                cfg.DATASETS.TRAIN
            )
        else:
            self.keypoint_hflip_indices = None

        if self.load_proposals:
            self.min_box_side_len = cfg.MODEL.PROPOSAL_GENERATOR.MIN_SIZE
            self.proposal_topk = (
                cfg.DATASETS.PRECOMPUTED_PROPOSAL_TOPK_TRAIN
                if is_train
                else cfg.DATASETS.PRECOMPUTED_PROPOSAL_TOPK_TEST
            )
        self.is_train = is_train

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        # USER: Write your own image loading if it's not from a file
        image = utils.read_image(dataset_dict["file_name"], format=self.img_format)
        utils.check_image_size(dataset_dict, image)

        image, transforms = T.apply_transform_gens(self.tfm_gens, image)
        image_shape = image.shape[:2]  # h, w

        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.

        if self.is_train:
            image = self.cutout_trf(image)

        dataset_dict["image"] = torch.as_tensor(
            np.ascontiguousarray(image.transpose(2, 0, 1))
        )

        if "annotations" in dataset_dict:
            # USER: Modify this if you want to keep them for some reason.
            for anno in dataset_dict["annotations"]:
                if not self.mask_on:
                    anno.pop("segmentation", None)
                if not self.keypoint_on:
                    anno.pop("keypoints", None)

            annos = [
                transform_instance_annotations(obj, transforms, image_shape)
                for obj in dataset_dict.pop("annotations")
                if obj.get("iscrowd", 0) == 0
            ]

            instances = utils.annotations_to_instances_rotated(annos, image_shape)

            dataset_dict["instances"] = utils.filter_empty_instances(instances)
        return dataset_dict

class Trainer(DefaultTrainer):
    
    def __init__(self, cfg):
        super(Trainer, self).__init__(cfg)
    
    @classmethod
    def build_test_loader(cls, cfg):
        print('# custome test loader!!')
        return build_detection_test_loader(cfg, CDatasetMapper(cfg, False))

    @classmethod
    def build_train_loader(cls, cfg):
        print('# custome train loader!!')
        return build_detection_train_loader(cfg, CDatasetMapper(cfg, True))