"""
An example config file to train a ImageNet classifier with detectron2.
Model and dataloader both come from torchvision.
This shows how to use detectron2 as a general engine for any new models and tasks.

To run, use the following command:

python tools/lazyconfig_train_net.py --config-file configs/Misc/torchvision_imagenet_R_50.py \
    --num-gpus 8 dataloader.train.dataset.root=/path/to/imagenet/

"""

import torch
from torch import nn
from torch.nn import functional as F
from omegaconf import OmegaConf
import torchvision
from torchvision.transforms import transforms as T
from torchvision.models.resnet import ResNet, Bottleneck
from fvcore.common.param_scheduler import MultiStepParamScheduler

from detectron2.solver import WarmupParamScheduler
from detectron2.solver.build import get_default_optimizer_params
from detectron2.config import LazyCall as L
from detectron2.model_zoo import get_config
from detectron2.data.samplers import TrainingSampler, InferenceSampler
from detectron2.evaluation import DatasetEvaluator
from detectron2.utils import comm


"""
Note: Here we put reusable code (models, evaluation, data) together with configs just as a
proof-of-concept, to easily demonstrate what's needed to train a ImageNet classifier in detectron2.
Writing code in configs offers extreme flexibility but is often not a good engineering practice.
In practice, you might want to put code in your project and import them instead.
"""


def build_data_loader(dataset, batch_size, num_workers, training=True):
    return torch.utils.data.DataLoader(
        dataset,
        sampler=(TrainingSampler if training else InferenceSampler)(len(dataset)),
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
    )


class ClassificationNet(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    @property
    def device(self):
        return list(self.model.parameters())[0].device

    def forward(self, inputs):
        image, label = inputs
        pred = self.model(image.to(self.device))
        if self.training:
            label = label.to(self.device)
            return F.cross_entropy(pred, label)
        else:
            return pred


class ClassificationAcc(DatasetEvaluator):
    def reset(self):
        self.corr = self.total = 0

    def process(self, inputs, outputs):
        image, label = inputs
        self.corr += (outputs.argmax(dim=1).cpu() == label.cpu()).sum().item()
        self.total += len(label)

    def evaluate(self):
        all_corr_total = comm.all_gather([self.corr, self.total])
        corr = sum(x[0] for x in all_corr_total)
        total = sum(x[1] for x in all_corr_total)
        return {"accuracy": corr / total}


# --- End of code that could be in a project and be imported


dataloader = OmegaConf.create()
dataloader.train = L(build_data_loader)(
    dataset=L(torchvision.datasets.ImageNet)(
        root="/path/to/imagenet",
        split="train",
        transform=L(T.Compose)(
            transforms=[
                L(T.RandomResizedCrop)(size=224),
                L(T.RandomHorizontalFlip)(),
                T.ToTensor(),
                L(T.Normalize)(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        ),
    ),
    batch_size=256 // 8,
    num_workers=4,
    training=True,
)

dataloader.test = L(build_data_loader)(
    dataset=L(torchvision.datasets.ImageNet)(
        root="${...train.dataset.root}",
        split="val",
        transform=L(T.Compose)(
            transforms=[
                L(T.Resize)(size=256),
                L(T.CenterCrop)(size=224),
                T.ToTensor(),
                L(T.Normalize)(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        ),
    ),
    batch_size=256 // 8,
    num_workers=4,
    training=False,
)

dataloader.evaluator = L(ClassificationAcc)()

model = L(ClassificationNet)(
    model=(ResNet)(block=Bottleneck, layers=[3, 4, 6, 3], zero_init_residual=True)
)


optimizer = L(torch.optim.SGD)(
    params=L(get_default_optimizer_params)(),
    lr=0.1,
    momentum=0.9,
    weight_decay=1e-4,
)

lr_multiplier = L(WarmupParamScheduler)(
    scheduler=L(MultiStepParamScheduler)(
        values=[1.0, 0.1, 0.01, 0.001], milestones=[30, 60, 90, 100]
    ),
    warmup_length=1 / 100,
    warmup_factor=0.1,
)


train = get_config("common/train.py").train
train.init_checkpoint = None
train.max_iter = 100 * 1281167 // 256
