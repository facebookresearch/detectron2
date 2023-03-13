import math
import torch
import torch.distributed as dist

from detectron2.modeling.roi_heads import FastRCNNConvFCHead, MaskRCNNConvUpsampleHead
from detectron2.utils import comm
from fvcore.nn.distributed import differentiable_all_gather


def concat_all_gather(input):
    bs_int = input.shape[0]
    size_list = comm.all_gather(bs_int)
    max_size = max(size_list)
    max_shape = (max_size,) + input.shape[1:]

    padded_input = input.new_zeros(max_shape)
    padded_input[:bs_int] = input
    all_inputs = differentiable_all_gather(padded_input)
    inputs = [x[:sz] for sz, x in zip(size_list, all_inputs)]
    return inputs, size_list


def batch_shuffle(x):
    # gather from all gpus
    batch_size_this = x.shape[0]
    all_xs, batch_size_all = concat_all_gather(x)
    all_xs_concat = torch.cat(all_xs, dim=0)
    total_bs = sum(batch_size_all)

    rank = dist.get_rank()
    assert batch_size_all[rank] == batch_size_this

    idx_range = (sum(batch_size_all[:rank]), sum(batch_size_all[: rank + 1]))

    # random shuffle index
    idx_shuffle = torch.randperm(total_bs, device=x.device)
    # broadcast to all gpus
    dist.broadcast(idx_shuffle, src=0)

    # index for restoring
    idx_unshuffle = torch.argsort(idx_shuffle)

    # shuffled index for this gpu
    splits = torch.split(idx_shuffle, math.ceil(total_bs / dist.get_world_size()))
    if len(splits) > rank:
        idx_this = splits[rank]
    else:
        idx_this = idx_shuffle.new_zeros([0])
    return all_xs_concat[idx_this], idx_unshuffle[idx_range[0] : idx_range[1]]


def batch_unshuffle(x, idx_unshuffle):
    all_x, _ = concat_all_gather(x)
    x_gather = torch.cat(all_x, dim=0)
    return x_gather[idx_unshuffle]


def wrap_shuffle(module_type, method):
    def new_method(self, x):
        if self.training:
            x, idx = batch_shuffle(x)
        x = getattr(module_type, method)(self, x)
        if self.training:
            x = batch_unshuffle(x, idx)
        return x

    return type(module_type.__name__ + "WithShuffle", (module_type,), {method: new_method})


from .mask_rcnn_BNhead import model, dataloader, lr_multiplier, optimizer, train


model.roi_heads.box_head._target_ = wrap_shuffle(FastRCNNConvFCHead, "forward")
model.roi_heads.mask_head._target_ = wrap_shuffle(MaskRCNNConvUpsampleHead, "layers")
