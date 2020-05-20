# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from torch import nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.nn.modules.utils import _pair

from detectron2 import _C


class _ROIAlign(Function):
    @staticmethod
    def forward(ctx, input, roi, output_size, spatial_scale, sampling_ratio, aligned):
        ctx.save_for_backward(roi)
        ctx.output_size = _pair(output_size)
        ctx.spatial_scale = spatial_scale
        ctx.sampling_ratio = sampling_ratio
        ctx.input_shape = input.size()
        ctx.aligned = aligned
        output = _C.roi_align_forward(
            input, roi, spatial_scale, output_size[0], output_size[1], sampling_ratio, aligned
        )
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        (rois,) = ctx.saved_tensors
        output_size = ctx.output_size
        spatial_scale = ctx.spatial_scale
        sampling_ratio = ctx.sampling_ratio
        bs, ch, h, w = ctx.input_shape
        grad_input = _C.roi_align_backward(
            grad_output,
            rois,
            spatial_scale,
            output_size[0],
            output_size[1],
            bs,
            ch,
            h,
            w,
            sampling_ratio,
            ctx.aligned,
        )
        return grad_input, None, None, None, None, None


roi_align = _ROIAlign.apply


class ROIAlign(nn.Module):
    def __init__(self, output_size, spatial_scale, sampling_ratio, aligned=True):
        """
        Args:
            output_size (tuple): h, w
            spatial_scale (float): scale the input boxes by this number
            sampling_ratio (int): number of inputs samples to take for each output
                sample. 0 to take samples densely.
            aligned (bool): if False, use the legacy implementation in
                Detectron. If True, align the results more perfectly.

        Note:
            The meaning of aligned=True:

            Given a continuous coordinate c, its two neighboring pixel indices (in our
            pixel model) are computed by floor(c - 0.5) and ceil(c - 0.5). For example,
            c=1.3 has pixel neighbors with discrete indices [0] and [1] (which are sampled
            from the underlying signal at continuous coordinates 0.5 and 1.5). But the original
            roi_align (aligned=False) does not subtract the 0.5 when computing neighboring
            pixel indices and therefore it uses pixels with a slightly incorrect alignment
            (relative to our pixel model) when performing bilinear interpolation.

            With `aligned=True`,
            we first appropriately scale the ROI and then shift it by -0.5
            prior to calling roi_align. This produces the correct neighbors; see
            detectron2/tests/test_roi_align.py for verification.

            The difference does not make a difference to the model's performance if
            ROIAlign is used together with conv layers.
        """
        super(ROIAlign, self).__init__()
        self.output_size = output_size
        self.spatial_scale = spatial_scale
        self.sampling_ratio = sampling_ratio
        self.aligned = aligned

    def forward(self, input, rois):
        """
        Args:
            input: NCHW images
            rois: Bx5 boxes. First column is the index into N. The other 4 columns are xyxy.
        """
        assert rois.dim() == 2 and rois.size(1) == 5
        return roi_align(
            input, rois, self.output_size, self.spatial_scale, self.sampling_ratio, self.aligned
        )

    def __repr__(self):
        tmpstr = self.__class__.__name__ + "("
        tmpstr += "output_size=" + str(self.output_size)
        tmpstr += ", spatial_scale=" + str(self.spatial_scale)
        tmpstr += ", sampling_ratio=" + str(self.sampling_ratio)
        tmpstr += ", aligned=" + str(self.aligned)
        tmpstr += ")"
        return tmpstr
