# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from typing import Iterator, List, Union
import torch

from detectron2.layers import cat
from detectron2.layers.rotated_boxes import pairwise_iou_rotated

from .boxes import Boxes


class RotatedBoxes(Boxes):
    """
    This structure stores a list of rotated boxes as a Nx5 torch.Tensor.
    It supports some common methods about boxes
    (`area`, `clip`, `nonempty`, etc),
    and also behaves like a Tensor
    (support indexing, `to(device)`, `.device`, and iteration over all boxes)
    """

    def __init__(self, tensor: torch.Tensor):
        """
        Args:
            tensor (Tensor[float]): a Nx5 matrix.  Each row is
            (x_center, y_center, width, height, angle),
            in which angle is represented in degrees.
            While there's no strict range restriction for it,
            the recommended principal range is between (-180, 180] degrees.

        Assume we have a horizontal box B = (x_center, y_center, width, height),
        where width is along the x-axis and height is along the y-axis.
        The rotated box B_rot (x_center, y_center, width, height, angle)
        can be seen as:

        1. When angle == 0:
          B_rot == B
        2. When angle > 0:
          B_rot is obtained by rotating B w.r.t its center by |angle| degrees CCW;
        3. When angle < 0:
          B_rot is obtained by rotating B w.r.t its center by |angle| degrees CW.

        Mathematically, since the right-handed coordinate system for image space
        is (y, x), where y is top->down and x is left->right, the 4 vertices of the
        rotated rectangle (yr_i, xr_i) (i = 1, 2, 3, 4) can be obtained from
        the vertices of the horizontal rectangle (y_i, x_i) (i = 1, 2, 3, 4)
        in the following way (theta = angle*pi/180 is the angle in radians,
        (y_c, x_c) is the center of the rectangle):
            yr_i = cos(theta) * (y_i - y_c) - sin(theta) * (x_i - x_c) + y_c,
            xr_i = sin(theta) * (y_i - y_c) + cos(theta) * (x_i - x_c) + x_c,
        which is the standard rigid-body rotation transformation.

        Intuitively, the angle is
        (1) the rotation angle from y-axis in image space
        to the height vector (top->down in the box's local coordinate system)
        of the box in CCW, and
        (2) the rotation angle from x-axis in image space
        to the width vector (left->right in the box's local coordinate system)
        of the box in CCW.

        More intuitively, consider the following horizontal box ABCD represented
        in (x1, y1, x2, y2):
        (3, 2, 7, 4),
        covering the [3, 7] x [2, 4] region of the continuous coordinate system
        which looks like this:

        O--------> x
        |
        |  A---B
        |  |   |
        |  D---C
        |
        v y

        Note that each capital letter represents one 0-dimensional geometric point
        instead of a 'square pixel' here.

        In the example above, using (x, y) to represent a point we have:
        O = (0, 0),
        A = (3, 2), B = (7, 2), C = (7, 4), D = (3, 4)

        We name vector AB = vector DC as the width vector in box's local coordinate system, and
        vector AD = vector BC as the height vector in box's local coordinate system. Initially,
        when angle = 0 degree, they're aligned with the positive directions of x-axis and y-axis
        in the image space, respectively.

        For better illustration, we denote the center of the box as E,

        O--------> x
        |
        |  A---B
        |  | E |
        |  D---C
        |
        v y

        where the center E = ((3+7)/2, (2+4)/2) = (5, 3).

        Also,
        width = |AB| = |CD| = 7 - 3 = 4,
        height = |AD| = |BC| = 4 - 2 = 2.

        Therefore, the corresponding representation for the same shape in rotated box in
        (x_center, y_center, width, height, angle) format is:

        (5, 3, 4, 2, 0),

        Now, let's consider (5, 3, 4, 2, 90), which is rotated by 90 degrees
        CCW (counter-clockwise) by definition. It looks like this:

        O--------> x
        |   B-C
        |   | |
        |   |E|
        |   | |
        |   A-D
        v y

        The center E is still located at the same point (5, 3), while the vertices
        ABCD are rotated by 90 degrees CCW with regard to E:
        A = (4, 5), B = (4, 1), C = (6, 1), D = (6, 5)

        Here, 90 degrees can be seen as the CCW angle to rotate from y-axis to
        vector AD or vector BC (the top->down height vector in box's local coordinate system),
        or the CCW angle to rotate from x-axis to vector AB or vector DC (the left->right
        width vector in box's local coordinate system).

        width = |AB| = |CD| = 5 - 1 = 4,
        height = |AD| = |BC| = 6 - 4 = 2.

        Next, how about (5, 3, 4, 2, -90), which is rotated by 90 degrees CW (clockwise)
        by definition? It looks like this:

        O--------> x
        |   D-A
        |   | |
        |   |E|
        |   | |
        |   C-B
        v y

        The center E is still located at the same point (5, 3), while the vertices
        ABCD are rotated by 90 degrees CW with regard to E:
        A = (6, 1), B = (6, 5), C = (4, 5), D = (4, 1)

        width = |AB| = |CD| = 5 - 1 = 4,
        height = |AD| = |BC| = 6 - 4 = 2.

        This covers exactly the same region as (5, 3, 4, 2, 90) does, and their IoU
        will be 1. However, these two will generate different RoI Pooling results and
        should not be treated as an identical box.

        On the other hand, it's easy to see that (X, Y, W, H, A) is identical to
        (X, Y, W, H, A+360N), for any integer N. For example (5, 3, 4, 2, 270) would be
        identical to (5, 3, 4, 2, -90), because rotating the shape 270 degrees CCW is
        equivalent to rotating the same shape 90 degrees CW.

        We could rotate further to get (5, 3, 4, 2, 180), or (5, 3, 4, 2, -180):

        O--------> x
        |
        |  C---D
        |  | E |
        |  B---A
        |
        v y

        A = (7, 4), B = (3, 4), C = (3, 2), D = (7, 2),

        width = |AB| = |CD| = 7 - 3 = 4,
        height = |AD| = |BC| = 4 - 2 = 2.

        Finally, this is a very inaccurate (heavily quantized) illustration of
        how (5, 3, 4, 2, 60) looks like in case anyone wonders:

        O--------> x
        |     B\
        |    /  C
        |   /E /
        |  A  /
        |   `D
        v y

        It's still a rectangle with center of (5, 3), width of 4 and height of 2,
        but its angle (and thus orientation) is somewhere between
        (5, 3, 4, 2, 0) and (5, 3, 4, 2, 90).
        """
        device = tensor.device if isinstance(tensor, torch.Tensor) else torch.device("cpu")
        tensor = torch.as_tensor(tensor, dtype=torch.float32, device=device)
        if tensor.numel() == 0:
            tensor = torch.zeros(0, 5, dtype=torch.float32, device=device)
        assert tensor.dim() == 2 and tensor.size(-1) == 5, tensor.size()

        self.tensor = tensor

    def clone(self) -> "RotatedBoxes":
        """
        Clone the RotatedBoxes.

        Returns:
            RotatedBoxes
        """
        return RotatedBoxes(self.tensor.clone())

    def to(self, device: str) -> "RotatedBoxes":
        return RotatedBoxes(self.tensor.to(device))

    def area(self) -> torch.Tensor:
        """
        Computes the area of all the boxes.

        Returns:
            torch.Tensor: a vector with areas of each box.
        """
        box = self.tensor
        area = box[:, 2] * box[:, 3]
        return area

    def normalize_angles(self) -> None:
        """
        Restrict angles to the range of (-180, 180] degrees
        """
        self.tensor[:, 4] = self.tensor[:, 4] % 360
        self.tensor[:, 4][torch.where(self.tensor[:, 4] > 180)] -= 360

    def clip(self, box_size: Boxes.BoxSizeType, clip_angle_threshold: float = 1.0) -> None:
        """
        Clip (in place) the boxes by limiting x coordinates to the range [0, width]
        and y coordinates to the range [0, height].

        For RRPN:
        Only clip boxes that are almost horizontal with a tolerance of
        clip_angle_threshold to maintain backward compatibility.

        Rotated boxes beyond this threshold are not clipped for two reasons:
        (1) There are potentially multiple ways to clip a rotated box to make it
            fit within the image.
        (2) It's tricky to make the entire rectangular box fit within the image
            and still be able to not leave out pixels of interest.
        Therefore we rely on ops like RoIAlignRotated to safely handle this.

        Args:
            box_size (height, width): The clipping box's size.
            clip_angle_threshold:
                Iff. abs(normalized(angle)) <= clip_angle_threshold (in degrees),
                we do the clipping as horizontal boxes.
        """
        h, w = box_size

        # normalize angles to be within (-180, 180] degrees
        self.normalize_angles()

        idx = torch.where(torch.abs(self.tensor[:, 4]) <= clip_angle_threshold)[0]

        # convert to (x1, y1, x2, y2)
        x1 = self.tensor[idx, 0] - self.tensor[idx, 2] / 2.0
        y1 = self.tensor[idx, 1] - self.tensor[idx, 3] / 2.0
        x2 = self.tensor[idx, 0] + self.tensor[idx, 2] / 2.0
        y2 = self.tensor[idx, 1] + self.tensor[idx, 3] / 2.0

        # clip
        x1.clamp_(min=0, max=w)
        y1.clamp_(min=0, max=h)
        x2.clamp_(min=0, max=w)
        y2.clamp_(min=0, max=h)

        # convert back to (xc, yc, w, h)
        self.tensor[idx, 0] = (x1 + x2) / 2.0
        self.tensor[idx, 1] = (y1 + y2) / 2.0
        # make sure widths and heights do not increase due to numerical errors
        self.tensor[idx, 2] = torch.min(self.tensor[idx, 2], x2 - x1)
        self.tensor[idx, 3] = torch.min(self.tensor[idx, 3], y2 - y1)

    def nonempty(self, threshold: int = 0) -> torch.Tensor:
        """
        Find boxes that are non-empty.
        A box is considered empty, if either of its side is no larger than threshold.

        Returns:
            Tensor: a binary vector which represents
                whether each box is empty (False) or non-empty (True).
        """
        box = self.tensor
        widths = box[:, 2]
        heights = box[:, 3]
        keep = (widths > threshold) & (heights > threshold)
        return keep

    def __getitem__(self, item: Union[int, slice, torch.BoolTensor]) -> "RotatedBoxes":
        """
        Returns:
            RotatedBoxes: Create a new :class:`RotatedBoxes` by indexing.

        The following usage are allowed:

        1. `new_boxes = boxes[3]`: return a `RotatedBoxes` which contains only one box.
        2. `new_boxes = boxes[2:10]`: return a slice of boxes.
        3. `new_boxes = boxes[vector]`, where vector is a torch.ByteTensor
           with `length = len(boxes)`. Nonzero elements in the vector will be selected.

        Note that the returned RotatedBoxes might share storage with this RotatedBoxes,
        subject to Pytorch's indexing semantics.
        """
        if isinstance(item, int):
            return RotatedBoxes(self.tensor[item].view(1, -1))
        b = self.tensor[item]
        assert b.dim() == 2, "Indexing on RotatedBoxes with {} failed to return a matrix!".format(
            item
        )
        return RotatedBoxes(b)

    def __len__(self) -> int:
        return self.tensor.shape[0]

    def __repr__(self) -> str:
        return "RotatedBoxes(" + str(self.tensor) + ")"

    def inside_box(self, box_size: Boxes.BoxSizeType, boundary_threshold: int = 0) -> torch.Tensor:
        """
        Args:
            box_size (height, width): Size of the reference box covering
                [0, width] x [0, height]
            boundary_threshold (int): Boxes that extend beyond the reference box
                boundary by more than boundary_threshold are considered "outside".

        For RRPN, it might not be necessary to call this function since it's common
        for rotated box to extend to outside of the image boundaries
        (the clip function only clips the near-horizontal boxes)

        Returns:
            a binary vector, indicating whether each box is inside the reference box.
        """
        height, width = box_size

        cnt_x = self.tensor[..., 0]
        cnt_y = self.tensor[..., 1]
        half_w = self.tensor[..., 2] / 2.0
        half_h = self.tensor[..., 3] / 2.0
        a = self.tensor[..., 4]
        c = torch.abs(torch.cos(a * torch.pi / 180.0))
        s = torch.abs(torch.sin(a * torch.pi / 180.0))
        # This basically computes the horizontal bounding rectangle of the rotated box
        max_rect_dx = c * half_w + s * half_h
        max_rect_dy = c * half_h + s * half_w

        inds_inside = (
            (cnt_x - max_rect_dx >= -boundary_threshold)
            & (cnt_y - max_rect_dy >= -boundary_threshold)
            & (cnt_x + max_rect_dx < width + boundary_threshold)
            & (cnt_y + max_rect_dy < height + boundary_threshold)
        )

        return inds_inside

    def get_centers(self) -> torch.Tensor:
        """
        Returns:
            The box centers in a Nx2 array of (x, y).
        """
        return self.tensor[:, :2]

    @staticmethod
    def cat(boxes_list: List["RotatedBoxes"]) -> "RotatedBoxes":  # type: ignore
        """
        Concatenates a list of RotatedBoxes into a single RotatedBoxes

        Arguments:
            boxes_list (list[RotatedBoxes])

        Returns:
            RotatedBoxes: the concatenated RotatedBoxes
        """
        assert isinstance(boxes_list, (list, tuple))
        assert len(boxes_list) > 0
        assert all(isinstance(box, RotatedBoxes) for box in boxes_list)

        cat_boxes = type(boxes_list[0])(cat([b.tensor for b in boxes_list], dim=0))
        return cat_boxes

    @property
    def device(self) -> str:
        return self.tensor.device

    def __iter__(self) -> Iterator[torch.Tensor]:
        """
        Yield a box as a Tensor of shape (5,) at a time.
        """
        yield from self.tensor


def pairwise_iou(boxes1: RotatedBoxes, boxes2: RotatedBoxes) -> None:
    """
    Given two lists of rotated boxes of size N and M,
    compute the IoU (intersection over union)
    between __all__ N x M pairs of boxes.
    The box order must be (x_center, y_center, width, height, angle).

    Args:
        boxes1, boxes2 (RotatedBoxes):
            two `RotatedBoxes`. Contains N & M rotated boxes, respectively.

    Returns:
        Tensor: IoU, sized [N,M].
    """

    return pairwise_iou_rotated(boxes1.tensor, boxes2.tensor)
