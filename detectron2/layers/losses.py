import math
import torch


def diou_loss(
    boxes1: torch.Tensor,
    boxes2: torch.Tensor,
    reduction: str = "none",
    eps: float = 1e-7,
) -> torch.Tensor:
    """
    Distance Intersection over Union Loss (Zhaohui Zheng et. al)
    https://arxiv.org/abs/1911.08287
    Args:
        boxes1, boxes2 (Tensor): box locations in XYXY format, shape (N, 4) or (4,).
        reduction: 'none' | 'mean' | 'sum'
                 'none': No reduction will be applied to the output.
                 'mean': The output will be averaged.
                 'sum': The output will be summed.
        eps (float): small number to prevent division by zero
    """

    x1, y1, x2, y2 = boxes1.unbind(dim=-1)
    x1g, y1g, x2g, y2g = boxes2.unbind(dim=-1)

    assert (x2 >= x1).all(), "bad box: x1 larger than x2"
    assert (y2 >= y1).all(), "bad box: y1 larger than y2"

    x_p = (x2 + x1) / 2
    y_p = (y2 + y1) / 2
    x_g = (x1g + x2g) / 2
    y_g = (y1g + y2g) / 2

    xkis1 = torch.max(x1, x1g)
    ykis1 = torch.max(y1, y1g)
    xkis2 = torch.min(x2, x2g)
    ykis2 = torch.min(y2, y2g)

    intsctk = torch.zeros_like(x1)
    mask = (ykis2 > ykis1) & (xkis2 > xkis1)
    intsctk[mask] = (xkis2[mask] - xkis1[mask]) * (ykis2[mask] - ykis1[mask])
    unionk = (x2 - x1) * (y2 - y1) + (x2g - x1g) * (y2g - y1g) - intsctk + eps
    iouk = intsctk / unionk

    xc1 = torch.min(x1, x1g)
    yc1 = torch.min(y1, y1g)
    xc2 = torch.max(x2, x2g)
    yc2 = torch.max(y2, y2g)

    c = ((xc2 - xc1) ** 2) + ((yc2 - yc1) ** 2) + eps
    d = ((x_p - x_g) ** 2) + ((y_p - y_g) ** 2)
    u = d / c
    diouk = iouk - u

    loss = 1 - diouk
    if reduction == "mean":
        loss = loss.mean() if loss.numel() > 0 else 0.0 * loss.sum()
    elif reduction == "sum":
        loss = loss.sum()

    return loss


def ciou_loss(
    boxes1: torch.Tensor,
    boxes2: torch.Tensor,
    reduction: str = "none",
    eps: float = 1e-7,
) -> torch.Tensor:
    """
    Complete Intersection over Union Loss (Zhaohui Zheng et. al)
    https://arxiv.org/abs/1911.08287
    Args:
        boxes1, boxes2 (Tensor): box locations in XYXY format, shape (N, 4) or (4,).
        reduction: 'none' | 'mean' | 'sum'
                 'none': No reduction will be applied to the output.
                 'mean': The output will be averaged.
                 'sum': The output will be summed.
        eps (float): small number to prevent division by zero
    """

    x1, y1, x2, y2 = boxes1.unbind(dim=-1)
    x1g, y1g, x2g, y2g = boxes2.unbind(dim=-1)

    assert (x2 >= x1).all(), "bad box: x1 larger than x2"
    assert (y2 >= y1).all(), "bad box: y1 larger than y2"

    w_pred = x2 - x1
    h_pred = y2 - y1
    w_gt = x2g - x1g
    h_gt = y2g - y1g

    x_center = (x2 + x1) / 2
    y_center = (y2 + y1) / 2
    x_center_g = (x1g + x2g) / 2
    y_center_g = (y1g + y2g) / 2

    xkis1 = torch.max(x1, x1g)
    ykis1 = torch.max(y1, y1g)
    xkis2 = torch.min(x2, x2g)
    ykis2 = torch.min(y2, y2g)

    intsctk = torch.zeros_like(x1)
    mask = (ykis2 > ykis1) & (xkis2 > xkis1)
    intsctk[mask] = (xkis2[mask] - xkis1[mask]) * (ykis2[mask] - ykis1[mask])
    unionk = (x2 - x1) * (y2 - y1) + (x2g - x1g) * (y2g - y1g) - intsctk + eps
    iouk = intsctk / unionk

    xc1 = torch.min(x1, x1g)
    yc1 = torch.min(y1, y1g)
    xc2 = torch.max(x2, x2g)
    yc2 = torch.max(y2, y2g)

    c = ((xc2 - xc1) ** 2) + ((yc2 - yc1) ** 2) + eps
    d = ((x_center - x_center_g) ** 2) + ((y_center - y_center_g) ** 2)
    u = d / c
    v = (4 / (math.pi ** 2)) * torch.pow((torch.atan(w_gt / h_gt) - torch.atan(w_pred / h_pred)), 2)
    with torch.no_grad():
        S = 1 - iouk
        alpha = v / (S + v + eps)
    ciouk = iouk - (u + alpha * v)

    loss = 1 - ciouk
    if reduction == "mean":
        loss = loss.mean() if loss.numel() > 0 else 0.0 * loss.sum()
    elif reduction == "sum":
        loss = loss.sum()

    return loss
