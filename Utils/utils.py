import torch
import numpy as np

def complete_path(path):
    return (path + '/') if path[-1] != '/' else path


def IoU(boxes1, boxes2, scale=None, eps=1e-8):
    """
    Calculates Intersection over Union (IoU) between bounding boxes.

    Args:
        boxes1 (torch.Tensor or np.ndarray): (batch_size, S, S, [x_cell, y_cell, w, h])
        boxes2 (torch.Tensor or np.ndarray): (batch_size, S, S, [x_cell, y_cell, w, h])
        scale (int, optional): Scaling factor. Defaults to None.
        eps (float, optional): Small value to avoid division by zero. Defaults to 1e-8.

    Returns:
        torch.Tensor or np.ndarray: IoU values.
    """
    outside_torch = not (isinstance(boxes1, torch.Tensor) and isinstance(boxes2, torch.Tensor))
    if outside_torch:
        boxes1 = torch.tensor(boxes1)
        boxes2 = torch.tensor(boxes2)

    x1, y1, w1, h1 = boxes1[..., 0], boxes1[..., 1], boxes1[..., 2], boxes1[..., 3]
    x2, y2, w2, h2 = boxes2[..., 0], boxes2[..., 1], boxes2[..., 2], boxes2[..., 3]

    if scale:
        x1, y1, x2, y2 = x1 / scale, y1 / scale, x2 / scale, y2 / scale

    x_min, x_max = torch.max(x1 - w1 / 2, x2 - w2 / 2), torch.min(x1 + w1 / 2, x2 + w2 / 2)
    y_min, y_max = torch.max(y1 - h1 / 2, y2 - h2 / 2), torch.min(y1 + h1 / 2, y2 + h2 / 2)

    intersections = torch.max(torch.zeros_like(x_min), x_max - x_min) * torch.max(torch.zeros_like(y_min),
                                                                                   y_max - y_min)
    unions = w1 * h1 + w2 * h2 - intersections

    if outside_torch:
        return (intersections / (unions + eps)).cpu().numpy()

    return intersections / (unions + eps)


def get_grid(n):
    """
    Calculates the grid dimensions (rows, cols) for displaying 'n' images.

    Args:
        n (int): Number of images.

    Returns:
        tuple: (rows, cols)
    """
    m = n
    couple = (0, 0)
    for d in range(1, int(np.sqrt(n)) + 1):
        if n % d == 0:
            q = n // d
            if q - d < m:
                m = q - d
                couple = (d, q)
    return couple


def get_affine_item(i, affine_params):
    """
    Extracts affine transformation parameters for a specific item in a batch.

    Args:
        i (int): Index of the item in the batch.
        affine_params (tuple): Affine transformation parameters.

    Returns:
        tuple: Extracted affine parameters.
    """
    res = ()
    for param in affine_params:
        if isinstance(param, torch.Tensor):
            res += (param[i].item(),)
        else:
            x = ()
            for p in param:
                x += (p[i].item(),)
            res += (x,)
    return res