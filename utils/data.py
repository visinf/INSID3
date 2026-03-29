"""Data preprocessing."""

import numpy as np
import torch
import torch.nn.functional as F
import pycocotools.mask as mask_util
from torchvision import transforms
from typing import List


def build_transform(image_size: int) -> transforms.Compose:
    """Build the standard image transform for INSID3 inference.

    Args:
        image_size: target spatial resolution.

    Returns:
        A torchvision Compose transform.
    """
    return transforms.Compose([
        transforms.Resize(size=(image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def polygons_to_bitmask(polygons: List[np.ndarray], height: int, width: int) -> np.ndarray:
    """Convert COCO polygon annotations to a binary mask."""
    if len(polygons) == 0:
        return np.zeros((height, width)).astype(bool)
    rles = mask_util.frPyObjects(polygons, height, width)
    rle = mask_util.merge(rles)
    return mask_util.decode(rle).astype(bool)


def denormalize(tens: torch.Tensor) -> torch.Tensor:
    """Undo ImageNet normalization on a (C, H, W) tensor."""
    mean = torch.tensor([0.485, 0.456, 0.406]).reshape(3, 1, 1).to(tens.device)
    std = torch.tensor([0.229, 0.224, 0.225]).reshape(3, 1, 1).to(tens.device)
    return (tens * std) + mean


def downsample_mask(mask: torch.Tensor, h: int, w: int) -> torch.Tensor:
    """Downsample a (1, 1, H, W) binary mask to feature resolution (h, w)."""
    down = F.interpolate(mask.float(), size=(h, w), mode='bilinear', align_corners=False)[0, 0] > 0.5
    if down.sum() == 0:
        down = F.interpolate(mask.float(), size=(h, w), mode='nearest')[0, 0] > 0.5
        if down.sum() == 0:
            center = torch.argwhere(mask[0, 0] > 0).float().mean(dim=0)
            scale = mask.shape[-1] // w
            cy, cx = (center / scale).int()
            down[cy, cx] = True
    return down
