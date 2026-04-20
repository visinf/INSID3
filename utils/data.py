"""Data preprocessing."""

import numpy as np
import torch
import torch.nn.functional as F
import pycocotools.mask as mask_util
from PIL import Image
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


def load_image(
    image: str | Image.Image | torch.Tensor,
    transform: transforms.Compose,
    device: torch.device | str,
) -> tuple[torch.Tensor, tuple[int, int]]:
    """Load an image as a batched tensor with shape (1, C, H, W)."""
    if isinstance(image, str):
        image = Image.open(image).convert("RGB")
        orig_size = (image.height, image.width)
        image_tensor = transform(image).unsqueeze(0)
    elif isinstance(image, torch.Tensor):
        image_tensor = image
        if image_tensor.ndim not in (3, 4):
            raise ValueError("Tensor image must have shape (C,H,W) or (1,C,H,W)")
        image_tensor = image_tensor.reshape(-1, image_tensor.shape[-3], image_tensor.shape[-2], image_tensor.shape[-1])
        if image_tensor.shape[0] != 1:
            raise ValueError("Tensor image must have shape (C,H,W) or (1,C,H,W)")
        orig_size = (image_tensor.shape[-2], image_tensor.shape[-1])
        image_tensor = F.interpolate(
            image_tensor,
            size=transform.transforms[0].size,
            mode="bilinear",
            align_corners=False,
        )
    else:
        orig_size = (image.height, image.width)
        image_tensor = transform(image).unsqueeze(0)

    return image_tensor.to(device), orig_size


def load_mask(
    mask: str | Image.Image | torch.Tensor,
    image_size: int,
    device: torch.device | str,
) -> torch.Tensor:
    """Load a binary mask as a tensor with shape (1, H, W)."""
    if isinstance(mask, torch.Tensor):
        mask_tensor = mask.unsqueeze(0) if mask.ndim == 2 else mask
        if mask_tensor.ndim != 3 or mask_tensor.shape[0] != 1:
            raise ValueError("Tensor mask must have shape (H,W) or (1,H,W)")
        mask_tensor = mask_tensor.to(device)
        if mask_tensor.dtype != torch.bool:
            mask_tensor = mask_tensor > 0
    else:
        if isinstance(mask, str):
            mask = Image.open(mask)
        mask_tensor = torch.tensor(
            np.array(mask) > 0, dtype=torch.bool
        ).unsqueeze(0).to(device)

    return F.interpolate(
        mask_tensor.unsqueeze(0).float(),
        size=(image_size, image_size),
        mode="nearest",
    ).squeeze(0) > 0.5


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
