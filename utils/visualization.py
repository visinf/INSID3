"""Visualization helpers."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image


def _load_image(image: str | Path | Image.Image) -> Image.Image:
    if isinstance(image, (str, Path)):
        return Image.open(image).convert("RGB")
    return image.convert("RGB")

# ──────── utils for in-context segmentation visualization ────────

def _load_mask(mask: str | Path | Image.Image | np.ndarray | torch.Tensor, size: tuple[int, int]) -> np.ndarray:
    if isinstance(mask, torch.Tensor):
        mask_array = mask.detach().to("cpu")
        if mask_array.ndim > 2:
            mask_array = mask_array.squeeze()
        mask_array = mask_array.numpy()
    elif isinstance(mask, np.ndarray):
        mask_array = mask
    elif isinstance(mask, (str, Path)):
        mask_array = np.array(Image.open(mask))
    else:
        mask_array = np.array(mask)

    mask_array = mask_array.squeeze()
    mask_array = mask_array > 0

    if mask_array.shape != size:
        mask_image = Image.fromarray(mask_array.astype(np.uint8) * 255)
        mask_array = np.array(mask_image.resize((size[1], size[0]), resample=Image.NEAREST)) > 0

    return mask_array


def _overlay_mask(image: np.ndarray, mask: np.ndarray, color: tuple[float, float, float], alpha: float) -> np.ndarray:
    overlay = image.astype(np.float32).copy()
    color_arr = np.array(color, dtype=np.float32) * 255.0
    overlay[mask] = (1.0 - alpha) * overlay[mask] + alpha * color_arr
    return np.clip(overlay, 0, 255).astype(np.uint8)


def visualize_prediction_segmentation(
    reference_image: str | Path | Image.Image,
    reference_mask: str | Path | Image.Image | np.ndarray | torch.Tensor,
    target_image: str | Path | Image.Image,
    predicted_mask: str | Path | Image.Image | np.ndarray | torch.Tensor,
    output_path: str | Path | None = None,
    *,
    alpha: float = 0.45,
    visualize: bool = False,
) -> None:
    """Save or show a visualization of the reference and target segmentation."""
    reference_pil = _load_image(reference_image)
    target_pil = _load_image(target_image)

    reference_np = np.array(reference_pil)
    target_np = np.array(target_pil)

    reference_mask_np = _load_mask(reference_mask, reference_np.shape[:2])
    predicted_mask_np = _load_mask(predicted_mask, target_np.shape[:2])

    reference_overlay = _overlay_mask(reference_np, reference_mask_np, color=(0.95, 0.25, 0.2), alpha=alpha)
    target_overlay = _overlay_mask(target_np, predicted_mask_np, color=(0.15, 0.8, 0.35), alpha=alpha)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6), constrained_layout=True)
    axes[0].imshow(reference_overlay)
    axes[0].set_title("Reference + Mask")
    axes[1].imshow(target_overlay)
    axes[1].set_title("Target + Prediction")

    for axis in axes:
        axis.axis("off")

    if visualize:
        plt.show()
        plt.close(fig)
        return

    if output_path is None:
        raise ValueError("output_path must be provided when visualize is False")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ──────── utils for semantic correspondence visualization ────────

# Palette — up to 10 keypoints; falls back to a generated colormap for more
_PALETTE = [
    '#e6194b', '#3cb44b', '#4363d8',
    '#f58231', '#911eb4', '#42d4f4',
    '#f032e6', '#bfef45', '#fabed4', '#469990',
]


def _get_colors(n: int):
    if n <= len(_PALETTE):
        return _PALETTE[:n]
    cmap = plt.cm.get_cmap('hsv', n)
    return [cmap(i) for i in range(n)]


def visualize_prediction_matching(
    reference_image: str | Path | Image.Image,
    reference_keypoints,
    target_image: str | Path | Image.Image,
    predicted_keypoints,
    output_path: str | Path | None = None,
    *,
    grid_steps: int = 6,
    visualize: bool = False,
) -> None:
    """Save or show a semantic-correspondence visualization."""
    reference_pil = _load_image(reference_image)
    target_pil = _load_image(target_image)

    reference_kps = reference_keypoints
    predicted_kps = predicted_keypoints.detach().cpu().tolist()

    colors = _get_colors(len(reference_kps))

    ref_w, ref_h = reference_pil.size
    tgt_w, tgt_h = target_pil.size

    fig, axes = plt.subplots(1, 2, figsize=(12, 6), constrained_layout=True)

    def _setup_axis_with_grid(ax, img, title: str, width: int, height: int, y_ticks_side: str = "left") -> None:
        ax.imshow(img)
        ax.set_title(title, fontsize=13)

        ax.set_xlim(0, width)
        ax.set_ylim(height, 0)

        xticks = np.linspace(0, width, grid_steps + 1)
        yticks = np.linspace(0, height, grid_steps + 1)

        ax.set_xticks(xticks)
        ax.set_yticks(yticks)
        ax.set_xticklabels([f"{int(round(x))}" for x in xticks], fontsize=8)
        ax.set_yticklabels([f"{int(round(y))}" for y in yticks], fontsize=8)

        ax.grid(True, color="white", alpha=0.28, linewidth=0.8)

        ax.tick_params(axis="x", which="both", bottom=True, top=False, labelbottom=True, length=3, color="white")

        if y_ticks_side == "left":
            ax.tick_params(axis="y", which="both", left=True, right=False, labelleft=True, labelright=False, length=3, color="white")
        else:
            ax.tick_params(axis="y", which="both", left=False, right=True, labelleft=False, labelright=True, length=3, color="white")

        for spine in ax.spines.values():
            spine.set_visible(False)

    _setup_axis_with_grid(
        axes[0], reference_pil, "Reference Image + Keypoints", ref_w, ref_h, y_ticks_side="left"
    )
    for i, (x, y) in enumerate(reference_kps):
        axes[0].plot(
            x, y, "o",
            color=colors[i],
            markersize=12,
            markeredgecolor="white",
            markeredgewidth=1.3,
        )
        axes[0].text(
            x + 0.02 * ref_w, y, f"{i + 1}",
            color=colors[i],
            fontsize=13,
            weight="bold",
            va="center",
            ha="left",
        )

    _setup_axis_with_grid(
        axes[1], target_pil, "Target Image + Predicted Keypoints", tgt_w, tgt_h, y_ticks_side="right"
    )
    for i, (x, y) in enumerate(predicted_kps):
        axes[1].plot(
            x, y, "o",
            color=colors[i],
            markersize=12,
            markeredgecolor="white",
            markeredgewidth=1.3,
        )
        axes[1].text(
            x + 0.02 * tgt_w, y, f"{i + 1}",
            color=colors[i],
            fontsize=13,
            weight="bold",
            va="center",
            ha="left",
        )

    if visualize:
        plt.show()
        plt.close(fig)
        return

    if output_path is None:
        raise ValueError("output_path must be provided when visualize is False")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)