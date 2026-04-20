import torch
from torch import Tensor


def rescale_points(
    points: torch.Tensor,
    src_size: tuple[int, int],
    dst_size: tuple[int, int],
) -> torch.Tensor:
    """Rescale 2D point coordinates from one image size to another."""
    src_h, src_w = src_size
    dst_h, dst_w = dst_size

    scaled = points.clone().float()
    scaled[..., 0] *= dst_w / src_w
    scaled[..., 1] *= dst_h / src_h
    return scaled


def create_grid(H: int, W: int, gap: int = 1, device: str = 'cpu') -> Tensor:
    """Create an unnormalised (x, y) meshgrid of shape (H//gap, W//gap, 2)."""
    x = torch.linspace(0, W - 1, W // gap)
    y = torch.linspace(0, H - 1, H // gap)
    yg, xg = torch.meshgrid(y, x, indexing="ij")
    return torch.stack((xg, yg), dim=2).to(device)


def softmax_with_temperature(x: Tensor, beta: float, dim: int) -> Tensor:
    """Temperature-scaled softmax."""
    M, _ = x.max(dim=dim, keepdim=True)
    x = x - M
    exp_x = torch.exp(x / beta)
    return exp_x / exp_x.sum(dim=dim, keepdim=True)


def apply_gaussian_kernel(scoremaps: Tensor, sigma: int = 7) -> Tensor:
    """Apply a Gaussian kernel centered at the argmax of each score map."""
    B, N, h, w = scoremaps.shape
    device = scoremaps.device

    idx = torch.max(scoremaps.view(B, N, -1), dim=-1).indices
    idx_y = (idx // w).view(B, N, 1, 1).float()
    idx_x = (idx % w).view(B, N, 1, 1).float()

    grid = create_grid(h, w, device=device).unsqueeze(0).unsqueeze(0)
    grid = grid.expand(B, N, -1, -1, -1)
    gauss = torch.exp(-((grid[..., 0] - idx_x) ** 2 + (grid[..., 1] - idx_y) ** 2) / (2 * sigma ** 2))
    return gauss * scoremaps


def kernel_softargmax_get_matches_logits(trg_logits: Tensor, softmax_temp: float,
                                         sigma: int = 7) -> Tensor:
    """Predict match coordinates via Gaussian-suppressed soft-argmax."""
    B, N, h, w = trg_logits.shape
    device = trg_logits.device

    scoremaps = apply_gaussian_kernel(trg_logits, sigma).view(B, N, -1)
    scoremaps = softmax_with_temperature(scoremaps, softmax_temp, -1)
    grid = create_grid(h, w, device=device)

    scoremaps = scoremaps.unsqueeze(-1).expand(-1, -1, -1, 2)
    grid = grid.view(-1, 2).unsqueeze(0).unsqueeze(0).expand(B, N, -1, -1)
    return (scoremaps * grid).sum(dim=2)
