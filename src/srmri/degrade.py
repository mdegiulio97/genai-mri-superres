"""Bicubic degradation and resampling of images.

Tensor convention: float32 in [0, 1], shape (N, C, H, W) or (C, H, W).
The HR -> LR downsample and the LR -> HR upsample use bicubic interpolation
(``torch.nn.functional.interpolate`` with ``mode="bicubic"``), the classic
baseline in super-resolution work (SRCNN/ESPCN).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _ensure_batch(x: torch.Tensor) -> tuple[torch.Tensor, bool]:
    """Bring a tensor to 4D (N, C, H, W); also return whether it was 3D."""
    if x.dim() == 3:
        return x.unsqueeze(0), True
    if x.dim() == 4:
        return x, False
    raise ValueError(f"Expected a 3D or 4D tensor, got shape {tuple(x.shape)}")


def downsample(hr: torch.Tensor, scale: int = 2) -> torch.Tensor:
    """Degrade HR -> LR by reducing the spatial dimensions by ``scale`` (bicubic).

    Args:
        hr: tensor (C, H, W) or (N, C, H, W) in [0, 1].
        scale: downscaling factor (e.g. 2 -> halves H and W).

    Returns:
        LR tensor with the same N, C but H/scale, W/scale.
    """
    if scale < 1:
        raise ValueError("scale must be >= 1")
    x, squeezed = _ensure_batch(hr)
    _, _, h, w = x.shape
    if h % scale != 0 or w % scale != 0:
        raise ValueError(f"H={h}, W={w} not divisible by scale={scale}")
    lr = F.interpolate(
        x, size=(h // scale, w // scale), mode="bicubic", align_corners=False
    )
    lr = lr.clamp(0.0, 1.0)
    return lr.squeeze(0) if squeezed else lr


def upsample(lr: torch.Tensor, scale: int = 2) -> torch.Tensor:
    """Upscale LR -> HR by increasing the spatial dimensions by ``scale`` (bicubic).

    This is the bicubic baseline, used both for comparison and as the
    pre-upscaling step for SRCNN.
    """
    if scale < 1:
        raise ValueError("scale must be >= 1")
    x, squeezed = _ensure_batch(lr)
    _, _, h, w = x.shape
    hr = F.interpolate(
        x, size=(h * scale, w * scale), mode="bicubic", align_corners=False
    )
    hr = hr.clamp(0.0, 1.0)
    return hr.squeeze(0) if squeezed else hr


def make_lr_hr_pair(hr: torch.Tensor, scale: int = 2) -> tuple[torch.Tensor, torch.Tensor]:
    """Produce the (LR, HR) training pair from an HR image."""
    lr = downsample(hr, scale=scale)
    return lr, hr
