"""Quality metrics for super-resolution: PSNR and SSIM.

They accept PyTorch float tensors in [0, 1] (C, H, W) or (N, C, H, W) and rely
on ``skimage.metrics`` for consistency with the literature. Images are
converted to numpy transparently.
"""

from __future__ import annotations

import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def _to_numpy(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().float().clamp(0.0, 1.0).numpy()


def _iter_images(x: np.ndarray):
    """Iterate over (H, W) images from a (C,H,W) or (N,C,H,W) array.

    Multi-channel images are converted to a single channel by averaging over
    channels (the task here is grayscale, so C is typically 1).
    """
    if x.ndim == 3:
        x = x[None, ...]  # -> (1, C, H, W)
    if x.ndim != 4:
        raise ValueError(f"Expected a 3D or 4D array, got {x.shape}")
    for img in x:  # (C, H, W)
        yield img.mean(axis=0)  # (H, W)


def psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Mean PSNR (dB) between ``pred`` and ``target`` for data range [0, 1]."""
    p = _to_numpy(pred)
    t = _to_numpy(target)
    vals = [
        peak_signal_noise_ratio(ti, pi, data_range=1.0)
        for pi, ti in zip(_iter_images(p), _iter_images(t))
    ]
    return float(np.mean(vals))


def ssim(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Mean SSIM between ``pred`` and ``target`` (grayscale images)."""
    p = _to_numpy(pred)
    t = _to_numpy(target)
    vals = []
    for pi, ti in zip(_iter_images(p), _iter_images(t)):
        win = min(7, pi.shape[0], pi.shape[1])
        if win % 2 == 0:
            win -= 1
        vals.append(
            structural_similarity(ti, pi, data_range=1.0, win_size=max(win, 3))
        )
    return float(np.mean(vals))


def psnr_per_image(pred: torch.Tensor, target: torch.Tensor) -> np.ndarray:
    """PSNR for each image in the batch (useful for histograms)."""
    p = _to_numpy(pred)
    t = _to_numpy(target)
    return np.array(
        [
            peak_signal_noise_ratio(ti, pi, data_range=1.0)
            for pi, ti in zip(_iter_images(p), _iter_images(t))
        ]
    )
