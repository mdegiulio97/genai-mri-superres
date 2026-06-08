"""Metriche di qualità per la super-risoluzione: PSNR e SSIM.

Accettano tensori PyTorch float in [0, 1] (C, H, W) o (N, C, H, W) e si
appoggiano a ``skimage.metrics`` per coerenza con la letteratura. Le immagini
sono convertite in numpy in modo trasparente.
"""

from __future__ import annotations

import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def _to_numpy(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().float().clamp(0.0, 1.0).numpy()


def _iter_images(x: np.ndarray):
    """Itera su immagini (H, W) a partire da array (C,H,W) o (N,C,H,W).

    Per immagini multicanale si converte a singolo canale facendo la media
    sui canali (il task qui è in scala di grigi, quindi C tipicamente = 1).
    """
    if x.ndim == 3:
        x = x[None, ...]  # -> (1, C, H, W)
    if x.ndim != 4:
        raise ValueError(f"Atteso array 3D o 4D, ricevuto {x.shape}")
    for img in x:  # (C, H, W)
        yield img.mean(axis=0)  # (H, W)


def psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    """PSNR medio (dB) tra ``pred`` e ``target`` su range dati in [0, 1]."""
    p = _to_numpy(pred)
    t = _to_numpy(target)
    vals = [
        peak_signal_noise_ratio(ti, pi, data_range=1.0)
        for pi, ti in zip(_iter_images(p), _iter_images(t))
    ]
    return float(np.mean(vals))


def ssim(pred: torch.Tensor, target: torch.Tensor) -> float:
    """SSIM medio tra ``pred`` e ``target`` (immagini in scala di grigi)."""
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
    """PSNR per ogni immagine del batch (utile per istogrammi)."""
    p = _to_numpy(pred)
    t = _to_numpy(target)
    return np.array(
        [
            peak_signal_noise_ratio(ti, pi, data_range=1.0)
            for pi, ti in zip(_iter_images(p), _iter_images(t))
        ]
    )
