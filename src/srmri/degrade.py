"""Degradazione e ricampionamento bicubico delle immagini.

Convenzione tensori: float32 in [0, 1], shape (N, C, H, W) oppure (C, H, W).
Il downsample HR -> LR e l'upsample LR -> HR usano interpolazione bicubica
(``torch.nn.functional.interpolate`` con ``mode="bicubic"``), che è la baseline
classica nei lavori di super-risoluzione (SRCNN/ESPCN).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _ensure_batch(x: torch.Tensor) -> tuple[torch.Tensor, bool]:
    """Porta un tensore a 4D (N, C, H, W); ritorna anche se era 3D."""
    if x.dim() == 3:
        return x.unsqueeze(0), True
    if x.dim() == 4:
        return x, False
    raise ValueError(f"Atteso tensore 3D o 4D, ricevuto shape {tuple(x.shape)}")


def downsample(hr: torch.Tensor, scale: int = 2) -> torch.Tensor:
    """Degrada HR -> LR riducendo le dimensioni spaziali di ``scale`` (bicubico).

    Args:
        hr: tensore (C, H, W) o (N, C, H, W) in [0, 1].
        scale: fattore di downscaling (es. 2 -> dimezza H e W).

    Returns:
        Tensore LR con stesse N, C ma H/scale, W/scale.
    """
    if scale < 1:
        raise ValueError("scale deve essere >= 1")
    x, squeezed = _ensure_batch(hr)
    _, _, h, w = x.shape
    if h % scale != 0 or w % scale != 0:
        raise ValueError(f"H={h}, W={w} non divisibili per scale={scale}")
    lr = F.interpolate(
        x, size=(h // scale, w // scale), mode="bicubic", align_corners=False
    )
    lr = lr.clamp(0.0, 1.0)
    return lr.squeeze(0) if squeezed else lr


def upsample(lr: torch.Tensor, scale: int = 2) -> torch.Tensor:
    """Upscala LR -> HR aumentando le dimensioni spaziali di ``scale`` (bicubico).

    Questa è la baseline bicubica usata sia come confronto sia come pre-upscaling
    per SRCNN.
    """
    if scale < 1:
        raise ValueError("scale deve essere >= 1")
    x, squeezed = _ensure_batch(lr)
    _, _, h, w = x.shape
    hr = F.interpolate(
        x, size=(h * scale, w * scale), mode="bicubic", align_corners=False
    )
    hr = hr.clamp(0.0, 1.0)
    return hr.squeeze(0) if squeezed else hr


def make_lr_hr_pair(hr: torch.Tensor, scale: int = 2) -> tuple[torch.Tensor, torch.Tensor]:
    """Da un'immagine HR produce la coppia (LR, HR) per il training."""
    lr = downsample(hr, scale=scale)
    return lr, hr
