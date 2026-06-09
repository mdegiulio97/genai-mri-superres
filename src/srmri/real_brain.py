"""Caricamento di slice 2D da MRI cerebrale REALE (NIfTI) per la super-risoluzione.

A differenza del track sintetico/MedMNIST, qui le immagini ad alta risoluzione
(HR) sono slice di volumi T1w cerebrali reali. La degradazione HR->LR resta
simulata (bicubica), come nella letteratura SR: il valore aggiunto è applicare e
valutare la rete su anatomia cerebrale reale.
"""

from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
import torch


def _crop_even(arr: np.ndarray, scale: int) -> np.ndarray:
    """Ritaglia (centrato) H e W al multiplo di ``scale`` più vicino per difetto."""
    h, w = arr.shape[-2:]
    h2 = h - (h % scale)
    w2 = w - (w % scale)
    top = (h - h2) // 2
    left = (w - w2) // 2
    return arr[..., top : top + h2, left : left + w2]


def _center_crop(arr: np.ndarray, size: int) -> np.ndarray:
    """Ritaglio centrato a (size, size); se la slice è più piccola, la lascia com'è."""
    h, w = arr.shape[-2:]
    ch, cw = min(size, h), min(size, w)
    top = (h - ch) // 2
    left = (w - cw) // 2
    return arr[..., top : top + ch, left : left + cw]


def load_brain_slices(
    nifti_path: Path | str,
    axis: int = 2,
    max_slices: int = 64,
    min_foreground: float = 0.08,
    scale: int = 2,
    pmax: float = 99.0,
    crop_size: int | None = None,
) -> torch.Tensor:
    """Estrae slice 2D normalizzate da un volume T1w cerebrale.

    Args:
        nifti_path:    percorso del NIfTI 3D.
        axis:          asse lungo cui affettare (default 2).
        max_slices:    numero massimo di slice da restituire (campionate uniformemente).
        min_foreground: frazione minima di pixel "tessuto" (>0.1 del max) per tenere
                        una slice; scarta quelle quasi vuote.
        scale:         fattore di super-risoluzione; H,W ritagliate a multipli di scale.
        pmax:          percentile usato per normalizzare l'intensità in [0, 1].

    Returns:
        Tensore float32 (N, 1, H, W) in [0, 1].
    """
    vol = np.asanyarray(nib.load(str(nifti_path)).dataobj, dtype=np.float64)
    pos = vol[vol > 0]
    hi = np.percentile(pos, pmax) if pos.size else float(vol.max())
    hi = max(hi, 1e-6)
    vol = np.clip(vol / hi, 0.0, 1.0)

    vol = np.moveaxis(vol, axis, -1)  # (H, W, S)
    kept = [
        vol[:, :, s]
        for s in range(vol.shape[-1])
        if (vol[:, :, s] > 0.1).mean() >= min_foreground
    ]
    if not kept:
        raise ValueError("Nessuna slice con tessuto sufficiente: controlla axis/min_foreground.")

    if len(kept) > max_slices:
        idx = np.linspace(0, len(kept) - 1, max_slices).round().astype(int)
        kept = [kept[i] for i in idx]

    arr = np.stack(kept, axis=0)  # (N, H, W)
    if crop_size is not None:
        arr = _center_crop(arr, crop_size)
    arr = _crop_even(arr, scale)
    tensor = torch.from_numpy(arr.astype(np.float32)).unsqueeze(1)  # (N, 1, H, W)
    return tensor
