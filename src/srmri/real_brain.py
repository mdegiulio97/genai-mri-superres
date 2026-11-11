"""Loading of 2D slices from REAL brain MRI (NIfTI) for super-resolution.

Unlike the synthetic/MedMNIST track, here the high-resolution (HR) images are
slices of real T1w brain volumes. The HR->LR degradation is still simulated
(bicubic), as in the SR literature: the added value is applying and evaluating
the network on real brain anatomy.
"""

from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
import torch


def _crop_even(arr: np.ndarray, scale: int) -> np.ndarray:
    """Center-crop H and W down to the nearest multiple of ``scale``."""
    h, w = arr.shape[-2:]
    h2 = h - (h % scale)
    w2 = w - (w % scale)
    top = (h - h2) // 2
    left = (w - w2) // 2
    return arr[..., top : top + h2, left : left + w2]


def _center_crop(arr: np.ndarray, size: int) -> np.ndarray:
    """Center crop to (size, size); if the slice is smaller, leave it as is."""
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
    """Extract normalized 2D slices from a T1w brain volume.

    Args:
        nifti_path:    path to the 3D NIfTI file.
        axis:          axis along which to slice (default 2).
        max_slices:    maximum number of slices to return (uniformly sampled).
        min_foreground: minimum fraction of "tissue" pixels (>0.1 of max) needed
                        to keep a slice; nearly empty slices are discarded.
        scale:         super-resolution factor; H,W are cropped to multiples of scale.
        pmax:          percentile used to normalize intensity to [0, 1].
        crop_size:     optional center-crop size applied before the even crop.

    Returns:
        float32 tensor (N, 1, H, W) in [0, 1].
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
        raise ValueError("No slice with enough tissue: check axis/min_foreground.")

    if len(kept) > max_slices:
        idx = np.linspace(0, len(kept) - 1, max_slices).round().astype(int)
        kept = [kept[i] for i in idx]

    arr = np.stack(kept, axis=0)  # (N, H, W)
    if crop_size is not None:
        arr = _center_crop(arr, crop_size)
    arr = _crop_even(arr, scale)
    tensor = torch.from_numpy(arr.astype(np.float32)).unsqueeze(1)  # (N, 1, H, W)
    return tensor
