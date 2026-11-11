"""Data loading: HR images from MedMNIST + HR->LR degradation + synthetic loader.

- MedMNIST images are loaded in **grayscale** at the chosen HR resolution
  (default 64x64, ``size=64`` in the ``medmnist`` library).
- The HR -> LR degradation is a bicubic downsample (see ``degrade.py``).
- Offline tests use :func:`fake_hr_dataset`, which generates synthetic
  patterns without downloading anything.

Dataset reference: MedMNIST v2, Yang et al., Scientific Data 2023,
DOI 10.1038/s41597-022-01721-8 (CC BY 4.0 license).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset

from .degrade import downsample

DEFAULT_FLAG = "bloodmnist"
DEFAULT_HR_SIZE = 64
DEFAULT_SCALE = 2


def _data_root() -> Path:
    root = Path(__file__).resolve().parents[2] / "data"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _to_gray_hr_tensor(images: np.ndarray) -> torch.Tensor:
    """Convert a MedMNIST array (N,H,W,C) or (N,H,W) to grayscale (N,1,H,W) float[0,1]."""
    arr = images.astype(np.float32) / 255.0
    if arr.ndim == 3:  # (N, H, W) -> add channel
        arr = arr[..., None]
    if arr.ndim != 4:
        raise ValueError(f"Unexpected image shape: {arr.shape}")
    gray = arr.mean(axis=-1, keepdims=True)  # mean over channels -> (N,H,W,1)
    t = torch.from_numpy(gray).permute(0, 3, 1, 2).contiguous()  # (N,1,H,W)
    return t.clamp(0.0, 1.0)


def load_hr_split(
    split: str,
    flag: str = DEFAULT_FLAG,
    hr_size: int = DEFAULT_HR_SIZE,
    download: bool = True,
) -> torch.Tensor:
    """Load a MedMNIST split as an HR tensor (N,1,hr_size,hr_size) in [0,1].

    ``medmnist`` is imported lazily so that offline tests do not need it.
    """
    import medmnist
    from medmnist import INFO

    if flag not in INFO:
        raise ValueError(f"Unknown MedMNIST flag: {flag!r}")
    cls = getattr(medmnist, INFO[flag]["python_class"])

    # medmnist>=3.0 supports the size parameter (28/64/128/224). Fall back to 28
    # if the installed version does not support it (effective hr_size = 28).
    kwargs = dict(split=split, download=download, root=str(_data_root()))
    try:
        ds = cls(size=hr_size, **kwargs)
    except TypeError:
        ds = cls(**kwargs)

    images = ds.imgs  # numpy array
    return _to_gray_hr_tensor(images)


def make_loaders(
    flag: str = DEFAULT_FLAG,
    hr_size: int = DEFAULT_HR_SIZE,
    scale: int = DEFAULT_SCALE,
    batch_size: int = 64,
    max_train: int | None = None,
    max_test: int | None = None,
    download: bool = True,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader]:
    """Build the (train, test) DataLoaders of (LR, HR) pairs.

    ``max_train`` / ``max_test`` cap the number of samples (useful for quick
    runs on CPU). Tensors are (LR: N,1,h,w) and (HR: N,1,H,W).
    """
    hr_train = load_hr_split("train", flag=flag, hr_size=hr_size, download=download)
    hr_test = load_hr_split("test", flag=flag, hr_size=hr_size, download=download)

    if max_train is not None:
        hr_train = hr_train[:max_train]
    if max_test is not None:
        hr_test = hr_test[:max_test]

    lr_train = downsample(hr_train, scale=scale)
    lr_test = downsample(hr_test, scale=scale)

    train_ds = TensorDataset(lr_train, hr_train)
    test_ds = TensorDataset(lr_test, hr_test)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    return train_loader, test_loader


def fake_hr_dataset(
    n: int = 16, hr_size: int = 32, channels: int = 1, seed: int = 0
) -> torch.Tensor:
    """Synthetic HR dataset (no download) for offline tests.

    Generates structured patterns (gradients + sine waves + mild noise) so that
    super-resolution has non-trivial high-frequency content.
    """
    g = torch.Generator().manual_seed(seed)
    yy, xx = torch.meshgrid(
        torch.linspace(0, 1, hr_size),
        torch.linspace(0, 1, hr_size),
        indexing="ij",
    )
    imgs = []
    for i in range(n):
        freq = 3.0 + (i % 5)
        phase = 0.3 * i
        base = 0.5 + 0.25 * torch.sin(2 * torch.pi * freq * xx + phase)
        base = base + 0.2 * yy
        noise = 0.03 * torch.randn(hr_size, hr_size, generator=g)
        img = (base + noise).clamp(0.0, 1.0)
        imgs.append(img)
    stacked = torch.stack(imgs).unsqueeze(1)  # (N,1,H,W)
    if channels > 1:
        stacked = stacked.repeat(1, channels, 1, 1)
    return stacked


def fake_loaders(
    n_train: int = 16,
    n_test: int = 8,
    hr_size: int = 32,
    scale: int = 2,
    batch_size: int = 8,
    channels: int = 1,
) -> tuple[DataLoader, DataLoader]:
    """Synthetic (LR, HR) loaders for offline tests."""
    hr_train = fake_hr_dataset(n_train, hr_size=hr_size, channels=channels, seed=0)
    hr_test = fake_hr_dataset(n_test, hr_size=hr_size, channels=channels, seed=1)
    lr_train = downsample(hr_train, scale=scale)
    lr_test = downsample(hr_test, scale=scale)
    train_loader = DataLoader(
        TensorDataset(lr_train, hr_train), batch_size=batch_size, shuffle=True
    )
    test_loader = DataLoader(
        TensorDataset(lr_test, hr_test), batch_size=batch_size, shuffle=False
    )
    return train_loader, test_loader


class _Identity(Dataset):  # pragma: no cover - minimal utility
    def __init__(self, x: torch.Tensor) -> None:
        self.x = x

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, i: int) -> torch.Tensor:
        return self.x[i]
