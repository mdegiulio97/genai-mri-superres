"""srmri — image-to-image super-resolution of 2D medical images.

Compact convolutional generative network (ESPCN/SRCNN) trained on MedMNIST v2,
compared with the bicubic baseline through PSNR/SSIM.
"""

from __future__ import annotations

from . import baseline, data, degrade, metrics, model, train

__all__ = ["baseline", "data", "degrade", "metrics", "model", "train"]
__version__ = "0.1.0"
