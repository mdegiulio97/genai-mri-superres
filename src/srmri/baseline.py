"""Super-resolution baseline: bicubic interpolation.

This is the "non-learned" reference against which the neural network is compared.
"""

from __future__ import annotations

import torch

from .degrade import upsample


def bicubic_sr(lr: torch.Tensor, scale: int = 2) -> torch.Tensor:
    """Bicubic super-resolution: upscale LR -> HR by a factor of ``scale``."""
    return upsample(lr, scale=scale)
