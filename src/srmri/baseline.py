"""Baseline di super-risoluzione: interpolazione bicubica.

È il riferimento "non-learned" contro cui si confronta la rete neurale.
"""

from __future__ import annotations

import torch

from .degrade import upsample


def bicubic_sr(lr: torch.Tensor, scale: int = 2) -> torch.Tensor:
    """Super-risoluzione bicubica: upscala LR -> HR del fattore ``scale``."""
    return upsample(lr, scale=scale)
