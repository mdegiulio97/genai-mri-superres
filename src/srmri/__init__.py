"""srmri — super-risoluzione image-to-image di immagini mediche 2D.

Rete generativa convoluzionale compatta (ESPCN/SRCNN) addestrata su MedMNIST v2,
confrontata con la baseline bicubica tramite PSNR/SSIM.
"""

from __future__ import annotations

from . import baseline, data, degrade, metrics, model, train

__all__ = ["baseline", "data", "degrade", "metrics", "model", "train"]
__version__ = "0.1.0"
