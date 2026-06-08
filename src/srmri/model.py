"""Reti generative compatte per super-risoluzione image-to-image.

- :class:`ESPCN` (default): Shi et al., CVPR 2016 (arXiv:1609.05158). Le feature
  sono estratte a risoluzione LR e l'upscaling avviene SOLO alla fine tramite
  sub-pixel convolution (``PixelShuffle``), il che la rende efficiente.
  L'input è l'immagine LR (H, W); l'output è (H*scale, W*scale).

- :class:`SRCNN`: Dong et al., TPAMI 2016 (DOI 10.1109/TPAMI.2015.2439281).
  Lavora su un'immagine GIÀ upscalata bicubicamente (stessa risoluzione
  dell'output) e ne raffina i dettagli ad alta frequenza.
"""

from __future__ import annotations

import torch
from torch import nn

from .degrade import upsample


class ESPCN(nn.Module):
    """Efficient Sub-Pixel CNN per upscaling ``scale``.

    Variante con **apprendimento residuo** rispetto alla baseline bicubica: le
    feature sono estratte a risoluzione LR e la sub-pixel convolution
    (``PixelShuffle``) genera un residuo ad alta frequenza che viene **sommato**
    all'upscaling bicubico. Questo è uno schema ben consolidato nei lavori di SR
    (es. VDSR, Kim et al. CVPR 2016) e rende il training più stabile: la rete
    parte di fatto dalla qualità bicubica e impara solo le correzioni, così non
    può degradare la baseline.

    L'ultimo strato è inizializzato a zero affinché all'inizio l'output coincida
    con l'interpolazione bicubica.

    Si usa ``LeakyReLU`` (anziché ReLU) nel feature extractor: senza
    normalizzazione e con l'ultimo strato a zero, le ReLU classiche tendono a
    "morire" (attivazioni a zero, gradiente nullo, residuo costantemente nullo →
    la rete eguaglia ma non supera la bicubica). La pendenza negativa non nulla
    mantiene vivo il gradiente e rende il training robusto al learning rate.

    Args:
        scale: fattore di upscaling (2 o 4).
        channels: numero di canali immagine (1 per scala di grigi).
    """

    def __init__(self, scale: int = 2, channels: int = 1) -> None:
        super().__init__()
        self.scale = scale
        self.channels = channels
        self.features = nn.Sequential(
            nn.Conv2d(channels, 64, kernel_size=5, padding=2),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
        )
        # Proietta su channels * scale^2 canali, poi riorganizza con PixelShuffle.
        self.sub_pixel_conv = nn.Conv2d(
            32, channels * scale * scale, kernel_size=3, padding=1
        )
        self.shuffle = nn.PixelShuffle(scale)
        # Inizializzazione a zero: residuo nullo all'avvio -> output = bicubica.
        nn.init.zeros_(self.sub_pixel_conv.weight)
        nn.init.zeros_(self.sub_pixel_conv.bias)

    def forward(self, lr: torch.Tensor) -> torch.Tensor:
        base = upsample(lr, scale=self.scale)  # baseline bicubica (HR)
        x = self.features(lr)
        residual = self.shuffle(self.sub_pixel_conv(x))
        return (base + residual).clamp(0.0, 1.0)


class SRCNN(nn.Module):
    """SRCNN: raffina un'immagine pre-upscalata bicubicamente.

    Il forward accetta l'immagine LR e fa internamente l'upscaling bicubico,
    così l'interfaccia (LR -> HR) è identica a ESPCN.
    """

    def __init__(self, scale: int = 2, channels: int = 1) -> None:
        super().__init__()
        self.scale = scale
        self.channels = channels
        self.body = nn.Sequential(
            nn.Conv2d(channels, 64, kernel_size=9, padding=4),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(64, 32, kernel_size=5, padding=2),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(32, channels, kernel_size=5, padding=2),
        )

    def forward(self, lr: torch.Tensor) -> torch.Tensor:
        up = upsample(lr, scale=self.scale)
        # Connessione residua: la rete predice il dettaglio sopra la bicubica.
        out = up + self.body(up)
        return out.clamp(0.0, 1.0)


def build_model(name: str = "espcn", scale: int = 2, channels: int = 1) -> nn.Module:
    """Factory: costruisce il modello dal nome ('espcn' o 'srcnn')."""
    name = name.lower()
    if name == "espcn":
        return ESPCN(scale=scale, channels=channels)
    if name == "srcnn":
        return SRCNN(scale=scale, channels=channels)
    raise ValueError(f"Modello sconosciuto: {name!r}. Usa 'espcn' o 'srcnn'.")
