"""Compact generative networks for image-to-image super-resolution.

- :class:`ESPCN` (default): Shi et al., CVPR 2016 (arXiv:1609.05158). Features
  are extracted at LR resolution and upscaling happens ONLY at the end through
  sub-pixel convolution (``PixelShuffle``), which makes it efficient.
  The input is the LR image (H, W); the output is (H*scale, W*scale).

- :class:`SRCNN`: Dong et al., TPAMI 2016 (DOI 10.1109/TPAMI.2015.2439281).
  Works on an image ALREADY upscaled bicubically (same resolution as the
  output) and refines its high-frequency details.
"""

from __future__ import annotations

import torch
from torch import nn

from .degrade import upsample


class ESPCN(nn.Module):
    """Efficient Sub-Pixel CNN for ``scale`` upscaling.

    Variant with **residual learning** with respect to the bicubic baseline:
    features are extracted at LR resolution and the sub-pixel convolution
    (``PixelShuffle``) produces a high-frequency residual that is **added** to
    the bicubic upscaling. This is a well-established scheme in SR work
    (e.g. VDSR, Kim et al. CVPR 2016) and makes training more stable: the
    network effectively starts from bicubic quality and learns only the
    corrections, so it cannot degrade the baseline.

    The last layer is initialized to zero so that, at the start, the output
    equals the bicubic interpolation.

    ``LeakyReLU`` (instead of ReLU) is used in the feature extractor: without
    normalization and with a zero-initialized last layer, plain ReLUs tend to
    "die" (zero activations, zero gradient, residual stuck at zero -> the
    network matches but does not beat bicubic). The non-zero negative slope
    keeps the gradient alive and makes training robust to the learning rate.

    Args:
        scale: upscaling factor (2 or 4).
        channels: number of image channels (1 for grayscale).
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
        # Project to channels * scale^2 channels, then rearrange with PixelShuffle.
        self.sub_pixel_conv = nn.Conv2d(
            32, channels * scale * scale, kernel_size=3, padding=1
        )
        self.shuffle = nn.PixelShuffle(scale)
        # Zero initialization: zero residual at start -> output = bicubic.
        nn.init.zeros_(self.sub_pixel_conv.weight)
        nn.init.zeros_(self.sub_pixel_conv.bias)

    def forward(self, lr: torch.Tensor) -> torch.Tensor:
        base = upsample(lr, scale=self.scale)  # bicubic baseline (HR)
        x = self.features(lr)
        residual = self.shuffle(self.sub_pixel_conv(x))
        return (base + residual).clamp(0.0, 1.0)


class SRCNN(nn.Module):
    """SRCNN: refines an image pre-upscaled with bicubic interpolation.

    The forward pass takes the LR image and performs the bicubic upscaling
    internally, so the interface (LR -> HR) is the same as ESPCN.
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
        # Residual connection: the network predicts the detail on top of bicubic.
        out = up + self.body(up)
        return out.clamp(0.0, 1.0)


def build_model(name: str = "espcn", scale: int = 2, channels: int = 1) -> nn.Module:
    """Factory: build the model from its name ('espcn' or 'srcnn')."""
    name = name.lower()
    if name == "espcn":
        return ESPCN(scale=scale, channels=channels)
    if name == "srcnn":
        return SRCNN(scale=scale, channels=channels)
    raise ValueError(f"Unknown model: {name!r}. Use 'espcn' or 'srcnn'.")
