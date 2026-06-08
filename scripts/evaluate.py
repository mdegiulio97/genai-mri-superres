"""Valuta SR vs bicubica sul test set: metrics.json + figure.

Esempio:
    uv run python scripts/evaluate.py

Produce:
    results/metrics.json          (PSNR/SSIM di SR e bicubica + delta)
    results/examples_before_after.png
    results/psnr_histogram.png
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from srmri.baseline import bicubic_sr  # noqa: E402
from srmri.data import make_loaders  # noqa: E402
from srmri.metrics import psnr, psnr_per_image, ssim  # noqa: E402
from srmri.model import build_model  # noqa: E402
from srmri.train import get_device  # noqa: E402

RESULTS = ROOT / "results"


def _load_checkpoint(device: torch.device):
    ckpt_path = RESULTS / "model.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Checkpoint mancante: {ckpt_path}. Esegui prima scripts/train.py."
        )
    ckpt = torch.load(ckpt_path, map_location=device)
    model = build_model(ckpt["model"], scale=ckpt["scale"], channels=ckpt["channels"])
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    return model, ckpt


@torch.no_grad()
def _collect(model, loader, scale, device):
    """Ritorna (sr, bic, hr) concatenati sul test set."""
    srs, bics, hrs = [], [], []
    for lr, hr in loader:
        lr = lr.to(device)
        srs.append(model(lr).cpu())
        bics.append(bicubic_sr(lr, scale=scale).cpu())
        hrs.append(hr)
    return torch.cat(srs), torch.cat(bics), torch.cat(hrs)


def _fig_examples(sr, bic, hr, path: Path, n: int = 5) -> None:
    n = min(n, sr.size(0))
    fig, axes = plt.subplots(4, n, figsize=(2.2 * n, 9))
    err = (sr - hr).abs()  # mappa di errore assoluto SR
    for j in range(n):
        axes[0, j].imshow(bic[j, 0], cmap="gray", vmin=0, vmax=1)
        axes[1, j].imshow(sr[j, 0], cmap="gray", vmin=0, vmax=1)
        axes[2, j].imshow(hr[j, 0], cmap="gray", vmin=0, vmax=1)
        axes[3, j].imshow(err[j, 0], cmap="magma", vmin=0, vmax=0.3)
        for r in range(4):
            axes[r, j].set_xticks([])
            axes[r, j].set_yticks([])
    labels = ["Bicubica", "SR (rete)", "HR (target)", "|SR - HR|"]
    for r, lab in enumerate(labels):
        axes[r, 0].set_ylabel(lab, fontsize=11)
    fig.suptitle("Super-risoluzione: confronto before/after", fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _fig_psnr_hist(sr_psnr: np.ndarray, bic_psnr: np.ndarray, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bins = np.linspace(
        min(sr_psnr.min(), bic_psnr.min()) - 1,
        max(sr_psnr.max(), bic_psnr.max()) + 1,
        40,
    )
    ax.hist(bic_psnr, bins=bins, alpha=0.6, label=f"Bicubica (μ={bic_psnr.mean():.2f} dB)")
    ax.hist(sr_psnr, bins=bins, alpha=0.6, label=f"SR rete (μ={sr_psnr.mean():.2f} dB)")
    ax.axvline(bic_psnr.mean(), color="C0", ls="--")
    ax.axvline(sr_psnr.mean(), color="C1", ls="--")
    ax.set_xlabel("PSNR per immagine (dB)")
    ax.set_ylabel("Conteggio")
    ax.set_title("Distribuzione del PSNR sul test set: SR vs bicubica")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main() -> None:
    device = get_device()
    print(f"Device: {device}")
    model, ckpt = _load_checkpoint(device)
    scale = ckpt["scale"]

    _, test_loader = make_loaders(
        flag=ckpt["flag"],
        hr_size=ckpt["hr_size"],
        scale=scale,
        batch_size=128,
        max_test=1000,
        download=True,
    )

    sr, bic, hr = _collect(model, test_loader, scale, device)

    metrics = {
        "dataset": ckpt["flag"],
        "model": ckpt["model"],
        "hr_size": ckpt["hr_size"],
        "scale": scale,
        "n_test": int(sr.size(0)),
        "sr": {"psnr": psnr(sr, hr), "ssim": ssim(sr, hr)},
        "bicubic": {"psnr": psnr(bic, hr), "ssim": ssim(bic, hr)},
    }
    metrics["delta"] = {
        "psnr": metrics["sr"]["psnr"] - metrics["bicubic"]["psnr"],
        "ssim": metrics["sr"]["ssim"] - metrics["bicubic"]["ssim"],
    }
    metrics["sr_beats_bicubic"] = bool(
        metrics["delta"]["psnr"] > 0 and metrics["delta"]["ssim"] > 0
    )

    RESULTS.mkdir(parents=True, exist_ok=True)
    with open(RESULTS / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    sr_psnr = psnr_per_image(sr, hr)
    bic_psnr = psnr_per_image(bic, hr)
    _fig_examples(sr, bic, hr, RESULTS / "examples_before_after.png")
    _fig_psnr_hist(sr_psnr, bic_psnr, RESULTS / "psnr_histogram.png")

    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    verdict = "SI" if metrics["sr_beats_bicubic"] else "NO"
    print(f"\nLa rete SR batte la bicubica (PSNR e SSIM)? {verdict}")
    print(f"Artefatti salvati in {RESULTS}")


if __name__ == "__main__":
    main()
