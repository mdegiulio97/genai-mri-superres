"""Super-risoluzione ×2 su MRI cerebrale T1w REALE (OpenNeuro ds007116) — ESPCN vs bicubico.

HR = slice di un volume T1w reale; LR = degradazione bicubica. Si addestra ESPCN
sulle slice di un soggetto e si valuta su un soggetto **held-out** (mai visto):
PSNR e SSIM della SR vs baseline bicubica.

Produce:
- results/brain_metrics.json
- results/brain_examples.png       (LR bicubico vs SR vs HR su slice reali)
- results/brain_psnr_hist.png      (distribuzione PSNR: SR vs bicubico)

Uso:
    uv run python scripts/download_brain.py
    uv run python scripts/evaluate_brain.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from srmri.degrade import downsample, upsample  # noqa: E402
from srmri.metrics import psnr, psnr_per_image, ssim  # noqa: E402
from srmri.model import build_model  # noqa: E402
from srmri.real_brain import load_brain_slices  # noqa: E402
from srmri.train import get_device, train  # noqa: E402

DATA = ROOT / "data" / "brain"
RESULTS = ROOT / "results"
SCALE = 2


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--crop", type=int, default=192)
    ap.add_argument("--max-train", type=int, default=48)
    ap.add_argument("--max-test", type=int, default=32)
    args = ap.parse_args()

    train_path = DATA / "train_T1w.nii.gz"
    test_path = DATA / "test_T1w.nii.gz"
    if not train_path.exists() or not test_path.exists():
        raise SystemExit("Dati assenti. Esegui prima:  uv run python scripts/download_brain.py")

    RESULTS.mkdir(exist_ok=True)
    torch.manual_seed(0)
    device = get_device()
    print(f"Device: {device}")

    hr_train = load_brain_slices(train_path, scale=SCALE, max_slices=args.max_train, crop_size=args.crop)
    hr_test = load_brain_slices(test_path, scale=SCALE, max_slices=args.max_test, crop_size=args.crop)
    print(f"Train HR: {tuple(hr_train.shape)} (soggetto sub-19861)")
    print(f"Test  HR: {tuple(hr_test.shape)} (soggetto held-out sub-138950)")

    lr_train = downsample(hr_train, scale=SCALE)
    lr_test = downsample(hr_test, scale=SCALE)

    loader = DataLoader(TensorDataset(lr_train, hr_train), batch_size=8, shuffle=True)
    model = build_model("espcn", scale=SCALE, channels=1)
    train(model, loader, epochs=args.epochs, lr=1e-3, loss="l1", device=device, verbose=False)

    model.eval()
    with torch.no_grad():
        sr = model(lr_test.to(device)).cpu()
    bicubic = upsample(lr_test, scale=SCALE)

    res = {
        "dataset": {
            "accession": "ds007116",
            "name": "Penn LEAD",
            "license": "CC0",
            "modality": "T1w (brain MRI)",
            "train_subject": "sub-19861",
            "test_subject": "sub-138950 (held-out)",
        },
        "scale": SCALE,
        "n_test_slices": int(hr_test.shape[0]),
        "espcn": {"psnr": psnr(sr, hr_test), "ssim": ssim(sr, hr_test)},
        "bicubic": {"psnr": psnr(bicubic, hr_test), "ssim": ssim(bicubic, hr_test)},
    }
    res["delta"] = {
        "psnr": res["espcn"]["psnr"] - res["bicubic"]["psnr"],
        "ssim": res["espcn"]["ssim"] - res["bicubic"]["ssim"],
    }
    res["sr_beats_bicubic"] = bool(res["delta"]["psnr"] > 0 and res["delta"]["ssim"] > 0)
    (RESULTS / "brain_metrics.json").write_text(json.dumps(res, indent=2))
    print(f"\nPSNR  SR={res['espcn']['psnr']:.2f} dB  bicubico={res['bicubic']['psnr']:.2f} dB  "
          f"(delta {res['delta']['psnr']:+.2f})")
    print(f"SSIM  SR={res['espcn']['ssim']:.4f}  bicubico={res['bicubic']['ssim']:.4f}  "
          f"(delta {res['delta']['ssim']:+.4f})")

    # --- Figura 1: esempi LR-bicubico / SR / HR ---
    n_show = min(3, hr_test.shape[0])
    pick = np.linspace(0, hr_test.shape[0] - 1, n_show).round().astype(int)
    fig, axes = plt.subplots(n_show, 3, figsize=(9, 3 * n_show))
    axes = np.atleast_2d(axes)
    for row, i in enumerate(pick):
        for col, (img, title) in enumerate(
            ((bicubic[i, 0], "Bicubico"), (sr[i, 0], "SR (ESPCN)"), (hr_test[i, 0], "HR reale"))
        ):
            axes[row, col].imshow(img.numpy(), cmap="gray", vmin=0, vmax=1)
            if row == 0:
                axes[row, col].set_title(title)
            axes[row, col].axis("off")
    fig.suptitle("Super-risoluzione ×2 su MRI cerebrale T1w reale (ds007116, CC0)")
    fig.tight_layout()
    fig.savefig(RESULTS / "brain_examples.png", dpi=120)
    plt.close(fig)

    # --- Figura 2: istogramma PSNR ---
    p_sr = psnr_per_image(sr, hr_test)
    p_bi = psnr_per_image(bicubic, hr_test)
    fig2, ax = plt.subplots(figsize=(7, 4))
    bins = np.linspace(min(p_sr.min(), p_bi.min()), max(p_sr.max(), p_bi.max()), 20)
    ax.hist(p_bi, bins=bins, alpha=0.6, label=f"Bicubico (μ={p_bi.mean():.2f} dB)")
    ax.hist(p_sr, bins=bins, alpha=0.6, label=f"SR ESPCN (μ={p_sr.mean():.2f} dB)")
    ax.set_xlabel("PSNR (dB)")
    ax.set_ylabel("n. slice")
    ax.set_title("PSNR per slice — soggetto held-out")
    ax.legend()
    fig2.tight_layout()
    fig2.savefig(RESULTS / "brain_psnr_hist.png", dpi=120)
    plt.close(fig2)
    print("Salvati: brain_metrics.json, brain_examples.png, brain_psnr_hist.png")


if __name__ == "__main__":
    main()
