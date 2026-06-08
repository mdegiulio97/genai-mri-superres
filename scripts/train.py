"""Addestra la rete di super-risoluzione e salva il checkpoint.

Esempio:
    uv run python scripts/train.py --epochs 30
    uv run python scripts/train.py --model srcnn --epochs 40 --scale 2
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from srmri.data import DEFAULT_FLAG, DEFAULT_HR_SIZE, make_loaders  # noqa: E402
from srmri.model import build_model  # noqa: E402
from srmri.train import get_device, train  # noqa: E402

RESULTS = ROOT / "results"


def main() -> None:
    parser = argparse.ArgumentParser(description="Addestra la SR CNN.")
    parser.add_argument("--model", default="espcn", choices=["espcn", "srcnn"])
    parser.add_argument("--flag", default=DEFAULT_FLAG)
    parser.add_argument("--hr-size", type=int, default=DEFAULT_HR_SIZE)
    parser.add_argument("--scale", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--loss", default="l1", choices=["l1", "mse"])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-train", type=int, default=4000,
                        help="limite campioni di training (per velocità su CPU)")
    parser.add_argument("--max-test", type=int, default=1000)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = get_device(args.device)
    print(f"Device: {device}")

    train_loader, val_loader = make_loaders(
        flag=args.flag,
        hr_size=args.hr_size,
        scale=args.scale,
        batch_size=args.batch_size,
        max_train=args.max_train,
        max_test=args.max_test,
        download=True,
    )

    model = build_model(args.model, scale=args.scale, channels=1)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Modello {args.model.upper()} - parametri: {n_params:,}")

    out = train(
        model,
        train_loader,
        epochs=args.epochs,
        lr=args.lr,
        loss=args.loss,
        device=device,
        val_loader=val_loader,
        verbose=True,
    )

    RESULTS.mkdir(parents=True, exist_ok=True)
    ckpt_path = RESULTS / "model.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "model": args.model,
            "scale": args.scale,
            "hr_size": args.hr_size,
            "channels": 1,
            "flag": args.flag,
            "history": out["history"],
        },
        ckpt_path,
    )
    print(f"Checkpoint salvato in {ckpt_path}")
    print(f"Train loss finale: {out['history']['train_loss'][-1]:.5f}")
    if out["history"]["val_loss"]:
        print(f"Val loss finale:   {out['history']['val_loss'][-1]:.5f}")


if __name__ == "__main__":
    main()
