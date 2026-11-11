"""Download the MedMNIST dataset (idempotent).

Example:
    uv run python scripts/download_data.py
    uv run python scripts/download_data.py --flag bloodmnist --size 64
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from srmri.data import DEFAULT_FLAG, DEFAULT_HR_SIZE, load_hr_split  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Download MedMNIST (idempotent).")
    parser.add_argument("--flag", default=DEFAULT_FLAG, help="MedMNIST dataset")
    parser.add_argument("--size", type=int, default=DEFAULT_HR_SIZE, help="HR resolution")
    args = parser.parse_args()

    print(f"Downloading/checking MedMNIST flag={args.flag} size={args.size} ...")
    for split in ("train", "val", "test"):
        t = load_hr_split(split, flag=args.flag, hr_size=args.size, download=True)
        print(f"  split={split:5s} -> HR tensor {tuple(t.shape)} in [{t.min():.2f},{t.max():.2f}]")
    print("Done. The .npz files are in", ROOT / "data")


if __name__ == "__main__":
    main()
