"""Scarica MRI cerebrale T1w reale da OpenNeuro (ds007116, CC0).

Dataset: OpenNeuro ds007116 "Penn LEAD" (licenza CC0). Si scaricano i T1w
anatomici (defaced) di due soggetti: uno per il training, uno (held-out) per il
test, così la valutazione è su un cervello mai visto. I dati grezzi NON sono
committati (cartella data/ gitignorata).

Esegui:  uv run python scripts/download_brain.py
"""

from __future__ import annotations

from pathlib import Path

import requests

ACCESSION = "ds007116"
DEST = Path(__file__).resolve().parents[1] / "data" / "brain"
# (sottocartella relativa nel dataset -> nome file locale)
SUBJECTS = {
    "train": "sub-19861/ses-1/anat/sub-19861_ses-1_rec-normrefaced_run-01_T1w.nii.gz",
    "test": "sub-138950/ses-1/anat/sub-138950_ses-1_rec-normrefaced_run-01_T1w.nii.gz",
}
BASE = f"https://s3.amazonaws.com/openneuro.org/{ACCESSION}"


def main() -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    print(f"Dataset: OpenNeuro {ACCESSION} (Penn LEAD) — licenza CC0")
    print("T1w cerebrale, train e test su soggetti distinti.\n")
    for split, rel in SUBJECTS.items():
        dest = DEST / f"{split}_T1w.nii.gz"
        if dest.exists() and dest.stat().st_size > 0:
            print(f"  [skip] {split}: {dest.name}")
            continue
        print(f"  [get ] {split}: {rel}")
        resp = requests.get(f"{BASE}/{rel}", timeout=180)
        resp.raise_for_status()
        dest.write_bytes(resp.content)
    print(f"\nScaricati in: {DEST}")
    print("Citazione: dataset OpenNeuro ds007116 (Penn LEAD), CC0.")


if __name__ == "__main__":
    main()
