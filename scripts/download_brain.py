"""Download real T1w brain MRI from OpenNeuro (ds007116, CC0).

Dataset: OpenNeuro ds007116 "Penn LEAD" (CC0 license). The anatomical T1w
images (defaced) of two subjects are downloaded: one for training and one
(held-out) for testing, so evaluation is on a brain never seen in training.
The raw data are NOT committed (the data/ folder is git-ignored).

Run:  uv run python scripts/download_brain.py
"""

from __future__ import annotations

from pathlib import Path

import requests

ACCESSION = "ds007116"
DEST = Path(__file__).resolve().parents[1] / "data" / "brain"
# (relative path within the dataset -> local file name)
SUBJECTS = {
    "train": "sub-19861/ses-1/anat/sub-19861_ses-1_rec-normrefaced_run-01_T1w.nii.gz",
    "test": "sub-138950/ses-1/anat/sub-138950_ses-1_rec-normrefaced_run-01_T1w.nii.gz",
}
BASE = f"https://s3.amazonaws.com/openneuro.org/{ACCESSION}"


def main() -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    print(f"Dataset: OpenNeuro {ACCESSION} (Penn LEAD) — CC0 license")
    print("Brain T1w, train and test on different subjects.\n")
    for split, rel in SUBJECTS.items():
        dest = DEST / f"{split}_T1w.nii.gz"
        if dest.exists() and dest.stat().st_size > 0:
            print(f"  [skip] {split}: {dest.name}")
            continue
        print(f"  [get ] {split}: {rel}")
        resp = requests.get(f"{BASE}/{rel}", timeout=180)
        resp.raise_for_status()
        dest.write_bytes(resp.content)
    print(f"\nDownloaded to: {DEST}")
    print("Citation: OpenNeuro dataset ds007116 (Penn LEAD), CC0.")


if __name__ == "__main__":
    main()
