"""Test offline del loader di slice cerebrali reali (nessun download).

Si scrive un piccolo NIfTI 3D sintetico (un blob luminoso nelle slice centrali,
bordi vuoti) e si verifica che il loader: normalizzi in [0, 1], scarti le slice
quasi vuote, ritagli a dimensioni pari (divisibili per lo scale) e rispetti il
numero massimo di slice richiesto.
"""

from __future__ import annotations

import nibabel as nib
import numpy as np

from srmri.real_brain import load_brain_slices


def _write_synthetic_brain(tmp_path, shape=(31, 33, 12)):
    vol = np.zeros(shape, dtype=np.float32)
    # Slice centrali con un "cervello" luminoso; le prime/ultime restano vuote.
    cx, cy = shape[0] // 2, shape[1] // 2
    yy, xx = np.mgrid[0 : shape[0], 0 : shape[1]]
    disk = ((xx - cx) ** 2 + (yy - cy) ** 2) < (min(shape[0], shape[1]) // 3) ** 2
    for z in range(3, shape[2] - 3):
        vol[:, :, z] = disk * (500.0 + 50.0 * z)
    path = tmp_path / "sub-x_T1w.nii.gz"
    nib.save(nib.Nifti1Image(vol, np.eye(4)), path)
    return path


def test_load_brain_slices_basic(tmp_path):
    path = _write_synthetic_brain(tmp_path)
    slices = load_brain_slices(path, max_slices=8, scale=2)
    n, c, h, w = slices.shape
    assert c == 1
    assert n <= 8
    assert n > 0
    assert h % 2 == 0 and w % 2 == 0  # ritaglio a dimensioni pari (scale=2)
    assert float(slices.min()) >= 0.0 and float(slices.max()) <= 1.0


def test_load_brain_slices_skips_empty(tmp_path):
    path = _write_synthetic_brain(tmp_path, shape=(20, 20, 12))
    # Solo 6 slice (z=3..8) hanno contenuto: con min_foreground tipico ne restano <= 6.
    slices = load_brain_slices(path, max_slices=64, scale=2, min_foreground=0.05)
    assert 0 < slices.shape[0] <= 6
