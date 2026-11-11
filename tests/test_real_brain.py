"""Offline test of the real brain slice loader (no download).

A small synthetic 3D NIfTI is written (a bright blob in the central slices,
empty borders) and the test checks that the loader normalizes to [0, 1],
discards nearly empty slices, crops to even sizes (divisible by the scale) and
respects the requested maximum number of slices.
"""

from __future__ import annotations

import nibabel as nib
import numpy as np

from srmri.real_brain import load_brain_slices


def _write_synthetic_brain(tmp_path, shape=(31, 33, 12)):
    vol = np.zeros(shape, dtype=np.float32)
    # Central slices with a bright "brain"; the first/last ones stay empty.
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
    assert h % 2 == 0 and w % 2 == 0  # cropped to even sizes (scale=2)
    assert float(slices.min()) >= 0.0 and float(slices.max()) <= 1.0


def test_load_brain_slices_skips_empty(tmp_path):
    path = _write_synthetic_brain(tmp_path, shape=(20, 20, 12))
    # Only 6 slices (z=3..8) have content: with a typical min_foreground, <= 6 remain.
    slices = load_brain_slices(path, max_slices=64, scale=2, min_foreground=0.05)
    assert 0 < slices.shape[0] <= 6
