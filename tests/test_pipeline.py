"""Test offline e veloci della pipeline di super-risoluzione.

Usano SOLO tensori sintetici: nessun download di MedMNIST.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from srmri.baseline import bicubic_sr  # noqa: E402
from srmri.data import fake_hr_dataset, fake_loaders  # noqa: E402
from srmri.degrade import downsample, make_lr_hr_pair, upsample  # noqa: E402
from srmri.metrics import psnr, psnr_per_image, ssim  # noqa: E402
from srmri.model import ESPCN, SRCNN, build_model  # noqa: E402
from srmri.train import get_device, train  # noqa: E402


def test_downsample_upsample_shapes():
    hr = fake_hr_dataset(n=4, hr_size=32)
    lr = downsample(hr, scale=2)
    assert lr.shape == (4, 1, 16, 16)
    back = upsample(lr, scale=2)
    assert back.shape == (4, 1, 32, 32)
    # range preservato in [0,1]
    assert lr.min() >= 0.0 and lr.max() <= 1.0


def test_downsample_3d_single_image():
    hr = fake_hr_dataset(n=1, hr_size=32)[0]  # (1,32,32)
    lr = downsample(hr, scale=2)
    assert lr.shape == (1, 16, 16)


def test_make_lr_hr_pair():
    hr = fake_hr_dataset(n=3, hr_size=64)
    lr, hr2 = make_lr_hr_pair(hr, scale=2)
    assert lr.shape == (3, 1, 32, 32)
    assert torch.equal(hr, hr2)


def test_espcn_forward_shape():
    model = ESPCN(scale=2, channels=1)
    lr = torch.rand(2, 1, 16, 16)
    out = model(lr)
    assert out.shape == (2, 1, 32, 32)
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_srcnn_forward_shape():
    model = SRCNN(scale=2, channels=1)
    lr = torch.rand(2, 1, 16, 16)
    out = model(lr)
    assert out.shape == (2, 1, 32, 32)


def test_build_model_factory():
    assert isinstance(build_model("espcn"), ESPCN)
    assert isinstance(build_model("srcnn"), SRCNN)
    try:
        build_model("foo")
    except ValueError:
        pass
    else:
        raise AssertionError("Atteso ValueError per modello sconosciuto")


def test_psnr_identical_is_high():
    hr = fake_hr_dataset(n=4, hr_size=32)
    # immagini identiche -> PSNR molto alto
    assert psnr(hr, hr) > 60.0
    assert ssim(hr, hr) > 0.99


def test_psnr_per_image_length():
    hr = fake_hr_dataset(n=5, hr_size=32)
    noisy = (hr + 0.05 * torch.randn_like(hr)).clamp(0, 1)
    vals = psnr_per_image(noisy, hr)
    assert vals.shape == (5,)
    assert (vals < 60).all()  # con rumore il PSNR scende


def test_bicubic_sr_matches_upsample():
    lr = torch.rand(2, 1, 16, 16)
    out = bicubic_sr(lr, scale=2)
    assert out.shape == (2, 1, 32, 32)
    assert torch.allclose(out, upsample(lr, scale=2))


def test_training_reduces_loss():
    device = get_device("cpu")
    train_loader, _ = fake_loaders(n_train=24, n_test=8, hr_size=32, scale=2, batch_size=8)
    model = build_model("espcn", scale=2, channels=1)
    out = train(model, train_loader, epochs=12, lr=1e-3, device=device)
    losses = out["history"]["train_loss"]
    assert len(losses) == 12
    # la loss finale deve essere sensibilmente inferiore a quella iniziale
    assert losses[-1] < losses[0]


# Config "di stress" che riproduce offline il dying-ReLU: batch grande (64) e HR
# 64 con learning rate elevato facevano morire tutte le ReLU del feature
# extractor (residuo nullo -> la rete eguagliava ma non superava la bicubica).
# Con LeakyReLU il gradiente resta vivo e la rete supera la bicubica.
_STRESS = dict(n_train=256, n_test=64, hr_size=64, scale=2, batch_size=64)


def test_trained_model_beats_bicubic_under_stress_lr():
    """Regressione dying-ReLU: con lr elevato la rete deve comunque SUPERARE la
    bicubica. Col vecchio ReLU le feature morivano e il delta era ~0 (fail)."""
    device = get_device("cpu")
    train_loader, test_loader = fake_loaders(batch_size=_STRESS["batch_size"],
                                             n_train=_STRESS["n_train"],
                                             n_test=_STRESS["n_test"],
                                             hr_size=_STRESS["hr_size"], scale=2)
    model = build_model("espcn", scale=2, channels=1)
    train(model, train_loader, epochs=40, lr=5e-3, device=device)
    model.eval()
    srs, bics, hrs = [], [], []
    with torch.no_grad():
        for lr, hr in test_loader:
            srs.append(model(lr))
            bics.append(bicubic_sr(lr, scale=2))
            hrs.append(hr)
    sr, bic, hr = torch.cat(srs), torch.cat(bics), torch.cat(hrs)
    assert psnr(sr, hr) > psnr(bic, hr) + 0.08


def test_residual_is_nontrivial_under_stress_lr():
    """Il residuo appreso (output - bicubica) non deve collassare a zero: è la
    firma diretta del dying-ReLU (residuo ~1e-4)."""
    device = get_device("cpu")
    train_loader, test_loader = fake_loaders(batch_size=_STRESS["batch_size"],
                                             n_train=_STRESS["n_train"],
                                             n_test=_STRESS["n_test"],
                                             hr_size=_STRESS["hr_size"], scale=2)
    model = build_model("espcn", scale=2, channels=1)
    train(model, train_loader, epochs=40, lr=5e-3, device=device)
    model.eval()
    lr, _ = next(iter(test_loader))
    with torch.no_grad():
        residual = (model(lr) - bicubic_sr(lr, scale=2)).abs().mean().item()
    assert residual > 1e-3, f"residuo collassato a zero (mean|res|={residual:.2e})"
