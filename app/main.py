"""App Streamlit: demo di super-risoluzione SR vs bicubica.

Mostra, per un'immagine di test:
  LR in input -> SR (rete) vs bicubica, confronto affiancato + PSNR/SSIM.

Per restare leggera al primo render NON esegue training. Carica il checkpoint
``results/model.pt`` solo su richiesta (bottone) e tramite cache. Se il
checkpoint manca, mostra un messaggio e si ferma con ``st.stop()``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from srmri.baseline import bicubic_sr  # noqa: E402
from srmri.data import fake_hr_dataset, load_hr_split  # noqa: E402
from srmri.degrade import downsample  # noqa: E402
from srmri.metrics import psnr, ssim  # noqa: E402
from srmri.model import build_model  # noqa: E402

CKPT = ROOT / "results" / "model.pt"

st.set_page_config(page_title="GenAI MRI Super-Resolution", layout="wide")
st.title("Super-risoluzione di immagini mediche — SR (rete) vs bicubica")
st.caption(
    "Rete generativa compatta (ESPCN/SRCNN) addestrata su MedMNIST v2. "
    "Confronto image-to-image contro la baseline bicubica."
)


@st.cache_resource(show_spinner=False)
def load_model():
    """Carica il checkpoint addestrato (cache). Ritorna (model, meta) o None."""
    if not CKPT.exists():
        return None
    ckpt = torch.load(CKPT, map_location="cpu")
    model = build_model(ckpt["model"], scale=ckpt["scale"], channels=ckpt["channels"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt


@st.cache_data(show_spinner=False)
def get_sample(flag: str, hr_size: int, scale: int, idx: int) -> torch.Tensor:
    """Restituisce un'immagine HR di test. Fallback sintetico se offline."""
    try:
        hr = load_hr_split("test", flag=flag, hr_size=hr_size, download=True)
    except Exception:
        hr = fake_hr_dataset(n=8, hr_size=hr_size)
    idx = idx % hr.size(0)
    return hr[idx : idx + 1]


loaded = load_model()
if loaded is None:
    st.warning(
        "Checkpoint `results/model.pt` non trovato. Esegui prima "
        "`uv run python scripts/train.py --epochs 30` per addestrare il modello."
    )
    st.stop()

model, ckpt = loaded
scale = ckpt["scale"]
hr_size = ckpt["hr_size"]
flag = ckpt["flag"]

st.sidebar.header("Parametri")
idx = st.sidebar.slider("Indice immagine di test", 0, 200, 0)
st.sidebar.write(f"Modello: **{ckpt['model'].upper()}**")
st.sidebar.write(f"Dataset: **{flag}**, HR **{hr_size}×{hr_size}**, scala **×{scale}**")

if st.button("Esegui super-risoluzione", type="primary"):
    hr = get_sample(flag, hr_size, scale, idx)
    lr = downsample(hr, scale=scale)
    with torch.no_grad():
        sr = model(lr)
    bic = bicubic_sr(lr, scale=scale)

    col1, col2, col3, col4 = st.columns(4)
    col1.image(lr[0, 0].numpy(), caption=f"LR input ({lr.shape[-1]}×{lr.shape[-2]})",
               clamp=True, use_container_width=True)
    col2.image(bic[0, 0].numpy(), caption="Bicubica", clamp=True, use_container_width=True)
    col3.image(sr[0, 0].numpy(), caption="SR (rete)", clamp=True, use_container_width=True)
    col4.image(hr[0, 0].numpy(), caption="HR (target)", clamp=True, use_container_width=True)

    st.subheader("Metriche (vs HR target)")
    m1, m2 = st.columns(2)
    m1.metric("PSNR bicubica", f"{psnr(bic, hr):.2f} dB")
    m1.metric("PSNR SR (rete)", f"{psnr(sr, hr):.2f} dB",
              delta=f"{psnr(sr, hr) - psnr(bic, hr):+.2f} dB")
    m2.metric("SSIM bicubica", f"{ssim(bic, hr):.4f}")
    m2.metric("SSIM SR (rete)", f"{ssim(sr, hr):.4f}",
              delta=f"{ssim(sr, hr) - ssim(bic, hr):+.4f}")
else:
    st.info("Premi **Esegui super-risoluzione** per generare il confronto.")
