# genai-mri-superres

**Image-to-image** super-resolution of 2D medical images with a compact
convolutional generative network in PyTorch. The project trains a
super-resolution CNN (**ESPCN** / **SRCNN**) to reconstruct a high-resolution
(HR) image from a low-resolution (LR) version obtained by bicubic degradation,
and compares it against the **bicubic** interpolation baseline using **PSNR**
and **SSIM**.

The project has two tracks:

- **Benchmark track (BloodMNIST).** The main benchmark uses BloodMNIST from
  MedMNIST v2, which contains **blood-cell microscopy images, not MRI**.
- **Real brain MRI track (OpenNeuro ds007116).** A separate track applies the
  same method to real T1w brain MRI slices, with evaluation on a held-out
  subject. This is the only MRI result in the repository.

The "GenAI" framing refers to conditional image-to-image generation: the network
generates the high-frequency detail missing from the LR image. A natural
extension (not implemented here, to keep the project runnable on CPU or an
entry-level GPU) is to replace the CNN with a conditional diffusion model (e.g.
SR3, Saharia et al. 2022, arXiv:2104.07636).

## Data

- **MedMNIST v2** (Yang et al., *Scientific Data* 2023,
  DOI [10.1038/s41597-022-01721-8](https://doi.org/10.1038/s41597-022-01721-8),
  CC BY 4.0 license). Downloaded automatically, no registration required.
- Dataset used: **BloodMNIST** at **HR = 64×64** (parameter `size=64` of the
  `medmnist` library). The LR images (32×32) are obtained by ×2 bicubic
  downsampling; the network learns to bring them back to 64×64.
- Images are converted to **grayscale** (1 channel) so that the task focuses on
  structural reconstruction, in line with classic super-resolution protocols
  (SRCNN/ESPCN operate on the luminance channel).

## Data and licenses

- **MedMNIST v2, BloodMNIST** — Yang J. et al., *MedMNIST v2 – A large-scale
  lightweight benchmark for 2D and 3D biomedical image classification*,
  *Scientific Data* 10:41 (2023),
  DOI [10.1038/s41597-022-01721-8](https://doi.org/10.1038/s41597-022-01721-8).
  License: **CC BY 4.0**. Downloaded at runtime by the `medmnist` library
  (`scripts/download_data.py`); not committed to this repository.
  BloodMNIST contains blood-cell microscopy images, so the benchmark track
  measures super-resolution on microscopy images, not on MRI.
- **OpenNeuro ds007116** ("Penn LEAD"), T1w brain MRI — License: **CC0**.
  Downloaded by `scripts/download_brain.py` into `data/brain/`, which is
  git-ignored; the data are not committed to this repository.
  This dataset is used only by the separate real brain MRI track.

## Architecture

- **ESPCN** (default): Shi et al., *Real-Time Single Image and Video
  Super-Resolution Using an Efficient Sub-Pixel CNN*, CVPR 2016,
  arXiv:[1609.05158](https://arxiv.org/abs/1609.05158). Features are extracted
  at the LR resolution, followed by a final upscaling step with **sub-pixel
  convolution** (`PixelShuffle`); this is efficient because upscaling happens
  only at the end.
- **SRCNN** (alternative): Dong et al., *Image Super-Resolution Using Deep
  Convolutional Networks*, TPAMI 2016,
  DOI [10.1109/TPAMI.2015.2439281](https://doi.org/10.1109/TPAMI.2015.2439281).
  Works on an image that has already been bicubically upscaled and refines its
  details.

## Layout

```
genai-mri-superres/
├── app/main.py                 # Streamlit: LR → SR vs bicubic, PSNR/SSIM
├── src/srmri/
│   ├── data.py                 # MedMNIST HR + synthetic fake loader
│   ├── model.py                # ESPCN / SRCNN
│   ├── degrade.py              # bicubic downsample/upsample
│   ├── train.py                # training loop (L1/MSE)
│   ├── metrics.py              # PSNR, SSIM
│   ├── baseline.py             # bicubic upscaling
│   └── real_brain.py           # 2D slices from real T1w brain MRI (NIfTI)
├── scripts/download_data.py    # downloads MedMNIST (idempotent)
├── scripts/train.py            # trains → results/model.pt
├── scripts/evaluate.py         # SR vs bicubic → results/metrics.json + figures
├── scripts/download_brain.py   # downloads two T1w volumes from OpenNeuro ds007116
├── scripts/evaluate_brain.py   # brain MRI track → results/brain_*
├── tests/                      # offline pytest (synthetic tensors)
├── notebooks/01_validation.ipynb
├── report/brain_superres.md    # short report on the brain MRI track
├── results/                    # figures + metrics JSON files
└── data/                       # downloaded datasets (not committed)
```

## Usage

```bash
uv sync --extra dev                       # install dependencies (torch is large)
uv run python scripts/download_data.py    # download BloodMNIST size=64
uv run python scripts/train.py --epochs 30
uv run python scripts/evaluate.py         # metrics + figures
uv run streamlit run app/main.py          # interactive demo
```

## Results (benchmark track, BloodMNIST)

Evaluation on the BloodMNIST test set (blood-cell microscopy, HR 64×64,
grayscale, ×2), ESPCN model trained for 30 epochs on CPU. Numbers produced by
`scripts/evaluate.py` (`results/metrics.json`):

| Method            | PSNR (dB) | SSIM   |
|-------------------|-----------|--------|
| Bicubic           | 34.47     | 0.9506 |
| **ESPCN (SR)**    | **36.65** | **0.9623** |
| **Δ (SR − bicubic)** | **+2.17** | **+0.0117** |

(BloodMNIST test set, 1000 images, HR 64×64 grayscale, ×2, ESPCN 30 epochs on CPU.)

The SR network **beats** the bicubic baseline in both PSNR (+2.17 dB) and SSIM,
averaged over the test set. Figures in `results/`: *before/after* examples and a
histogram of per-image PSNR (SR vs bicubic). Numbers and figures are regenerated
by `scripts/evaluate.py` (`results/metrics.json`).

> **Methodological note (dying ReLU).** A first version of ESPCN used `ReLU` in
> the feature extractor: with typical learning rates and large batches the
> activations "died" (zero output, zero gradient), the residual collapsed and
> the network only **matched** bicubic (Δ ≈ 0, frozen loss). The problem was
> diagnosed by measuring the fraction of zero activations (= 1.000) and fixed by
> replacing `ReLU` with `LeakyReLU` (non-zero negative slope, so the gradient
> stays alive). A regression test (`tests/test_pipeline.py`) reproduces the
> failure offline and prevents it from coming back.

## Real brain MRI track (OpenNeuro ds007116, CC0)

Besides the BloodMNIST benchmark, the project applies super-resolution to
**real T1w brain MRI** (OpenNeuro **ds007116** "Penn LEAD", CC0). The HR images
are slices of real T1w volumes; the HR→LR degradation is still simulated
(bicubic), as is standard in SR work. Training uses one subject and evaluation
uses a **held-out subject** (never seen during training), to measure
cross-subject generalization.

```bash
uv run python scripts/download_brain.py     # ~15 MB from S3 (data git-ignored)
uv run python scripts/evaluate_brain.py     # ESPCN vs bicubic → results/brain_*
```

Numbers from `results/brain_metrics.json` (held-out subject sub-138950, 32
slices 176×192, ×2, ESPCN 40 epochs on CPU):

| Method | PSNR (dB) | SSIM |
|---|---|---|
| Bicubic | 27.63 | 0.9201 |
| **ESPCN (SR)** | **28.16** | **0.9271** |
| **Δ (SR − bicubic)** | **+0.52** | **+0.0070** |

On real brain anatomy and on an unseen subject, SR still **beats** bicubic on
both metrics; the gain is smaller than on the BloodMNIST benchmark, which is
expected given the cross-subject evaluation and training on a single brain.
Figures: `results/brain_examples.png`, `results/brain_psnr_hist.png`.
Report with citations: [`report/brain_superres.md`](report/brain_superres.md).

## Tests

```bash
uv run ruff check .
uv run pytest -q
```

The tests are **offline** and fast: they use synthetic images (tensors) and do
not download MedMNIST or OpenNeuro data. They check degradation/reconstruction
shapes, model forward passes, high PSNR for identical images, loss reduction
during training, the brain slice loader on a small synthetic NIfTI volume, and
a smoke test of the Streamlit app.

## Development

A `Makefile` collects the most common commands. The targets are plain
**shortcuts** for the `uv` commands documented above (they add no new logic);
they require `make` and `uv` to be installed.

| Command       | Shortcut for                                | Description                                            |
|---------------|---------------------------------------------|--------------------------------------------------------|
| `make help`   | —                                           | Lists the available targets (default).                 |
| `make setup`  | `uv sync --extra dev`                       | Installs runtime and development dependencies.         |
| `make data`   | `uv run python scripts/download_data.py`    | Downloads the BloodMNIST dataset (idempotent).         |
| `make train`  | `uv run python scripts/train.py`            | Trains the ESPCN model → `results/model.pt`.           |
| `make test`   | `uv run pytest -q`                          | Runs the offline test suite.                           |
| `make lint`   | `uv run ruff check .`                       | Runs the ruff linter on the project.                   |
| `make app`    | `uv run streamlit run app/main.py`          | Starts the interactive Streamlit demo.                 |
| `make all`    | `setup` + `lint` + `test`                   | Sets up the environment, then runs lint and tests.     |

## References

1. Shi et al., *Real-Time Single Image and Video Super-Resolution Using an
   Efficient Sub-Pixel Convolutional Neural Network*, CVPR 2016.
   arXiv:[1609.05158](https://arxiv.org/abs/1609.05158)
2. Dong et al., *Image Super-Resolution Using Deep Convolutional Networks*,
   IEEE TPAMI 2016.
   DOI [10.1109/TPAMI.2015.2439281](https://doi.org/10.1109/TPAMI.2015.2439281)
3. Yang et al., *MedMNIST v2 — A large-scale lightweight benchmark for 2D and 3D
   biomedical image classification*, *Scientific Data* 2023.
   DOI [10.1038/s41597-022-01721-8](https://doi.org/10.1038/s41597-022-01721-8)
4. Saharia et al., *Image Super-Resolution via Iterative Refinement* (SR3), 2022.
   arXiv:[2104.07636](https://arxiv.org/abs/2104.07636)
5. OpenNeuro dataset **ds007116** "Penn LEAD", CC0.
   [https://openneuro.org/datasets/ds007116](https://openneuro.org/datasets/ds007116)

## License

Code: MIT © Michele De Giulio 2026 (see `LICENSE`). The MedMNIST v2 dataset is
distributed under the CC BY 4.0 license by its authors, and OpenNeuro ds007116
under CC0; follow their terms.
