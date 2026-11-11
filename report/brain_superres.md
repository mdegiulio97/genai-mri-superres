# ×2 super-resolution of real T1w brain MRI: ESPCN vs bicubic (held-out subject)

Short report on the *real-data track* of `genai-mri-superres`. It extends the
benchmark demo (MedMNIST BloodMNIST, blood-cell microscopy) to **real brain MRI**,
evaluating generalization on a subject never seen during training.

## Data

OpenNeuro **ds007116** ("Penn LEAD"), **CC0** license [3]. Anatomical **T1w**
volumes (defaced) at 3T. The high-resolution (HR) images are 2D slices extracted
and normalized to [0, 1]; the low-resolution (LR) version is obtained by ×2
**bicubic degradation** (*simulated* super-resolution, as is standard in the
literature). **Training** on the slices of subject `sub-19861`, **test** on a
**held-out** subject (`sub-138950`, never seen), so there is no leakage between
train and test. The raw data are not redistributed: they are downloaded with
`scripts/download_brain.py`.

## Method

- **ESPCN** [1]: extracts features at LR resolution and upscales with sub-pixel
  convolution; here in a **residual** variant on top of bicubic (the last layer is
  initialized to zero, so training starts from bicubic quality and learns only the
  corrections).
- **Baseline**: **bicubic** upscaling.
- Metrics: **PSNR** and **SSIM** (skimage), on 32 slices of the held-out subject.

## Results

Numbers from `results/brain_metrics.json` (held-out subject, 32 slices 176×192,
×2, ESPCN 40 epochs on CPU, fixed seed):

| Method | PSNR (dB) | SSIM |
|---|---|---|
| Bicubic | 27.63 | 0.9201 |
| **ESPCN (SR)** | **28.16** | **0.9271** |
| **Δ (SR − bicubic)** | **+0.52** | **+0.0070** |

On real brain anatomy and on an unseen subject, SR **beats** bicubic on both
metrics. The gain is smaller than on the MedMNIST benchmark (+2.17 dB), which is
expected when training on a single brain and evaluating across subjects.
Figures: `results/brain_examples.png` (bicubic/SR/HR),
`results/brain_psnr_hist.png` (per-slice PSNR distribution).

## Limitations

- **Simulated degradation** (bicubic): it does not reproduce the real acquisition
  degradation (k-space, noise, motion).
- **One training subject, one test subject**: a demonstration, not a study; a
  cohort is needed for robust estimates.
- **PSNR/SSIM** do not measure **diagnostic fidelity**; super-resolution can
  introduce plausible but non-existent details (a risk in clinical settings).
- **Not a clinical tool**: research demo on public CC0 data.

## References

1. Shi W, Caballero J, Huszár F, et al. *Real-Time Single Image and Video
   Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network.* CVPR
   2016. DOI: [10.1109/CVPR.2016.207](https://doi.org/10.1109/CVPR.2016.207)
2. Dong C, Loy CC, He K, Tang X. *Image Super-Resolution Using Deep Convolutional
   Networks.* IEEE TPAMI 2016. DOI:
   [10.1109/TPAMI.2015.2439281](https://doi.org/10.1109/TPAMI.2015.2439281)
3. OpenNeuro dataset **ds007116** "Penn LEAD", CC0 license.
   [https://openneuro.org/datasets/ds007116](https://openneuro.org/datasets/ds007116)
