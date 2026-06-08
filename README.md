# genai-mri-superres

Super-risoluzione **image-to-image** di immagini mediche 2D tramite una rete
generativa convoluzionale compatta in PyTorch. Il progetto addestra una CNN di
super-risoluzione (**ESPCN** / **SRCNN**) a ricostruire un'immagine ad alta
risoluzione (HR) a partire da una versione a bassa risoluzione (LR) ottenuta per
degradazione bicubica, e la confronta contro la baseline di interpolazione
**bicubica** usando le metriche **PSNR** e **SSIM**.

L'inquadramento "GenAI" è la generazione condizionata image-to-image: la rete
genera i dettagli ad alta frequenza mancanti nell'immagine LR. Un'estensione
naturale (non implementata qui per restare eseguibile su CPU/GPU entry-level) è
sostituire la CNN con un modello a diffusione condizionata (es. SR3,
Saharia et al. 2022, arXiv:2104.07636).

## Dati

- **MedMNIST v2** (Yang et al., *Scientific Data* 2023,
  DOI [10.1038/s41597-022-01721-8](https://doi.org/10.1038/s41597-022-01721-8),
  licenza CC BY 4.0). Auto-scaricabile, nessuna registrazione.
- Dataset usato: **BloodMNIST** alla risoluzione **HR = 64×64** (parametro
  `size=64` della libreria `medmnist`). Le immagini LR (32×32) sono ottenute per
  downsample bicubico ×2; la rete impara a riportarle a 64×64.
- Le immagini sono convertite in **scala di grigi** (1 canale) per concentrare il
  task sulla ricostruzione strutturale, coerentemente con i protocolli di
  super-risoluzione classici (SRCNN/ESPCN operano sul canale di luminanza).

## Architettura

- **ESPCN** (default): Shi et al., *Real-Time Single Image and Video
  Super-Resolution Using an Efficient Sub-Pixel CNN*, CVPR 2016,
  arXiv:[1609.05158](https://arxiv.org/abs/1609.05158). Estrazione di feature a
  piena risoluzione LR + upscaling finale tramite **sub-pixel convolution**
  (`PixelShuffle`), efficiente perché l'upscaling avviene solo alla fine.
- **SRCNN** (alternativa): Dong et al., *Image Super-Resolution Using Deep
  Convolutional Networks*, TPAMI 2016,
  DOI [10.1109/TPAMI.2015.2439281](https://doi.org/10.1109/TPAMI.2015.2439281).
  Lavora su un'immagine già upscalata bicubicamente e ne raffina i dettagli.

## Struttura

```
genai-mri-superres/
├── app/main.py                 # Streamlit: LR → SR vs bicubica, PSNR/SSIM
├── src/srmri/
│   ├── data.py                 # MedMNIST HR + fake loader sintetico
│   ├── model.py                # ESPCN / SRCNN
│   ├── degrade.py              # downsample/upsample bicubico
│   ├── train.py                # loop di training (L1/MSE)
│   ├── metrics.py              # PSNR, SSIM
│   └── baseline.py             # upscaling bicubico
├── scripts/download_data.py    # scarica MedMNIST (idempotente)
├── scripts/train.py            # addestra → results/model.pt
├── scripts/evaluate.py         # SR vs bicubica → results/metrics.json + figure
├── tests/                      # pytest offline (tensori sintetici)
├── notebooks/01_validation.ipynb
├── results/                    # figure + metrics.json
└── data/                       # dataset scaricato (non committato)
```

## Uso

```bash
uv sync --extra dev                       # installa dipendenze (torch è pesante)
uv run python scripts/download_data.py    # scarica BloodMNIST size=64
uv run python scripts/train.py --epochs 30
uv run python scripts/evaluate.py         # metriche + figure
uv run streamlit run app/main.py          # demo interattiva
```

## Risultati

Valutazione su test set BloodMNIST (HR 64×64, scala di grigi, ×2), modello ESPCN
addestrato 30 epoche su CPU. Numeri reali rigenerati da `scripts/evaluate.py`
(`results/metrics.json`):

| Metodo            | PSNR (dB) | SSIM   |
|-------------------|-----------|--------|
| Bicubica          | 34.47     | 0.9506 |
| **ESPCN (SR)**    | **36.65** | **0.9623** |
| **Δ (SR − bicubica)** | **+2.17** | **+0.0117** |

(test set BloodMNIST, 1000 immagini, HR 64×64 grigi, ×2, ESPCN 30 epoche su CPU.)

La rete SR **batte** la baseline bicubica sia in PSNR (+2.17 dB) sia in SSIM su
tutto il test set. Figure in `results/`: esempi *before/after* e istogramma del
PSNR per-immagine (SR vs bicubica). Numeri e figure sono rigenerati
automaticamente da `scripts/evaluate.py` (`results/metrics.json`).

> **Nota metodologica (dying ReLU).** Una prima versione dell'ESPCN usava `ReLU`
> nel feature extractor: con learning rate tipici e batch grandi le attivazioni
> "morivano" (output a zero, gradiente nullo), il residuo collassava e la rete
> si limitava a **eguagliare** la bicubica (Δ ≈ 0, loss congelata). Il problema è
> stato diagnosticato misurando la frazione di attivazioni nulle (= 1.000) e
> risolto sostituendo `ReLU` con `LeakyReLU` (pendenza negativa non nulla → il
> gradiente resta vivo). Un test di regressione (`tests/test_pipeline.py`)
> riproduce il guasto offline e ne previene il ritorno.

## Test

```bash
uv run ruff check .
uv run pytest -q
```

I test sono **offline** e veloci: usano immagini sintetiche (tensori), non
scaricano MedMNIST. Verificano shape di degradazione/ricostruzione, forward dei
modelli, PSNR alto per immagini identiche, riduzione della loss in training e
robustezza dello smoke test dell'app Streamlit.

## Riferimenti

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

## Licenza

Codice: MIT © Michele De Giulio 2026 (vedi `LICENSE`). Il dataset MedMNIST v2 è
distribuito sotto licenza CC BY 4.0 dai rispettivi autori; rispettarne i termini.
