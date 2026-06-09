# Model Card — genai-mri-superres

> Documento di trasparenza sul modello. Segue lo spirito di Mitchell et al., *Model Cards
> for Model Reporting* (FAccT 2019, DOI 10.1145/3287560.3287596).

## Dettagli del modello
- **Compito**: super-risoluzione image-to-image ×2 di immagini mediche 2D (da 32×32 a 64×64).
- **Architettura**: ESPCN con sub-pixel convolution (Shi et al., CVPR 2016,
  DOI 10.1109/CVPR.2016.207); disponibile anche SRCNN (Dong et al., TPAMI 2016,
  DOI 10.1109/TPAMI.2015.2439281). Attivazioni **LeakyReLU** (vedi nota sotto).
- **Loss**: L1 (default) o MSE. **Config**: vedi `scripts/train.py`
  (default 30 epoche, batch=64, lr=1e-3, max 4000 campioni di training su CPU).
- **Baseline di confronto**: upsampling bicubico.

## Dati
- **BloodMNIST** 64×64 (MedMNIST v2, Yang et al., *Scientific Data* 2023,
  DOI 10.1038/s41597-022-01721-8), licenza **CC BY 4.0**.
- Le coppie LR/HR sono generate per **degradazione bicubica** controllata: è una
  super-risoluzione *simulata*, non da acquisizioni reali a bassa risoluzione.
- Dataset benchmark; **non** imaging clinico reale. Scaricato da `scripts/download_data.py`.

## Metriche (riproducibili, `results/metrics.json`, n_test=1000)
| | SR (ESPCN) | Bicubico |
|---|---|---|
| PSNR (dB) | 36.65 | 34.47 |
| SSIM | 0.962 | 0.951 |
- Guadagno: **+2.17 dB PSNR**, +0.0117 SSIM.

## Nota metodologica (debugging documentato)
Con ReLU standard la rete cadeva in **dying-ReLU** (residuo nullo → la SR copiava la
bicubica). Diagnosi: frazione di attivazioni nulle = 1.0. Fix: ReLU → **LeakyReLU** +
**test di regressione** che impedisce la regressione del bug.

## Uso previsto
- **Esclusivamente dimostrativo/didattico**.
- **NON è un dispositivo medico** e NON deve essere usato per ricostruire immagini destinate
  a diagnosi: la super-risoluzione può **introdurre dettagli plausibili ma non reali**
  (hallucination), rischio particolarmente serio in ambito clinico.

## Limiti e rischi
- Degradazione simulata (bicubica) ≠ degradazione reale di acquisizione.
- Dataset giocattolo → metriche non trasferibili a immagini cliniche reali.
- PSNR/SSIM non catturano la **fedeltà diagnostica**: servono metriche task-based e revisione
  esperta prima di qualsiasi uso reale.

## Come riprodurre
```
uv sync --extra dev
uv run python scripts/download_data.py
uv run python scripts/train.py --epochs 30
uv run python scripts/evaluate.py
```
