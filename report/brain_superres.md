# Super-risoluzione ×2 di MRI cerebrale T1w reale: ESPCN vs bicubico (soggetto held-out)

Mini-report del *track dati reali* di `genai-mri-superres`. Estende la demo su
benchmark (MedMNIST) a **MRI cerebrale reale**, valutando la generalizzazione su un
soggetto mai visto.

## Dati

OpenNeuro **ds007116** ("Penn LEAD"), licenza **CC0** [3]. Volumi anatomici **T1w**
(defaced) a 3T. Le immagini ad alta risoluzione (HR) sono slice 2D estratte e
normalizzate in [0, 1]; la versione a bassa risoluzione (LR) è ottenuta per
**degradazione bicubica** ×2 (super-risoluzione *simulata*, come standard in
letteratura). **Training** sulle slice del soggetto `sub-19861`, **test** su un
soggetto **held-out** (`sub-138950`, mai visto) — niente leakage tra train e test.
I dati grezzi non sono ridistribuiti: si scaricano con `scripts/download_brain.py`.

## Metodo

- **ESPCN** [1]: estrae feature a risoluzione LR e fa l'upscaling con sub-pixel
  convolution; qui in variante a **residuo** sopra la bicubica (l'ultimo strato è
  inizializzato a zero → parte dalla qualità bicubica e impara solo le correzioni).
- **Baseline**: upscaling **bicubico**.
- Metriche: **PSNR** e **SSIM** (skimage), su 32 slice del soggetto held-out.

## Risultati

Numeri reali da `results/brain_metrics.json` (soggetto held-out, 32 slice 176×192,
×2, ESPCN 40 epoche su CPU, seed fisso):

| Metodo | PSNR (dB) | SSIM |
|---|---|---|
| Bicubico | 27.63 | 0.9201 |
| **ESPCN (SR)** | **28.16** | **0.9271** |
| **Δ (SR − bicubico)** | **+0.52** | **+0.0070** |

Su anatomia cerebrale reale e su un soggetto mai visto, la SR **batte** la bicubica
su entrambe le metriche. Il guadagno è più contenuto del benchmark MedMNIST
(+2.17 dB): risultato **onesto**, atteso quando si addestra su un solo cervello e si
valuta cross-soggetto. Figure: `results/brain_examples.png` (bicubico/SR/HR),
`results/brain_psnr_hist.png` (distribuzione PSNR per slice).

## Limiti

- **Degradazione simulata** (bicubica): non riproduce la vera degradazione di
  acquisizione (k-space, rumore, moto).
- **Un soggetto di training, un soggetto di test**: dimostrazione, non uno studio;
  serve una coorte per stime robuste.
- **PSNR/SSIM** non misurano la **fedeltà diagnostica**; la super-risoluzione può
  introdurre dettagli plausibili ma non reali (rischio in ambito clinico).
- **Non è uno strumento clinico**: demo di ricerca su dati pubblici CC0.

## Riferimenti

1. Shi W, Caballero J, Huszár F, et al. *Real-Time Single Image and Video
   Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network.* CVPR
   2016. DOI: [10.1109/CVPR.2016.207](https://doi.org/10.1109/CVPR.2016.207)
2. Dong C, Loy CC, He K, Tang X. *Image Super-Resolution Using Deep Convolutional
   Networks.* IEEE TPAMI 2016. DOI:
   [10.1109/TPAMI.2015.2439281](https://doi.org/10.1109/TPAMI.2015.2439281)
3. OpenNeuro dataset **ds007116** "Penn LEAD", licenza CC0.
   [https://openneuro.org/datasets/ds007116](https://openneuro.org/datasets/ds007116)
