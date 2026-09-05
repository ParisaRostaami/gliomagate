# GliomaGate

One axial T1-post-contrast slice and a short consult note go in. The output is a tumor overlay, a structured report, and an evidence ledger.

Each field is tagged `image`, `retrieval`, or `ungrounded`. WHO grade is not allowed to come from the slice alone. If the note says left and the mask centroid is right, that is recorded as a contradiction.

I built this as a public slice-plus-note loop for neuro-oncology. Segmentation is trained on Cheng 2017 T1c glioma MRI (Figshare 1512427), split by patient. Cheng does not include WHO grade or lobe, so those fields are not treated as clinical labels.

**Demo:** [huggingface.co/spaces/Parisa/gliomagate](https://huggingface.co/spaces/Parisa/gliomagate)  
**Notebook:** [notebooks/gliomagate_results.ipynb](notebooks/gliomagate_results.ipynb)

## Data

Cheng 2017: 3064 T1c slices from 233 patients (meningioma / glioma / pituitary). This repo uses glioma only. Last run: **500 train / 48 test**, 128×128, patient-wise split (17 test patients). Masks are the published binary tumor outlines. Notes only state laterality and that the lesion is an enhancing glioma.

```text
python scripts/download_cheng.py
python scripts/train.py
```

Cheng, J. brain tumor dataset. Figshare 1512427. CC BY 4.0.

## Training

**Segmenter.** 2D U-Net, ~1.9M parameters, 25 epochs, BCE+Dice, flips/rotations. Train Dice **0.52**. Weights: `models/unet.pt`.

**Extractor.** Hash embeddings and rank-4 LoRA. Laterality on the held-out patients is **0.96**.

![Training curves](docs/figures/train_curves.png)

*Left: U-Net epoch loss. Right: LoRA student loss.*

## Held-out results (48 slices, 17 patients)

From `models/metrics.json`.

| Metric | Score |
| --- | ---: |
| Tumor Dice | 0.48 |
| Laterality | 0.96 |
| Contradiction rate (note vs mask laterality) | 0.10 |
| p95 latency | 87 ms |

![Segmentation gallery](docs/figures/gallery.png)

*Top: Cheng T1c. Middle: published mask. Bottom: prediction (red).*

![Dice histogram](docs/figures/dice_hist.png)

![Held-out scores](docs/figures/metrics_bars.png)

![Evidence ledger](docs/figures/ledger.png)

![Latency](docs/figures/latency.png)

## Method

Laterality and enhancement may be image-grounded. Grade may only be retrieval-grounded (`app/rag.py`). Missing both → `ungrounded`.

FastAPI on port 7860. `POST /v1/infer` returns the overlay, the ledger, retrieved chunks, and `latency_ms`.

## Run

```text
python -m pip install -r requirements.txt
python scripts/train.py
python scripts/make_figures.py
python scripts/build_notebook.py
python -m uvicorn app.main:app --port 7860
python -m pytest -q --cov=app --cov-fail-under=80
```

http://127.0.0.1:7860

```text
docker build -t gliomagate .
docker run --rm -p 7860:7860 gliomagate
```

## Layout

```text
app/            API, segmenter, LoRA, RAG, Cheng loader
scripts/        download_cheng, train, figures, notebook
docs/figures/   gallery from the Cheng held-out set
models/         unet.pt, LoRA adapters, train_log.json, metrics.json
data/test/      held-out Cheng slices (after train)
data/cheng_raw/ Figshare archives (gitignored)
```

MIT License
