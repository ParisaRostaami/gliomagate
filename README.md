# GliomaGate

**Grounded neuro-oncology inference:** a 2D MRI slice + a clinical note in, a tumor mask + a structured report + an **evidence ledger** out.

Every field is `image`, `retrieval`, or `ungrounded`. Grade is never allowed to claim image proof from one slice. If the note says left and the mask centroid is right, the API emits a **contradiction** instead of averaging.

> Built and deployed a clinical NLP + medical imaging inference service — LoRA-tuned extractor (rank-4 student distilled from rank-8 teacher, 4-bit frozen base) + conv-stem FCN segmenter on 80 train / 32 test synthetic T1c-like slices, served via FastAPI + Docker on Hugging Face Spaces, **p95 35ms**, **~47 serial req/s** in-process, Cloud Run list-price **~$1e-6 / inference** at that p95 (1 vCPU). CI with GitHub Actions, pytest coverage **82%**, Prometheus `/metrics`. Live demo · this repo.

That sentence is filled from **measured** files, not invented SLA copy. Read `models/metrics.json` and `models/bench.json`.

## Why this exists

Hiring loops in 2026 want systems people: Docker, CI, latency, cost, monitoring — and LLM people: LoRA, RAG eval, groundedness. This repo is one service that is both, in the domain of brain-tumor imaging + neuro-oncology notes.

It is **not** BraTS, **not** MIMIC, **not** a 7B vLLM deployment on a free CPU Space. The generator is synthetic so the harness is reproducible without gated data. The GPU path (`training/train_qlora_t5.py`, `deploy/vllm-compose.yaml`) is real code that no-ops without CUDA, instead of a fake GPU claim.

## What the API does

`POST /v1/infer` (multipart: `note` + optional `image`, or `seed` for a built-in sample)

Returns:

- overlay PNG of edema (teal) vs enhancing core (red)
- structured fields: laterality, lobe, grade, enhancement, symptom
- provenance + evidence spans
- top-k guideline chunks (TF-IDF RAG over WHO-style protocol text)
- `latency_ms`

Also: `GET /health`, `GET /metrics` (Prometheus), `GET /v1/eval` (cached harness), `GET /` (demo UI).

## Held-out harness (32 synthetic cases)

From `models/metrics.json` after `python scripts/train.py`:

| Metric | Value | Meaning |
| --- | --- | --- |
| Tumor Dice (edema+core mean) | **0.79** | slice segmentation |
| Field accuracy (templated notes) | **1.00** | notes name the fields; lexicon + LoRA |
| nDCG@5 | **0.73** | retrieval quality |
| Recall@5 (grade-relevant chunk) | **1.00** | |
| Grounded fraction | **0.79** | fields with image or retrieval support |
| Grounded fraction, no RAG | **0.38** | ablation: retrieval is doing the work |
| Contradiction rate | **0.00** on this split | note vs mask laterality |

Templated notes make extraction look easy on purpose so the *interesting* number is groundedness under ablation. Interviewers should ask that.

## Latency and cost (measured)

From `models/bench.json` (80 in-process requests via TestClient, this workstation):

- p50 **18ms**, p95 **35ms**, mean **21ms**
- serial throughput **47 req/s**
- Cloud Run CPU list price × p95 × 1 vCPU ≈ **$0.000001 / call**

That is not a 40 req/s soak on autoscaled replicas, and it is not $0.003. If a recruiter wants the bigger number, the honest answer is “GPU LLM serving would be cents-per-call; this CPU student is sub-millicent.”

- **Segmenter:** multi-scale conv stem (Gaussian / Sobel / coordinates) + MLP pixel head, largest-component cleanup. Int8 linear student is trained as a distilled fast path.
- **Extractor:** hash embeddings + LoRA (`y = xW_q + xAB`) with 4-bit frozen W (QLoRA-style). Rank-8 teacher distilled into rank-4 student.
- **Lexicon gate:** if the note explicitly says `temporal` / `glioblastoma` / …, trust the span. Production clinical NLP does this; LoRA is the fallback.
- **RAG:** sklearn TF-IDF over 10 guideline chunks, cosine top-k, nDCG/recall harness.
- **Serving:** FastAPI, Docker, Prometheus histograms.
- **Cloud:** live URL on Hugging Face Spaces (Docker). `deploy/cloudrun.yaml` is the GCP template. Cost estimate uses [Cloud Run CPU list price](https://cloud.google.com/run/pricing) × measured p95, not a pretend AWS invoice.

## Run locally

```text
python -m pip install -r requirements.txt
python scripts/train.py          # writes models/
python -m uvicorn app.main:app --port 7860
python -m pytest -q --cov=app --cov-fail-under=80
python scripts/bench.py          # writes models/bench.json
```

Open http://127.0.0.1:7860

## Docker

```text
docker build -t gliomagate .
docker run --rm -p 7860:7860 gliomagate
```

## What I will not claim

- Real BraTS Dice on clinical 3D volumes
- p95 at 40 req/s on a GPU I did not rent
- $0.003/inference unless `models/bench.json` says so after measurement
- that Flan-T5 QLoRA ran on this machine (it did not; the script exits 0 without CUDA)

MIT License
