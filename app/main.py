"""GliomaGate FastAPI service + demo UI."""

from __future__ import annotations

import base64
import io
import json
import time
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from starlette.responses import Response

from app.eval_harness import run_harness
from app.ground import assemble_ledger
from app.lora_extract import extract, load_lora
from app.rag import GuidelineIndex
from app.segment import image_findings, load_segmenter, predict_mask_mlp
from app.synth import SIZE, load_case, load_manifest

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
STATIC = ROOT / "static"

INFER_LAT = Histogram(
    "gliomagate_infer_seconds",
    "End-to-end inference latency",
    buckets=(0.01, 0.02, 0.05, 0.08, 0.12, 0.2, 0.4, 0.8, 1.5),
)
INFER_N = Counter("gliomagate_infer_total", "Inference calls", ["status"])

STATE: dict = {}


def _png_b64(arr: np.ndarray) -> str:
    if arr.ndim == 2:
        img = Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8), mode="L")
    else:
        img = Image.fromarray(arr.astype(np.uint8), mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def overlay_rgb(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    rgb = np.stack([image, image, image], axis=-1)
    rgb = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
    out = rgb.copy()
    if (mask == 2).any():
        out[mask == 1] = (0.45 * rgb[mask == 1] + 0.55 * np.array([40, 180, 120])).astype(np.uint8)
        out[mask == 2] = (0.35 * rgb[mask == 2] + 0.65 * np.array([220, 70, 80])).astype(np.uint8)
    else:
        out[mask > 0] = (0.35 * rgb[mask > 0] + 0.65 * np.array([220, 70, 80])).astype(np.uint8)
    return out


def decode_image(raw: bytes) -> np.ndarray:
    im = Image.open(io.BytesIO(raw)).convert("L").resize((SIZE, SIZE))
    return np.asarray(im, dtype=np.float32) / 255.0


@asynccontextmanager
async def lifespan(app: FastAPI):
    mlp, head = load_segmenter(MODELS / "segmenter.joblib")
    STATE["mlp"] = mlp
    STATE["head"] = head
    STATE["lora"] = load_lora(MODELS)
    STATE["index"] = GuidelineIndex()
    STATE["metrics"] = json.loads((MODELS / "metrics.json").read_text(encoding="utf-8")) if (MODELS / "metrics.json").exists() else {}
    yield
    STATE.clear()


app = FastAPI(
    title="GliomaGate",
    description="Grounded neuro-oncology inference: mask + structured report + evidence ledger.",
    version="0.1.0",
    lifespan=lifespan,
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
if STATIC.exists():
    app.mount("/assets", StaticFiles(directory=STATIC), name="assets")


@app.get("/", response_class=HTMLResponse)
def demo():
    return HTMLResponse((STATIC / "index.html").read_text(encoding="utf-8"))


@app.get("/health")
def health():
    return {"ok": True, "model_dir": str(MODELS), "metrics": STATE.get("metrics", {})}


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/v1/sample")
def sample(seed: int = 0):
    rows = load_manifest(ROOT / "data" / "test")
    if not rows:
        raise HTTPException(500, "No test slices. Run python scripts/train.py first.")
    meta = rows[int(seed) % len(rows)]
    image, labels, m = load_case(ROOT / "data" / "test", meta["case_id"])
    return {
        "image_png": _png_b64(image),
        "note": m["note"],
        "seed": seed,
        "case_id": m["case_id"],
        "hint": {"laterality": m.get("laterality"), "pid": m.get("pid")},
    }


def _infer(image: np.ndarray, note: str) -> dict:
    t0 = time.perf_counter()
    mask = predict_mask_mlp(STATE["mlp"], image)
    findings = image_findings(mask)
    extracted = extract(STATE["lora"], note, findings)
    hits = STATE["index"].search(STATE["index"].query_for_case(note, findings, extracted), k=4)
    ledger = assemble_ledger(extracted, findings, hits)
    elapsed = time.perf_counter() - t0
    INFER_LAT.observe(elapsed)
    INFER_N.labels("ok").inc()
    return {
        "latency_ms": round(elapsed * 1000, 2),
        "mask_png": _png_b64(overlay_rgb(image, mask)),
        "source_png": _png_b64(image),
        "report": ledger,
        "retrieval": [
            {"id": h.chunk_id, "title": h.title, "score": round(h.score, 3), "text": h.text} for h in hits
        ],
    }


@app.post("/v1/infer")
async def infer(
    note: str = Form(...),
    image: UploadFile | None = File(default=None),
    seed: int | None = Form(default=None),
):
    try:
        if image is not None and image.filename:
            arr = decode_image(await image.read())
        elif seed is not None:
            rows = load_manifest(ROOT / "data" / "test")
            meta = rows[int(seed) % len(rows)]
            arr, _, _ = load_case(ROOT / "data" / "test", meta["case_id"])
        else:
            raise HTTPException(400, "Provide an image upload or a sample seed")
        if not note.strip():
            raise HTTPException(400, "note is required")
        return _infer(arr, note)
    except HTTPException:
        INFER_N.labels("error").inc()
        raise
    except Exception as exc:
        INFER_N.labels("error").inc()
        raise HTTPException(500, str(exc)) from exc


@app.get("/v1/eval")
def eval_cached():
    return STATE.get("metrics", {})


@app.post("/v1/eval/run")
def eval_run():
    cases = []
    for meta in load_manifest(ROOT / "data" / "test")[:16]:
        img, labels, m = load_case(ROOT / "data" / "test", meta["case_id"])
        rec = dict(m)
        rec["image"] = img
        rec["labels"] = labels
        cases.append(rec)
    result = run_harness(cases, STATE["mlp"], STATE["lora"])
    return result.as_dict()
