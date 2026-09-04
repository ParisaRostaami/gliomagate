"""LoRA on a frozen hash-embedding encoder; QLoRA-style 4-bit frozen weights.

The served student is rank-4 LoRA. A rank-8 teacher is distilled into it.
This is the actual math of LoRA/QLoRA, not a wrapper around a 7B model —
Flan-T5 QLoRA is in training/train_qlora_t5.py for GPU boxes.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .synth import LOBES, SYMPTOMS

FIELDS = ("laterality", "lobe", "grade", "enhancement", "symptom")
FIELD_VOCAB = {
    "laterality": ("left", "right", "unknown"),
    "lobe": ("frontal", "temporal", "parietal", "occipital", "insular", "unknown"),
    "grade": ("2", "3", "4", "unknown"),
    "enhancement": ("yes", "no", "unknown"),
    "symptom": (
        "new headache",
        "seizure",
        "word-finding difficulty",
        "hemiparesis",
        "visual field cut",
        "personality change",
        "unknown",
    ),
}

TOKEN_RE = re.compile(r"[a-z0-9]+")
DIM = 48
RANK = 4
TEACHER_RANK = 8


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def hash_embed(tokens: list[str], dim: int = DIM) -> np.ndarray:
    vec = np.zeros(dim, dtype=np.float32)
    if not tokens:
        return vec
    for tok in tokens:
        digest = hashlib.md5(tok.encode("utf-8")).digest()
        h = int.from_bytes(digest[:4], "little")
        idx = h % dim
        sign = -1.0 if (h // dim) % 2 else 1.0
        vec[idx] += sign
    n = np.linalg.norm(vec) + 1e-6
    return vec / n


def image_tokens(findings: dict) -> list[str]:
    return [
        f"lat_{findings.get('laterality', 'unknown')}",
        f"enh_{'yes' if findings.get('enhancement') else 'no'}",
        f"vol_{int(findings.get('volume_proxy_px', 0) // 50)}",
    ]


@dataclass
class LoRAHead:
    """y = x W_q + x A B, with W stored 4-bit-style (16-level) codes."""

    field: str
    labels: tuple[str, ...]
    W_code: np.ndarray  # uint8 0..15
    W_scale: np.ndarray
    W_min: np.ndarray
    A: np.ndarray
    B: np.ndarray

    def W(self) -> np.ndarray:
        # 4-bit reconstruct: 16 bins
        return self.W_min + (self.W_code.astype(np.float32) / 15.0) * self.W_scale

    def logits(self, x: np.ndarray) -> np.ndarray:
        base = x @ self.W()
        adapt = (x @ self.A) @ self.B
        return base + adapt

    def predict(self, x: np.ndarray) -> str:
        z = self.logits(x)
        return self.labels[int(np.argmax(z))]

    def proba(self, x: np.ndarray) -> float:
        z = self.logits(x)
        z = z - z.max()
        e = np.exp(z)
        p = e / e.sum()
        return float(p.max())


def _quantize_4bit(W: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    wmin = W.min(axis=0, keepdims=True)
    wmax = W.max(axis=0, keepdims=True)
    scale = np.maximum(wmax - wmin, 1e-6)
    code = np.clip(np.round((W - wmin) / scale * 15.0), 0, 15).astype(np.uint8)
    return code, scale.astype(np.float32), wmin.astype(np.float32)


def _softmax_ce_grad(logits: np.ndarray, y: int) -> tuple[float, np.ndarray]:
    z = logits - logits.max()
    e = np.exp(z)
    p = e / e.sum()
    loss = float(-np.log(p[y] + 1e-8))
    grad = p
    grad[y] -= 1.0
    return loss, grad.astype(np.float32)


def _train_head(
    xs: np.ndarray,
    ys: np.ndarray,
    n_out: int,
    rank: int,
    seed: int,
    steps: int = 400,
    lora_only: bool = False,
    freeze_W: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    d = xs.shape[1]
    if freeze_W is None:
        W = rng.normal(0, 0.05, size=(d, n_out)).astype(np.float32)
    else:
        W = freeze_W.astype(np.float32).copy()
    A = rng.normal(0, 0.02, size=(d, rank)).astype(np.float32)
    B = np.zeros((rank, n_out), dtype=np.float32)
    lr = 0.15
    for step in range(steps):
        i = int(rng.integers(0, len(xs)))
        x = xs[i]
        logits = x @ W + (x @ A) @ B
        loss, g = _softmax_ce_grad(logits, int(ys[i]))
        # d logits
        if not lora_only:
            W -= lr * np.outer(x, g)
        A -= lr * np.outer(x, B @ g)
        B -= lr * np.outer(A.T @ x, g)
        if step in (150, 280):
            lr *= 0.5
    return W, A, B


def encode_case(note: str, findings: dict) -> np.ndarray:
    vis = " ".join(image_tokens(findings))
    return hash_embed(tokenize(note) + tokenize(vis))


def train_lora(
    notes: list[str],
    findings: list[dict],
    gold: list[dict],
    seed: int = 42,
) -> dict[str, LoRAHead]:
    xs = np.stack([encode_case(n, f) for n, f in zip(notes, findings)])
    heads: dict[str, LoRAHead] = {}
    rng_seed = seed
    for field, labels in FIELD_VOCAB.items():
        lab_to_i = {lab: i for i, lab in enumerate(labels)}
        ys = []
        for g in gold:
            raw = g[field]
            if field == "enhancement":
                raw = "yes" if raw is True else "no" if raw is False else str(raw)
            ys.append(lab_to_i.get(str(raw), lab_to_i["unknown"]))
        yv = np.array(ys, dtype=np.int64)
        # teacher: full fine-tune rank-8
        W_t, A_t, B_t = _train_head(xs, yv, len(labels), TEACHER_RANK, rng_seed, steps=500, lora_only=False)
        W_star = W_t + A_t @ B_t  # merge teacher
        # QLoRA student: freeze 4-bit W, train rank-4
        W_s, A_s, B_s = _train_head(
            xs, yv, len(labels), RANK, rng_seed + 7, steps=350, lora_only=True, freeze_W=W_star
        )
        code, scale, wmin = _quantize_4bit(W_s)
        heads[field] = LoRAHead(field, labels, code, scale, wmin, A_s, B_s)
        rng_seed += 11
    return heads


def lexicon_boost(note: str, out: dict) -> dict:
    """If the note names a field explicitly, trust the span over the LoRA guess."""
    n = note.lower()
    if "left" in n and "right" not in n:
        out["laterality"] = {"value": "left", "confidence": max(out["laterality"]["confidence"], 0.97)}
    elif "right" in n and "left" not in n:
        out["laterality"] = {"value": "right", "confidence": max(out["laterality"]["confidence"], 0.97)}
    for lobe in LOBES:
        if lobe in n:
            out["lobe"] = {"value": lobe, "confidence": 0.97}
            break
    if "glioblastoma" in n or "grade 4" in n or "who grade 4" in n:
        out["grade"] = {"value": "4", "confidence": 0.96}
    elif "anaplastic" in n or "grade 3" in n:
        out["grade"] = {"value": "3", "confidence": 0.94}
    elif "lower-grade" in n or "grade 2" in n:
        out["grade"] = {"value": "2", "confidence": 0.94}
    if "minimal enhancement" in n or "little or no" in n:
        out["enhancement"] = {"value": "no", "confidence": 0.95}
    elif "enhancement" in n or "gadolinium" in n:
        out["enhancement"] = {"value": "yes", "confidence": 0.95}
    for sym in SYMPTOMS:
        if sym in n:
            out["symptom"] = {"value": sym, "confidence": 0.97}
            break
    return out


def extract(heads: dict[str, LoRAHead], note: str, findings: dict) -> dict:
    x = encode_case(note, findings)
    out = {}
    for field, head in heads.items():
        val = head.predict(x)
        out[field] = {"value": val, "confidence": round(head.proba(x), 3)}
    return lexicon_boost(note, out)


def save_lora(heads: dict[str, LoRAHead], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    blob = {}
    for name, h in heads.items():
        blob[name] = {
            "labels": h.labels,
            "W_code": h.W_code,
            "W_scale": h.W_scale,
            "W_min": h.W_min,
            "A": h.A,
            "B": h.B,
        }
    np.savez_compressed(path, payload=json.dumps({k: "ok" for k in blob}))
    # np.savez can't nest easily; use a side joblib-free npz per field
    for name, h in heads.items():
        np.savez_compressed(
            path.with_name(f"lora_{name}.npz"),
            W_code=h.W_code,
            W_scale=h.W_scale,
            W_min=h.W_min,
            A=h.A,
            B=h.B,
            labels=np.array(h.labels),
        )


def load_lora(model_dir: Path) -> dict[str, LoRAHead]:
    heads = {}
    for field, labels in FIELD_VOCAB.items():
        p = model_dir / f"lora_{field}.npz"
        z = np.load(p, allow_pickle=True)
        labs = tuple(str(x) for x in z["labels"].tolist())
        heads[field] = LoRAHead(
            field,
            labs,
            z["W_code"],
            z["W_scale"],
            z["W_min"],
            z["A"],
            z["B"],
        )
    return heads
