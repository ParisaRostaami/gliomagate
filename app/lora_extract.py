"""LoRA extractor on hash embeddings of the note and image findings.

The served student is rank-4 on a frozen 4-bit-style linear base. A rank-8
teacher is trained first. GPU Flan-T5 path: training/train_qlora_t5.py.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

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
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[float]]:
    rng = np.random.default_rng(seed)
    d = xs.shape[1]
    if freeze_W is None:
        W = rng.normal(0, 0.05, size=(d, n_out)).astype(np.float32)
    else:
        W = freeze_W.astype(np.float32).copy()
    A = rng.normal(0, 0.02, size=(d, rank)).astype(np.float32)
    B = np.zeros((rank, n_out), dtype=np.float32)
    lr = 0.15
    losses: list[float] = []
    for step in range(steps):
        i = int(rng.integers(0, len(xs)))
        x = xs[i]
        logits = x @ W + (x @ A) @ B
        loss, g = _softmax_ce_grad(logits, int(ys[i]))
        if not lora_only:
            W -= lr * np.outer(x, g)
        A -= lr * np.outer(x, B @ g)
        B -= lr * np.outer(A.T @ x, g)
        if step % 25 == 0 or step == steps - 1:
            losses.append(float(loss))
        if step in (150, 280):
            lr *= 0.5
    return W, A, B, losses


def encode_case(note: str, findings: dict) -> np.ndarray:
    vis = " ".join(image_tokens(findings))
    return hash_embed(tokenize(note) + tokenize(vis))


def _gold_label(g: dict, field: str) -> str:
    raw = g[field]
    if field == "enhancement":
        return "yes" if raw is True else "no" if raw is False else str(raw)
    return str(raw)


def field_accuracy(heads: dict[str, LoRAHead], notes: list[str], findings: list[dict], gold: list[dict]) -> dict[str, float]:
    acc: dict[str, float] = {}
    for field in FIELD_VOCAB:
        hits = 0
        for note, find, g in zip(notes, findings, gold):
            pred = extract(heads, note, find)[field]["value"]
            hits += int(pred == _gold_label(g, field))
        acc[field] = hits / max(len(gold), 1)
    return acc


def train_lora(
    notes: list[str],
    findings: list[dict],
    gold: list[dict],
    seed: int = 42,
) -> dict[str, LoRAHead]:
    xs = np.stack([encode_case(n, f) for n, f in zip(notes, findings)])
    heads: dict[str, LoRAHead] = {}
    rng_seed = seed
    loss_curves: dict[str, list[float]] = {}
    for field, labels in FIELD_VOCAB.items():
        lab_to_i = {lab: i for i, lab in enumerate(labels)}
        yv = np.array(
            [lab_to_i.get(_gold_label(g, field), lab_to_i["unknown"]) for g in gold],
            dtype=np.int64,
        )
        W_t, A_t, B_t, _ = _train_head(xs, yv, len(labels), TEACHER_RANK, rng_seed, steps=500, lora_only=False)
        W_star = W_t + A_t @ B_t
        W_s, A_s, B_s, student_loss = _train_head(
            xs, yv, len(labels), RANK, rng_seed + 7, steps=350, lora_only=True, freeze_W=W_star
        )
        code, scale, wmin = _quantize_4bit(W_s)
        heads[field] = LoRAHead(field, labels, code, scale, wmin, A_s, B_s)
        loss_curves[field] = student_loss
        rng_seed += 11
        print(f"  LoRA {field}: student CE {student_loss[0]:.3f} -> {student_loss[-1]:.3f}")
    train_lora.last_loss_curves = loss_curves  # type: ignore[attr-defined]
    acc = field_accuracy(heads, notes, findings, gold)
    train_lora.last_train_acc = acc  # type: ignore[attr-defined]
    print("  LoRA train field acc:", {k: round(v, 3) for k, v in acc.items()})
    return heads


def extract(heads: dict[str, LoRAHead], note: str, findings: dict) -> dict:
    x = encode_case(note, findings)
    out = {}
    for field, head in heads.items():
        val = head.predict(x)
        out[field] = {"value": val, "confidence": round(head.proba(x), 3)}
    return out


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
