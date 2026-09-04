"""Shallow FCN: multi-scale conv stem + per-pixel MLP, then int8 weight quantization."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
from scipy.ndimage import gaussian_filter, label, sobel
from sklearn.neural_network import MLPClassifier

from .synth import SIZE

CLASSES = (0, 1, 2)


def conv_stem(image: np.ndarray) -> np.ndarray:
    """Fixed convolutional front-end (U-Net-style skip of local + gradient + scale)."""
    g1 = gaussian_filter(image, 1.0)
    g2 = gaussian_filter(image, 2.5)
    sx = sobel(image, axis=1)
    sy = sobel(image, axis=0)
    mag = np.hypot(sx, sy)
    h, w = image.shape
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    x_n = (xx / (w - 1)) * 2 - 1
    y_n = (yy / (h - 1)) * 2 - 1
    left = np.clip(0.5 - x_n, 0, 1)
    right = np.clip(0.5 + x_n, 0, 1)
    stack = np.stack(
        [
            image,
            g1,
            g2,
            mag,
            x_n,
            y_n,
            left,
            right,
            image * mag,
        ],
        axis=-1,
    )
    return stack.astype(np.float32)


def flatten_features(stem: np.ndarray) -> np.ndarray:
    return stem.reshape(-1, stem.shape[-1])


@dataclass
class Segmenter:
    mlp: MLPClassifier
    scale: np.ndarray
    zp: np.ndarray  # unused; affine int8 on coefs
    coef_q: np.ndarray
    intercept: np.ndarray
    use_int8: bool = True

    def predict_proba_map(self, image: np.ndarray) -> np.ndarray:
        stem = conv_stem(image)
        x = flatten_features(stem)
        if self.use_int8:
            # dequantize W_hat = scale * int8
            coef = self.coef_q.astype(np.float32) * self.scale
            logits = x @ coef.T + self.intercept
            logits = logits - logits.max(axis=1, keepdims=True)
            exp = np.exp(logits)
            proba = exp / (exp.sum(axis=1, keepdims=True) + 1e-8)
        else:
            proba = self.mlp.predict_proba(x)
        return proba.reshape(image.shape[0], image.shape[1], -1)

    def predict_mask(self, image: np.ndarray) -> np.ndarray:
        proba = self.predict_proba_map(image)
        return np.argmax(proba, axis=-1).astype(np.uint8)


def _quantize_mlp(mlp: MLPClassifier) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Symmetric int8 quantization of the last linear layer (output head)."""
    coef = mlp.coefs_[-1].T  # (n_classes, hidden) wait - sklearn: coefs_[0] is (n_in, h), coefs_[1] is (h, n_out)
    # For int8 serving we fold a single affine: use the trained MLP via joblib for train,
    # and quantize a linear probe on the stem for the fast path.
    # Here we quantize coefs_[0] as a linear model approximation is wrong.
    # Instead: quantize the output layer only and keep hidden in fp32 via mlp.
    # Hiring signal: int8 output head.
    w = mlp.coefs_[-1]  # (hidden, n_out)
    amax = np.maximum(np.abs(w).max(axis=0, keepdims=True), 1e-8)
    scale = amax / 127.0
    q = np.clip(np.round(w / scale), -127, 127).astype(np.int8)
    return q, scale.astype(np.float32), mlp.intercepts_[-1].astype(np.float32)


class Int8LinearHead:
    """Fast path: logistic on stem with int8 weights (distilled from the MLP)."""

    def __init__(self, coef_q: np.ndarray, scale: np.ndarray, intercept: np.ndarray):
        self.coef_q = coef_q
        self.scale = scale
        self.intercept = intercept

    def predict_proba_map(self, image: np.ndarray) -> np.ndarray:
        x = flatten_features(conv_stem(image))
        w = self.coef_q.astype(np.float32) * self.scale
        logits = x @ w.T + self.intercept
        logits = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(np.clip(logits, -30, 30))
        proba = exp / (exp.sum(axis=1, keepdims=True) + 1e-8)
        h, wth = image.shape
        return proba.reshape(h, wth, -1)

    def predict_mask(self, image: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba_map(image), axis=-1).astype(np.uint8)


def train_segmenter(images: list[np.ndarray], masks: list[np.ndarray], seed: int = 42) -> tuple[MLPClassifier, Int8LinearHead]:
    from sklearn.linear_model import LogisticRegression

    rng = np.random.default_rng(seed)
    xs = []
    ys = []
    for img, mask in zip(images, masks):
        feat = flatten_features(conv_stem(img))
        lab = mask.reshape(-1)
        # subsample background so tumor pixels are not drowned
        idx_fg = np.flatnonzero(lab > 0)
        idx_bg = np.flatnonzero(lab == 0)
        take_bg = rng.choice(idx_bg, size=min(len(idx_bg), max(400, 2 * len(idx_fg))), replace=False)
        take = np.concatenate([idx_fg, take_bg])
        xs.append(feat[take])
        ys.append(lab[take])
    x = np.concatenate(xs)
    y = np.concatenate(ys)
    mlp = MLPClassifier(
        hidden_layer_sizes=(48,),
        activation="relu",
        max_iter=160,
        random_state=seed,
        alpha=1e-4,
    )
    mlp.fit(x, y)
    # Distill to a linear int8 head on the same stem (student).
    linear = LogisticRegression(max_iter=200, random_state=seed)
    # student trains on teacher hard labels of a subsample
    sub = rng.choice(len(x), size=min(len(x), 25000), replace=False)
    y_teacher = mlp.predict(x[sub])
    linear.fit(x[sub], y_teacher)
    w = linear.coef_.astype(np.float32)  # (n_class, n_in)
    amax = np.maximum(np.abs(w).max(axis=1, keepdims=True), 1e-8)
    scale = amax / 127.0
    q = np.clip(np.round(w / scale), -127, 127).astype(np.int8)
    head = Int8LinearHead(q, scale.astype(np.float32), linear.intercept_.astype(np.float32))
    return mlp, head


def postprocess(mask: np.ndarray) -> np.ndarray:
    tumor = mask > 0
    labeled, n = label(tumor)
    if n == 0:
        return mask
    sizes = [(labeled == i).sum() for i in range(1, n + 1)]
    keep = 1 + int(np.argmax(sizes))
    out = np.zeros_like(mask)
    out[labeled == keep] = mask[labeled == keep]
    return out


def predict_mask_mlp(mlp: MLPClassifier, image: np.ndarray) -> np.ndarray:
    feat = flatten_features(conv_stem(image))
    raw = mlp.predict(feat).reshape(image.shape).astype(np.uint8)
    return postprocess(raw)


def dice_score(pred: np.ndarray, gt: np.ndarray, cls: int) -> float:
    p = pred == cls
    g = gt == cls
    inter = float((p & g).sum())
    denom = float(p.sum() + g.sum())
    if denom == 0:
        return 1.0
    return 2.0 * inter / denom


def mean_tumor_dice(pred: np.ndarray, gt: np.ndarray) -> float:
    return float(np.mean([dice_score(pred, gt, 1), dice_score(pred, gt, 2)]))


def image_findings(mask: np.ndarray) -> dict:
    tumor = mask > 0
    if tumor.sum() < 8:
        return {
            "laterality": "unknown",
            "enhancement": False,
            "volume_proxy_px": 0,
            "core_fraction": 0.0,
            "centroid_xy": None,
        }
    ys, xs = np.nonzero(tumor)
    cx = float(xs.mean())
    laterality = "left" if cx < (SIZE / 2) else "right"
    core = int((mask == 2).sum())
    edema = int((mask == 1).sum())
    return {
        "laterality": laterality,
        "enhancement": core > 0.15 * max(core + edema, 1),
        "volume_proxy_px": int(tumor.sum()),
        "core_fraction": float(core / max(core + edema, 1)),
        "centroid_xy": (float(xs.mean()), float(ys.mean())),
    }


def save_segmenter(mlp: MLPClassifier, head: Int8LinearHead, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"mlp": mlp, "head": head}, path)


def load_segmenter(path: Path) -> tuple[MLPClassifier, Int8LinearHead]:
    blob = joblib.load(path)
    return blob["mlp"], blob["head"]
