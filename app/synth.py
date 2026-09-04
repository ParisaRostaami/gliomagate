"""Synthetic T1c/FLAIR-like slices and paired neuro-oncology notes.

Ellipsoidal enhancing core + edema halo, bias field, and Rician-ish noise.
Not BraTS. Labels and notes are generated so every metric is reproducible.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter

SIZE = 128
LOBES = ("frontal", "temporal", "parietal", "occipital", "insular")
GRADES = ("2", "3", "4")
SYMPTOMS = (
    "new headache",
    "seizure",
    "word-finding difficulty",
    "hemiparesis",
    "visual field cut",
    "personality change",
)


@dataclass(frozen=True)
class Case:
    case_id: str
    laterality: str
    lobe: str
    grade: str
    enhancement: bool
    symptom: str
    core_pixels: int
    edema_pixels: int
    note: str


def _ellipse_mask(h: int, w: int, cy: float, cx: float, ry: float, rx: float) -> np.ndarray:
    yy, xx = np.ogrid[:h, :w]
    return ((yy - cy) ** 2) / (ry ** 2 + 1e-6) + ((xx - cx) ** 2) / (rx ** 2 + 1e-6) <= 1.0


def _bias_field(h: int, w: int, rng: np.random.Generator) -> np.ndarray:
    field = rng.normal(0.0, 1.0, size=(8, 8))
    field = gaussian_filter(field, sigma=1.2)
    field = np.kron(field, np.ones((h // 8, w // 8)))[:h, :w]
    field = (field - field.min()) / (np.ptp(field) + 1e-6)
    return 0.75 + 0.5 * field


def render_slice(rng: np.random.Generator, laterality: str, lobe: str, enhancement: bool) -> tuple[np.ndarray, np.ndarray]:
    """Return float image in [0, 1] and label map {0 bg, 1 edema, 2 core}."""
    h = w = SIZE
    img = rng.normal(0.28, 0.04, size=(h, w))
    # crude skull / CSF ring
    yy, xx = np.ogrid[:h, :w]
    r = np.sqrt((yy - 63.5) ** 2 + (xx - 63.5) ** 2)
    img = np.where(r > 58, 0.05, img)
    img = np.where((r > 52) & (r <= 58), 0.55, img)

    lobe_prior = {
        "frontal": (38, None),
        "parietal": (46, None),
        "temporal": (72, None),
        "occipital": (88, None),
        "insular": (60, None),
    }
    cy, _ = lobe_prior[lobe]
    cx = 40.0 if laterality == "left" else 88.0
    cy = float(np.clip(cy + rng.normal(0, 4), 28, 100))
    cx = float(np.clip(cx + rng.normal(0, 3), 22, 106))
    core_rx = rng.uniform(6.5, 11.5)
    core_ry = rng.uniform(6.0, 11.0)
    edema_rx = core_rx * rng.uniform(1.7, 2.4)
    edema_ry = core_ry * rng.uniform(1.7, 2.4)
    edema = _ellipse_mask(h, w, cy, cx, edema_ry, edema_rx) & (r < 52)
    core = _ellipse_mask(h, w, cy, cx, core_ry, core_rx) & edema
    labels = np.zeros((h, w), dtype=np.uint8)
    labels[edema] = 1
    labels[core] = 2

    img = img.copy()
    img[edema] = rng.uniform(0.42, 0.55)
    if enhancement:
        img[core] = rng.uniform(0.78, 0.95)
        # necrotic dip
        inner = _ellipse_mask(h, w, cy, cx, core_ry * 0.45, core_rx * 0.45)
        img[inner] = rng.uniform(0.22, 0.34)
    else:
        img[core] = rng.uniform(0.48, 0.62)

    img = img * _bias_field(h, w, rng)
    noise = rng.normal(0, 0.025, size=img.shape)
    img = np.clip(img + noise, 0, 1).astype(np.float32)
    return img, labels


def write_note(case: dict) -> str:
    enhance = "vivid gadolinium enhancement with central necrosis" if case["enhancement"] else "minimal enhancement"
    grade_talk = {
        "2": "favors a lower-grade glioma; mitotic activity is expected to be modest",
        "3": "raises concern for anaplastic (WHO grade 3) histology",
        "4": "is most consistent with glioblastoma, IDH-wildtype, WHO grade 4",
    }[case["grade"]]
    return (
        f"Neuro-oncology consult. {case['symptom'].capitalize()} prompted MRI. "
        f"There is a {case['laterality']}-sided {case['lobe']} mass. "
        f"The lesion shows {enhance}. Imaging {grade_talk}. "
        f"I recommend discussion at tumor board and molecular testing (IDH, MGMT) "
        f"per institutional glioma protocol."
    )


def generate_split(n: int, seed: int, prefix: str) -> list[dict]:
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    for i in range(n):
        laterality = str(rng.choice(("left", "right")))
        lobe = str(rng.choice(LOBES))
        grade = str(rng.choice(GRADES, p=(0.25, 0.25, 0.50)))
        enhancement = bool(grade == "4" or rng.random() < 0.25)
        symptom = str(rng.choice(SYMPTOMS))
        img, labels = render_slice(rng, laterality, lobe, enhancement)
        rec = {
            "case_id": f"{prefix}-{i:04d}",
            "laterality": laterality,
            "lobe": lobe,
            "grade": grade,
            "enhancement": enhancement,
            "symptom": symptom,
            "core_pixels": int((labels == 2).sum()),
            "edema_pixels": int((labels == 1).sum()),
            "image": img,
            "labels": labels,
        }
        rec["note"] = write_note(rec)
        rows.append(rec)
    return rows


def save_cases(rows: list[dict], path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    meta = []
    for rec in rows:
        np.savez_compressed(
            path / f"{rec['case_id']}.npz",
            image=rec["image"],
            labels=rec["labels"],
        )
        meta.append({k: v for k, v in rec.items() if k not in {"image", "labels"}})
    (path / "manifest.jsonl").write_text(
        "\n".join(json.dumps(m) for m in meta) + "\n",
        encoding="utf-8",
    )


def load_manifest(path: Path) -> list[dict]:
    lines = (path / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def load_case(path: Path, case_id: str) -> tuple[np.ndarray, np.ndarray, dict]:
    blob = np.load(path / f"{case_id}.npz")
    meta = next(m for m in load_manifest(path) if m["case_id"] == case_id)
    return blob["image"], blob["labels"], meta
