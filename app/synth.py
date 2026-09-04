"""Synthetic T1c-like slices and paired neuro-oncology notes.

Ellipsoidal enhancing core + edema halo, bias field, and noise.
Public stand-in for BraTS-style data (labels and notes are generated).
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


def write_note(case: dict, rng: np.random.Generator | None = None) -> str:
    rng = rng or np.random.default_rng(0)
    lat, lobe, grade, symptom = case["laterality"], case["lobe"], case["grade"], case["symptom"]
    if lobe == "unknown" or grade == "unknown":
        enhance = "T1-weighted post-contrast enhancement" if case.get("enhancement", True) else "little contrast uptake"
        return str(
            rng.choice(
                (
                    f"T1-post-contrast MRI. Enhancing mass {lat} of midline, compatible with glioma. {enhance}. Histologic grade not assigned on this slice.",
                    f"Neuro-oncology. {lat.capitalize()}-sided enhancing glioma on T1c. Grade, lobe, and presenting symptom are not specified in the record.",
                    f"Axial T1c. There is a {lat}-sided enhancing lesion consistent with glioma. WHO grade unknown from imaging alone.",
                )
            )
        )
    if case["enhancement"]:
        enhance = rng.choice(
            (
                "avid gadolinium uptake with a necrotic center",
                "ring enhancement and central necrosis",
                "strong contrast enhancement",
            )
        )
    else:
        enhance = rng.choice(
            (
                "little contrast uptake",
                "faint enhancement only",
                "essentially non-enhancing tissue",
            )
        )
    grade_talk = {
        "2": rng.choice(
            (
                "Imaging favors a lower-grade glioma.",
                "Findings are more in keeping with WHO grade 2 disease.",
                "Mitotic activity is expected to be modest (grade 2).",
            )
        ),
        "3": rng.choice(
            (
                "This raises concern for anaplastic (WHO grade 3) histology.",
                "I would treat this as a grade 3 glioma pending pathology.",
                "Features are compatible with anaplastic glioma.",
            )
        ),
        "4": rng.choice(
            (
                "The appearance is most consistent with glioblastoma, WHO grade 4.",
                "This is a high-grade pattern, likely GBM.",
                "I favor IDH-wildtype glioblastoma (grade 4).",
            )
        ),
    }[grade]
    opener = rng.choice(
        (
            f"Consult note. Presentation: {symptom}.",
            f"Neuro-oncology. Chief complaint is {symptom}.",
            f"{symptom.capitalize()} led to this MRI.",
        )
    )
    location = rng.choice(
        (
            f"There is a {lat}-sided {lobe} mass.",
            f"MRI shows a {lobe} lesion on the {lat}.",
            f"A {lobe} mass is present {lat} of midline.",
        )
    )
    close = rng.choice(
        (
            "Please discuss at tumor board. IDH and MGMT pending.",
            "Recommend molecular testing (IDH, MGMT) and tumor-board review.",
            "Plan: resection vs biopsy after multidisciplinary review.",
        )
    )
    return f"{opener} {location} The lesion shows {enhance}. {grade_talk} {close}"


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
        rec["note"] = write_note(rec, rng)
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
