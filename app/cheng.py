"""Cheng 2017 T1-contrast brain tumor MRI (Figshare 1512427).

3064 T1c slices from 233 patients: meningioma, glioma, pituitary.
We use glioma slices only, split by patient id.
"""

from __future__ import annotations

import zipfile
from pathlib import Path
from urllib.request import urlretrieve

import h5py
import numpy as np
from PIL import Image

from .synth import SIZE, write_note

LABEL_NAME = {1: "meningioma", 2: "glioma", 3: "pituitary"}
FIGSHARE_FILES = (
    ("brainTumorDataPublic_1-766.zip", "https://ndownloader.figshare.com/files/3381290"),
    ("brainTumorDataPublic_767-1532.zip", "https://ndownloader.figshare.com/files/3381296"),
    ("brainTumorDataPublic_1533-2298.zip", "https://ndownloader.figshare.com/files/3381293"),
    ("brainTumorDataPublic_2299-3064.zip", "https://ndownloader.figshare.com/files/3381302"),
)


def download_cheng(raw_dir: Path) -> Path:
    raw_dir.mkdir(parents=True, exist_ok=True)
    mats_dir = raw_dir / "mats"
    mats = list(raw_dir.rglob("*.mat"))
    if len(mats) >= 3000:
        return raw_dir
    print("downloading Cheng 2017 T1c MRI from Figshare (4 archives, ~880 MB)...")
    bad = raw_dir / "cheng_1512427.zip"
    if bad.exists() and bad.stat().st_size < 1_000_000:
        bad.unlink()
    for name, url in FIGSHARE_FILES:
        dest = raw_dir / name
        if dest.exists() and dest.stat().st_size > 1_000_000:
            print(f"  have {name} ({dest.stat().st_size} bytes)")
        else:
            print(f"  fetching {name}")
            urlretrieve(url, dest)
        with zipfile.ZipFile(dest) as zf:
            zf.extractall(mats_dir)
    n = len(list(mats_dir.rglob("*.mat")))
    print(f"extracted {n} .mat slices")
    return raw_dir


def _decode_pid(raw) -> str:
    arr = np.array(raw).reshape(-1)
    if arr.dtype.kind in "SU":
        return "".join(arr.astype(str).tolist())
    chars = []
    for x in arr:
        xi = int(x)
        if 32 <= xi < 127:
            chars.append(chr(xi))
    return "".join(chars) or "unknown"


def _load_mat(path: Path) -> dict:
    with h5py.File(path, "r") as f:
        g = f["cjdata"] if "cjdata" in f else f
        image = np.array(g["image"]).T
        mask = np.array(g["tumorMask"]).T
        lab = int(np.array(g["label"]).reshape(-1)[0])
        pid = _decode_pid(g["PID"]) if "PID" in g else path.stem
    image = image.astype(np.float32)
    image = image / (float(image.max()) + 1e-6)
    mask = (mask > 0).astype(np.uint8)
    return {"image": image, "labels": mask, "tumor_type": LABEL_NAME.get(lab, "unknown"), "pid": pid, "label_id": lab}


def _resize(img: np.ndarray, size: int, nearest: bool) -> np.ndarray:
    mode = Image.NEAREST if nearest else Image.BILINEAR
    if img.ndim == 2 and nearest:
        im = Image.fromarray((img > 0).astype(np.uint8) * 255, mode="L")
        out = np.array(im.resize((size, size), mode), dtype=np.uint8)
        return (out > 127).astype(np.uint8)
    im = Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8), mode="L")
    out = np.array(im.resize((size, size), mode), dtype=np.float32) / 255.0
    return out


def iter_glioma(raw_dir: Path):
    mats = sorted(p for p in raw_dir.rglob("*.mat") if p.name != "cvind.mat")
    for p in mats:
        rec = _load_mat(p)
        if rec["label_id"] != 2:
            continue
        rec["image"] = _resize(rec["image"], SIZE, nearest=False)
        rec["labels"] = _resize(rec["labels"], SIZE, nearest=True)
        rec["source"] = p.name
        yield rec


def laterality_from_mask(mask: np.ndarray) -> str:
    ys, xs = np.nonzero(mask > 0)
    if len(xs) < 8:
        return "unknown"
    return "left" if float(xs.mean()) < (SIZE / 2) else "right"


def as_case(rec: dict, case_id: str, rng: np.random.Generator) -> dict:
    lat = laterality_from_mask(rec["labels"])
    row = {
        "case_id": case_id,
        "pid": rec["pid"],
        "laterality": lat,
        "lobe": "unknown",
        "grade": "unknown",
        "enhancement": True,
        "symptom": "unknown",
        "tumor_type": rec["tumor_type"],
        "image": rec["image"],
        "labels": rec["labels"],
        "core_pixels": 0,
        "edema_pixels": int((rec["labels"] > 0).sum()),
    }
    row["note"] = write_note(row, rng)
    return row


def build_splits(
    raw_dir: Path,
    n_train: int = 80,
    n_test: int = 32,
    seed: int = 7,
) -> tuple[list[dict], list[dict]]:
    rng = np.random.default_rng(seed)
    by_pid: dict[str, list[dict]] = {}
    for rec in iter_glioma(raw_dir):
        by_pid.setdefault(rec["pid"], []).append(rec)
    pids = list(by_pid)
    rng.shuffle(pids)
    train_recs: list[dict] = []
    test_recs: list[dict] = []
    for i, pid in enumerate(pids):
        bucket = test_recs if i % 4 == 0 else train_recs
        bucket.extend(by_pid[pid])
    rng.shuffle(train_recs)
    rng.shuffle(test_recs)
    train = [as_case(r, f"tr-{i:04d}", rng) for i, r in enumerate(train_recs[:n_train])]
    test = [as_case(r, f"te-{i:04d}", rng) for i, r in enumerate(test_recs[:n_test])]
    return train, test
