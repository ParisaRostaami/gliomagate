"""Download Cheng 2017 T1c MRI and write train/test npz splits."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.cheng import build_splits, download_cheng
from app.synth import save_cases


def main() -> None:
    raw = download_cheng(ROOT / "data" / "cheng_raw")
    train, test = build_splits(raw, n_train=80, n_test=32, seed=7)
    save_cases(train, ROOT / "data" / "train")
    save_cases(test, ROOT / "data" / "test")
    print(f"wrote {len(train)} train and {len(test)} test real T1c glioma slices")
    print("example note:", test[0]["note"][:180])
    print("test laterality:", [c["laterality"] for c in test[:8]])


if __name__ == "__main__":
    main()
