"""Train segmenter + LoRA extractor, run harness, write models/metrics.json."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.eval_harness import run_harness
from app.lora_extract import save_lora, train_lora
from app.segment import image_findings, predict_mask_mlp, save_segmenter, train_segmenter
from app.synth import generate_split, save_cases


def main() -> None:
    data = ROOT / "data"
    models = ROOT / "models"
    train = generate_split(80, seed=42, prefix="tr")
    test = generate_split(32, seed=7, prefix="te")
    save_cases(train, data / "train")
    save_cases(test, data / "test")

    mlp, head = train_segmenter([c["image"] for c in train], [c["labels"] for c in train], seed=42)
    save_segmenter(mlp, head, models / "segmenter.joblib")

    findings = [image_findings(predict_mask_mlp(mlp, c["image"])) for c in train]
    heads = train_lora([c["note"] for c in train], findings, train, seed=42)
    save_lora(heads, models / "lora.npz")

    result = run_harness(test, mlp, heads)
    metrics = result.as_dict()
    models.mkdir(parents=True, exist_ok=True)
    (models / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
