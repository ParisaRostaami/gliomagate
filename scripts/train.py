"""Train the segmenter and LoRA extractor on Cheng 2017 T1c glioma slices."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.cheng import build_splits, download_cheng
from app.eval_harness import run_harness
from app.lora_extract import field_accuracy, save_lora, train_lora
from app.segment import image_findings, mean_tumor_dice, predict_mask_mlp, save_segmenter, train_segmenter
from app.synth import save_cases


def main() -> None:
    data = ROOT / "data"
    models = ROOT / "models"
    models.mkdir(parents=True, exist_ok=True)

    print("loading Cheng 2017 glioma T1c slices (patient-wise split)...")
    raw = download_cheng(data / "cheng_raw")
    train, test = build_splits(raw, n_train=240, n_test=40, seed=7)
    save_cases(train, data / "train")
    save_cases(test, data / "test")
    print(f"train={len(train)} test={len(test)} patients in test: {len({c['pid'] for c in test})}")

    print("fitting pixel MLP on conv-stem features...")
    mlp, head = train_segmenter([c["image"] for c in train], [c["labels"] for c in train], seed=42)
    train_dices = [mean_tumor_dice(predict_mask_mlp(mlp, c["image"]), c["labels"]) for c in train]
    train_dice = float(sum(train_dices) / len(train_dices))
    print(
        f"segmenter: n_iter={mlp.n_iter_}  final CE={mlp.loss_:.4f}  "
        f"train Dice={train_dice:.3f} (n={len(train)})"
    )
    save_segmenter(mlp, head, models / "segmenter.joblib")

    print("training LoRA extractor (rank-8 teacher -> rank-4 student)...")
    findings = [image_findings(predict_mask_mlp(mlp, c["image"])) for c in train]
    heads = train_lora([c["note"] for c in train], findings, train, seed=42)
    save_lora(heads, models / "lora.npz")

    test_findings = [image_findings(predict_mask_mlp(mlp, c["image"])) for c in test]
    test_acc = field_accuracy(heads, [c["note"] for c in test], test_findings, test)
    print("LoRA test field acc:", {k: round(v, 3) for k, v in test_acc.items()})

    print("evaluating held-out patients...")
    result = run_harness(test, mlp, heads)
    metrics = result.as_dict()
    metrics["dataset"] = "Cheng2017 T1c glioma (Figshare 1512427)"
    metrics["split"] = "patient-wise"
    (models / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    train_log = {
        "dataset": "Cheng2017 T1c glioma",
        "segmenter": {
            "n_iter": int(mlp.n_iter_),
            "final_loss": float(mlp.loss_),
            "loss_curve": [float(x) for x in mlp.loss_curve_],
            "train_dice": train_dice,
            "n_train": len(train),
        },
        "lora": {
            "train_field_acc": getattr(train_lora, "last_train_acc", {}),
            "test_field_acc": test_acc,
            "loss_curves": getattr(train_lora, "last_loss_curves", {}),
        },
        "test_harness": metrics,
    }
    (models / "train_log.json").write_text(json.dumps(train_log, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print("wrote", models / "train_log.json")


if __name__ == "__main__":
    main()
