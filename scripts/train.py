"""Train the segmenter and LoRA extractor, then write models/metrics.json."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.eval_harness import run_harness
from app.lora_extract import field_accuracy, save_lora, train_lora
from app.segment import image_findings, mean_tumor_dice, predict_mask_mlp, save_segmenter, train_segmenter
from app.synth import generate_split, save_cases


def main() -> None:
    data = ROOT / "data"
    models = ROOT / "models"
    models.mkdir(parents=True, exist_ok=True)

    print("generating 80 train / 32 test synthetic slices...")
    train = generate_split(80, seed=42, prefix="tr")
    test = generate_split(32, seed=7, prefix="te")
    save_cases(train, data / "train")
    save_cases(test, data / "test")

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

    print("running held-out harness...")
    result = run_harness(test, mlp, heads)
    metrics = result.as_dict()
    (models / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    train_log = {
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
