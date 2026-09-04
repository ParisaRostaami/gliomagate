"""Render gallery, metric charts, ledger example, and latency histogram."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.eval_harness import run_harness
from app.ground import assemble_ledger
from app.lora_extract import extract, load_lora
from app.main import overlay_rgb
from app.rag import GuidelineIndex
from app.segment import image_findings, load_segmenter, mean_tumor_dice, predict_mask_mlp
from app.synth import generate_split, load_case, load_manifest, save_cases

OUT = ROOT / "docs" / "figures"
MODELS = ROOT / "models"
TEST = ROOT / "data" / "test"


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#223044",
            "axes.labelcolor": "#223044",
            "text.color": "#223044",
            "font.size": 11,
            "axes.titlesize": 13,
            "figure.dpi": 140,
        }
    )


def load_test_cases(n: int | None = None) -> list[dict]:
    if not (TEST / "manifest.jsonl").exists():
        rows = generate_split(32, seed=7, prefix="te")
        save_cases(rows, TEST)
    rows = []
    for meta in load_manifest(TEST):
        img, labels, m = load_case(TEST, meta["case_id"])
        rec = dict(m)
        rec["image"] = img
        rec["labels"] = labels
        rows.append(rec)
        if n and len(rows) >= n:
            break
    return rows


def gallery(mlp, cases: list[dict], path: Path) -> None:
    pick = cases[:6]
    fig, axes = plt.subplots(3, 6, figsize=(14.5, 7.4))
    for j, rec in enumerate(pick):
        pred = predict_mask_mlp(mlp, rec["image"])
        dice = mean_tumor_dice(pred, rec["labels"])
        axes[0, j].imshow(rec["image"], cmap="gray")
        axes[0, j].set_title(rec["case_id"], fontsize=9)
        axes[1, j].imshow(rec["labels"], cmap="viridis", vmin=0, vmax=2)
        axes[2, j].imshow(overlay_rgb(rec["image"], pred))
        axes[2, j].set_xlabel(f"Dice {dice:.2f}", fontsize=9)
        for i in range(3):
            axes[i, j].axis("off")
    axes[0, 0].set_ylabel("slice", fontsize=10)
    axes[1, 0].set_ylabel("gt mask", fontsize=10)
    axes[2, 0].set_ylabel("pred overlay", fontsize=10)
    fig.suptitle("Held-out slices — ground truth vs GliomaGate overlay (teal edema, red core)", y=0.98)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def dice_hist(mlp, cases: list[dict], path: Path) -> list[float]:
    dices = [mean_tumor_dice(predict_mask_mlp(mlp, c["image"]), c["labels"]) for c in cases]
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.hist(dices, bins=10, color="#355070", edgecolor="white")
    ax.axvline(float(np.mean(dices)), color="#e76f51", lw=2, label=f"mean {np.mean(dices):.2f}")
    ax.set_xlabel("mean tumor Dice (edema + core)")
    ax.set_ylabel("cases")
    ax.set_title("Segmentation Dice on 32 held-out synthetic slices")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return dices


def metric_bars(metrics: dict, path: Path) -> None:
    labels = [
        "Dice",
        "nDCG@5",
        "Recall@5",
        "Grounded",
        "Grounded\n(no RAG)",
        "Field acc\n(templated)",
    ]
    vals = [
        metrics["dice"],
        metrics["ndcg5"],
        metrics["recall5"],
        metrics["grounded_fraction"],
        metrics["ablation"]["no_rag_grounded_fraction"],
        metrics["field_acc_mean"],
    ]
    colors = ["#355070", "#4a7c9b", "#4a7c9b", "#2a9d8f", "#e9c46a", "#6c757d"]
    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    bars = ax.bar(labels, vals, color=colors)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("score")
    ax.set_title("Held-out harness (n=32)")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.2f}", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def ablation_chart(metrics: dict, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.6, 3.8))
    names = ["Full system\ngrounded fraction", "No RAG\ngrounded fraction"]
    vals = [metrics["grounded_fraction"], metrics["ablation"]["no_rag_grounded_fraction"]]
    ax.bar(names, vals, color=["#2a9d8f", "#e76f51"], width=0.55)
    ax.set_ylim(0, 1.05)
    ax.set_title("Ablation: retrieval is what grounds the report")
    for i, v in enumerate(vals):
        ax.text(i, v + 0.03, f"{v:.2f}", ha="center")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def latency_chart(path: Path) -> dict:
    mlp, _ = load_segmenter(MODELS / "segmenter.joblib")
    lora = load_lora(MODELS)
    index = GuidelineIndex()
    cases = load_test_cases()
    times = []
    for rec in cases * 3:
        t0 = time.perf_counter()
        pred = predict_mask_mlp(mlp, rec["image"])
        findings = image_findings(pred)
        extracted = extract(lora, rec["note"], findings)
        hits = index.search(index.query_for_case(rec["note"], findings, extracted), k=4)
        assemble_ledger(extracted, findings, hits)
        times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    p50 = times[len(times) // 2]
    p95 = times[int(0.95 * (len(times) - 1))]
    fig, ax = plt.subplots(figsize=(6.6, 3.8))
    ax.hist(times, bins=16, color="#355070", edgecolor="white")
    ax.axvline(p50, color="#2a9d8f", lw=2, label=f"p50 {p50:.1f} ms")
    ax.axvline(p95, color="#e76f51", lw=2, label=f"p95 {p95:.1f} ms")
    ax.set_xlabel("end-to-end latency (ms)")
    ax.set_ylabel("requests")
    ax.set_title(f"Inference latency on {len(times)} local runs")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return {"n": len(times), "p50_ms": round(p50, 2), "p95_ms": round(p95, 2), "mean_ms": round(float(np.mean(times)), 2)}


def ledger_figure(mlp, lora, rec: dict, path: Path) -> dict:
    pred = predict_mask_mlp(mlp, rec["image"])
    findings = image_findings(pred)
    extracted = extract(lora, rec["note"], findings)
    index = GuidelineIndex()
    hits = index.search(index.query_for_case(rec["note"], findings, extracted), k=4)
    ledger = assemble_ledger(extracted, findings, hits)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), gridspec_kw={"width_ratios": [1, 1.15]})
    axes[0].imshow(overlay_rgb(rec["image"], pred))
    axes[0].axis("off")
    axes[0].set_title(f"{rec['case_id']}  Dice={mean_tumor_dice(pred, rec['labels']):.2f}")
    axes[1].axis("off")
    y = 0.95
    axes[1].text(0, y, "Evidence ledger", fontsize=13, fontweight="bold", transform=axes[1].transAxes)
    y -= 0.08
    colors = {"image": "#0f766e", "retrieval": "#3730a3", "ungrounded": "#b42318"}
    for field, payload in ledger["fields"].items():
        axes[1].text(
            0,
            y,
            f"{field}: {payload['value']}   [{payload['provenance']}]",
            color=colors.get(payload["provenance"], "#223044"),
            fontsize=10,
            family="monospace",
            transform=axes[1].transAxes,
        )
        y -= 0.07
        ev = "; ".join(payload.get("evidence") or [])[:88]
        axes[1].text(0.02, y, ev, fontsize=8, color="#5b6b7c", transform=axes[1].transAxes)
        y -= 0.08
    axes[1].text(
        0,
        0.06,
        f"grounded_fraction={ledger['grounded_fraction']}   contradictions={len(ledger['contradictions'])}",
        fontsize=9,
        transform=axes[1].transAxes,
    )
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return {"case_id": rec["case_id"], "ledger": ledger, "hits": [{"id": h.chunk_id, "title": h.title, "score": h.score} for h in hits], "note": rec["note"]}


def main() -> None:
    _style()
    OUT.mkdir(parents=True, exist_ok=True)
    mlp, _head = load_segmenter(MODELS / "segmenter.joblib")
    lora = load_lora(MODELS)
    cases = load_test_cases()
    metrics = json.loads((MODELS / "metrics.json").read_text(encoding="utf-8"))

    gallery(mlp, cases, OUT / "gallery.png")
    dices = dice_hist(mlp, cases, OUT / "dice_hist.png")
    metric_bars(metrics, OUT / "metrics_bars.png")
    ablation_chart(metrics, OUT / "ablation.png")
    lat = latency_chart(OUT / "latency.png")
    example = ledger_figure(mlp, lora, cases[2], OUT / "ledger.png")

    # refresh harness numbers from the same weights
    harness = run_harness(cases, mlp, lora).as_dict()
    summary = {
        "harness": harness,
        "dice_mean": float(np.mean(dices)),
        "dice_p10": float(np.percentile(dices, 10)),
        "dice_p90": float(np.percentile(dices, 90)),
        "latency": lat,
        "example_case": example["case_id"],
        "example_ledger": example["ledger"],
        "example_hits": example["hits"],
        "example_note": example["note"],
    }
    (ROOT / "docs" / "results.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    (MODELS / "metrics.json").write_text(json.dumps(harness, indent=2), encoding="utf-8")
    print(json.dumps({"wrote": str(OUT), "dice_mean": summary["dice_mean"], "p95_ms": lat["p95_ms"]}, indent=2))


if __name__ == "__main__":
    main()
