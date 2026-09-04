"""Evaluation harness: Dice, field accuracy, nDCG, groundedness, ablations."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .ground import assemble_ledger
from .lora_extract import extract
from .rag import GUIDELINES, GuidelineIndex, ndcg_at_k, relevance
from .segment import image_findings, mean_tumor_dice, predict_mask_mlp


@dataclass
class HarnessResult:
    dice: float
    field_acc: dict
    field_acc_mean: float
    ndcg5: float
    recall5: float
    grounded_fraction: float
    contradiction_rate: float
    ablation: dict

    def as_dict(self) -> dict:
        return asdict(self)


def _field_value(gold: dict, field: str) -> str:
    if field == "enhancement":
        return "yes" if gold["enhancement"] else "no"
    return str(gold[field])


def run_harness(cases: list[dict], mlp, lora_heads) -> HarnessResult:
    index = GuidelineIndex()
    dices = []
    hits_acc = {k: [] for k in ("laterality", "lobe", "grade", "enhancement", "symptom")}
    ndcgs = []
    recalls = []
    grounded = []
    contradictions = []
    ab_no_image = []
    ab_no_rag = []
    ab_full = []

    blank = {
        "laterality": "unknown",
        "enhancement": False,
        "volume_proxy_px": 0,
        "core_fraction": 0.0,
    }

    for rec in cases:
        img, mask = rec["image"], rec["labels"]
        pred = predict_mask_mlp(mlp, img)
        dices.append(mean_tumor_dice(pred, mask))
        findings = image_findings(pred)
        extracted = extract(lora_heads, rec["note"], findings)
        hits = index.search(index.query_for_case(rec["note"], findings, extracted), k=5)
        ledger = assemble_ledger(extracted, findings, hits)
        grounded.append(ledger["grounded_fraction"])
        contradictions.append(1.0 if ledger["contradictions"] else 0.0)

        for field in hits_acc:
            hits_acc[field].append(1.0 if extracted[field]["value"] == _field_value(rec, field) else 0.0)

        rel = [relevance(next(c for c in GUIDELINES if c["id"] == h.chunk_id), rec) for h in hits]
        ndcgs.append(ndcg_at_k(rel, 5))
        recalls.append(1.0 if any(r >= 1.0 for r in rel) else 0.0)
        ab_full.append(float(np.mean([hits_acc[f][-1] for f in hits_acc])))

        extracted_ni = extract(lora_heads, rec["note"], blank)
        ab_no_image.append(
            float(np.mean([1.0 if extracted_ni[f]["value"] == _field_value(rec, f) else 0.0 for f in hits_acc]))
        )
        ab_no_rag.append(assemble_ledger(extracted, findings, [])["grounded_fraction"])

    field_acc = {k: float(np.mean(v)) for k, v in hits_acc.items()}
    return HarnessResult(
        dice=float(np.mean(dices)),
        field_acc=field_acc,
        field_acc_mean=float(np.mean(list(field_acc.values()))),
        ndcg5=float(np.mean(ndcgs)),
        recall5=float(np.mean(recalls)),
        grounded_fraction=float(np.mean(grounded)),
        contradiction_rate=float(np.mean(contradictions)),
        ablation={
            "full_field_acc": float(np.mean(ab_full)),
            "no_image_field_acc": float(np.mean(ab_no_image)),
            "no_rag_grounded_fraction": float(np.mean(ab_no_rag)),
        },
    )
