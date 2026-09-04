from __future__ import annotations

import numpy as np

from app.ground import assemble_ledger
from app.lora_extract import extract, train_lora
from app.rag import GuidelineIndex, ndcg_at_k
from app.segment import image_findings, mean_tumor_dice, predict_mask_mlp, train_segmenter
from app.synth import generate_split, render_slice


def test_laterality_follows_centroid():
    rng = np.random.default_rng(0)
    img, mask = render_slice(rng, "left", "temporal", True)
    f = image_findings(mask)
    assert f["laterality"] == "left"
    assert f["volume_proxy_px"] > 20


def test_dice_perfect_on_identity():
    m = np.zeros((16, 16), dtype=np.uint8)
    m[4:10, 4:10] = 1
    m[6:8, 6:8] = 2
    assert mean_tumor_dice(m, m) == 1.0


def test_ndcg_monotonic():
    assert ndcg_at_k([1.0, 0.0, 0.0], 3) > ndcg_at_k([0.0, 0.0, 1.0], 3)


def test_rag_returns_hits():
    idx = GuidelineIndex()
    hits = idx.search("glioblastoma ring enhancement WHO grade 4", k=3)
    assert hits[0].score > 0
    assert any("gbm" in h.chunk_id or "grade" in h.text.lower() for h in hits)


def test_grade_not_image_groundable():
    extracted = {
        "laterality": {"value": "left", "confidence": 0.9},
        "lobe": {"value": "temporal", "confidence": 0.8},
        "grade": {"value": "4", "confidence": 0.7},
        "enhancement": {"value": "yes", "confidence": 0.8},
        "symptom": {"value": "seizure", "confidence": 0.6},
    }
    findings = {"laterality": "left", "enhancement": True, "volume_proxy_px": 100, "core_fraction": 0.4}
    ledger = assemble_ledger(extracted, findings, [])
    assert ledger["fields"]["grade"]["provenance"] == "ungrounded"
    assert ledger["fields"]["laterality"]["provenance"] == "image"


def test_contradiction_when_note_and_mask_disagree():
    extracted = {
        "laterality": {"value": "right", "confidence": 0.9},
        "lobe": {"value": "unknown", "confidence": 0.2},
        "grade": {"value": "unknown", "confidence": 0.2},
        "enhancement": {"value": "no", "confidence": 0.2},
        "symptom": {"value": "unknown", "confidence": 0.2},
    }
    findings = {"laterality": "left", "enhancement": False, "volume_proxy_px": 40, "core_fraction": 0.1}
    ledger = assemble_ledger(extracted, findings, [])
    assert ledger["contradictions"]


def test_train_tiny_pipeline():
    cases = generate_split(12, seed=1, prefix="u")
    mlp, head = train_segmenter([c["image"] for c in cases], [c["labels"] for c in cases], seed=0)
    dices = [mean_tumor_dice(predict_mask_mlp(mlp, c["image"]), c["labels"]) for c in cases]
    assert float(np.mean(dices)) > 0.55
    findings = [image_findings(predict_mask_mlp(mlp, c["image"])) for c in cases]
    heads = train_lora([c["note"] for c in cases], findings, cases, seed=0)
    pred = extract(heads, cases[0]["note"], findings[0])
    assert pred["laterality"]["value"] in {"left", "right", "unknown"}
