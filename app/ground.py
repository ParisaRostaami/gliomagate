"""Tag each extracted field as image-supported, retrieval-supported, or ungrounded."""

from __future__ import annotations

from .rag import grounded_span
from .segment import image_findings

# Grade and molecular claims cannot be proven from one synthetic slice.
IMAGE_OK = {"laterality", "enhancement"}
RETRIEVAL_OK = {"grade", "lobe", "symptom", "enhancement", "laterality"}


def assemble_ledger(
    extracted: dict,
    findings: dict,
    hits: list,
    gold: dict | None = None,
) -> dict:
    texts = [h.text for h in hits]
    ledger = {}
    contradictions = []

    img_lat = findings.get("laterality")
    note_lat = extracted.get("laterality", {}).get("value")
    if img_lat in {"left", "right"} and note_lat in {"left", "right"} and img_lat != note_lat:
        contradictions.append(
            {
                "field": "laterality",
                "image": img_lat,
                "note": note_lat,
                "policy": "Do not average. Surface both and prefer image for mask-derived laterality.",
            }
        )

    for field, payload in extracted.items():
        value = payload["value"]
        image_ok = False
        retrieval_ok = grounded_span(value, texts)
        evidence = []

        if field == "laterality" and img_lat in {"left", "right"}:
            image_ok = value == img_lat or value == "unknown"
            if value == img_lat:
                evidence.append(f"mask centroid is on the {img_lat} half of the slice")
            if contradictions:
                evidence.append("note laterality conflicts with the mask")
        if field == "enhancement":
            img_enh = bool(findings.get("enhancement"))
            want = "yes" if img_enh else "no"
            image_ok = value == want
            if image_ok:
                evidence.append(
                    f"core fraction={findings.get('core_fraction', 0):.2f} "
                    f"volume_px={findings.get('volume_proxy_px', 0)}"
                )
        if retrieval_ok:
            evidence.append("supported by retrieved guideline span")

        if field == "grade":
            # Explicit scientific constraint: one slice does not grade a glioma.
            image_ok = False
            if not retrieval_ok:
                evidence.append("grade is not image-groundable from a single slice")

        if image_ok and field in IMAGE_OK:
            provenance = "image"
        elif retrieval_ok:
            provenance = "retrieval"
        else:
            provenance = "ungrounded"

        ledger[field] = {
            "value": value,
            "confidence": payload.get("confidence", 0.0),
            "provenance": provenance,
            "evidence": evidence,
        }

    grounded_frac = float(np_mean([1.0 if v["provenance"] != "ungrounded" else 0.0 for v in ledger.values()]))
    return {
        "fields": ledger,
        "contradictions": contradictions,
        "grounded_fraction": round(grounded_frac, 3),
        "image_findings": {
            k: (bool(v) if type(v).__name__ == "bool_" or isinstance(v, bool) else v)
            for k, v in findings.items()
            if k != "centroid_xy"
        },
    }


def np_mean(xs: list[float]) -> float:
    return sum(xs) / max(len(xs), 1)
