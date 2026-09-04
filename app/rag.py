"""Guideline RAG with retrieval metrics and groundedness scoring."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

TOKEN = re.compile(r"[a-z0-9]+")


GUIDELINES = [
    {
        "id": "who-cns-2021-gbm",
        "title": "WHO CNS 2021 glioblastoma",
        "text": (
            "Glioblastoma, IDH-wildtype, is WHO grade 4. Typical imaging shows a necrotic "
            "ring-enhancing mass with surrounding T2/FLAIR edema in adults. Diagnosis is "
            "integrated: histology plus IDH and usually MGMT promoter methylation for treatment planning."
        ),
        "tags": ["grade", "enhancement", "4"],
    },
    {
        "id": "who-cns-2021-lgg",
        "title": "Lower-grade diffuse gliomas",
        "text": (
            "WHO grade 2 diffuse gliomas often show little or no gadolinium enhancement. "
            "Grade 3 anaplastic tumors may enhance focally. Grade cannot be assigned from a single "
            "MRI slice alone; molecular markers (IDH, 1p/19q) are required."
        ),
        "tags": ["grade", "2", "3", "enhancement"],
    },
    {
        "id": "laterality-eloquent",
        "title": "Laterality and eloquent cortex",
        "text": (
            "Report patient laterality, not viewer laterality. Left-hemisphere temporal and frontal "
            "tumors more often present with word-finding difficulty or seizures. Right parietal "
            "lesions may present with neglect. Always reconcile laterality between the note and the mask."
        ),
        "tags": ["laterality", "symptom", "temporal", "frontal"],
    },
    {
        "id": "imaging-t1c-flair",
        "title": "T1c and FLAIR roles",
        "text": (
            "Contrast-enhancing core on T1-weighted post-contrast imaging approximates cellular tumor "
            "and necrosis in glioblastoma. FLAIR hyperintensity around the core includes edema and "
            "infiltrative non-enhancing tumor and should not be called pure edema."
        ),
        "tags": ["enhancement", "edema", "core"],
    },
    {
        "id": "mgmt-idh",
        "title": "IDH and MGMT",
        "text": (
            "IDH-mutant gliomas have a better prognosis than IDH-wildtype. MGMT promoter methylation "
            "predicts benefit from temozolomide in glioblastoma. These results are laboratory, not "
            "visible on a single conventional MRI slice."
        ),
        "tags": ["grade", "molecular"],
    },
    {
        "id": "lobe-symptoms",
        "title": "Lobe-symptom priors",
        "text": (
            "Frontal: personality change or hemiparesis. Temporal: seizure or word-finding difficulty. "
            "Parietal: sensory or visual-spatial symptoms. Occipital: visual field cut. Insular: mixed "
            "seizure and language symptoms. Priors are not diagnoses."
        ),
        "tags": ["lobe", "symptom"],
    },
    {
        "id": "tumor-board",
        "title": "Multidisciplinary glioma protocol",
        "text": (
            "New enhancing intra-axial masses should be discussed at neuro-oncology tumor board. "
            "Standard workup includes maximal safe resection when feasible, IDH immunohistochemistry, "
            "and MGMT. Do not infer WHO grade from enhancement alone."
        ),
        "tags": ["grade", "enhancement", "protocol"],
    },
    {
        "id": "volume-caution",
        "title": "2D volume proxy",
        "text": (
            "Pixel counts on one axial slice are a volume proxy, not a 3D resection-cavity or "
            "RANO measurement. Use them only as a relative size cue and never as a billing or "
            "trial-eligibility volume."
        ),
        "tags": ["volume"],
    },
    {
        "id": "contradiction-policy",
        "title": "Cross-modal contradiction policy",
        "text": (
            "If the note says left and the segmentation centroid is right (or the reverse), emit a "
            "CONTRADICTION. Do not silently average. Prefer image laterality for the mask-derived "
            "field and keep the note value as a conflicting span."
        ),
        "tags": ["laterality"],
    },
    {
        "id": "seizure-temporal",
        "title": "Seizure and temporal lobe",
        "text": (
            "New adult seizure is a classic presentation of temporal-lobe glioma. Enhancement still "
            "does not prove grade 4; correlate with necrosis, growth, and molecular data."
        ),
        "tags": ["symptom", "temporal", "seizure"],
    },
]


@dataclass
class Hit:
    chunk_id: str
    title: str
    text: str
    score: float


class GuidelineIndex:
    def __init__(self, chunks: list[dict] | None = None):
        self.chunks = chunks or GUIDELINES
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
        self.matrix = self.vectorizer.fit_transform(c["text"] + " " + c["title"] for c in self.chunks)

    def search(self, query: str, k: int = 4) -> list[Hit]:
        q = self.vectorizer.transform([query])
        sims = cosine_similarity(q, self.matrix).ravel()
        order = np.argsort(-sims)[:k]
        hits = []
        for i in order:
            hits.append(
                Hit(
                    chunk_id=self.chunks[int(i)]["id"],
                    title=self.chunks[int(i)]["title"],
                    text=self.chunks[int(i)]["text"],
                    score=float(sims[int(i)]),
                )
            )
        return hits

    def query_for_case(self, note: str, findings: dict, extracted: dict) -> str:
        bits = [
            note,
            f"laterality {findings.get('laterality')}",
            f"enhancement {findings.get('enhancement')}",
            f"grade {extracted.get('grade', {}).get('value', '')}",
            f"lobe {extracted.get('lobe', {}).get('value', '')}",
        ]
        return " ".join(str(b) for b in bits)


def ndcg_at_k(gains: list[float], k: int) -> float:
    gains = gains[:k]
    dcg = sum(g / math.log2(i + 2) for i, g in enumerate(gains))
    ideal = sorted(gains, reverse=True)
    idcg = sum(g / math.log2(i + 2) for i, g in enumerate(ideal[:k]))
    if idcg == 0:
        return 0.0
    return dcg / idcg


def relevance(chunk: dict, gold: dict) -> float:
    """Cheap graded relevance for the eval harness (1.0 / 0.5 / 0.0)."""
    tags = set(chunk.get("tags", []))
    score = 0.0
    if gold.get("grade") and gold["grade"] in tags:
        score = max(score, 1.0)
    if gold.get("laterality") and "laterality" in tags:
        score = max(score, 0.5)
    if gold.get("lobe") and gold["lobe"] in chunk.get("text", "").lower():
        score = max(score, 0.5)
    if gold.get("enhancement") and "enhancement" in tags:
        score = max(score, 0.5)
    return score


def grounded_span(value: str, texts: list[str]) -> bool:
    v = str(value).lower().strip()
    if not v or v == "unknown":
        return False
    blob = " ".join(texts).lower()
    if v in blob:
        return True
    aliases = {
        "4": ["grade 4", "glioblastoma", "who grade 4"],
        "3": ["grade 3", "anaplastic"],
        "2": ["grade 2", "lower-grade"],
        "yes": ["enhanc"],
        "no": ["little or no", "minimal enhancement", "no gadolinium"],
    }
    return any(a in blob for a in aliases.get(v, []))
