"""Build a static Space: sample inferences + figures, no paid Docker hardware."""

from __future__ import annotations

import base64
import json
import sys
from io import BytesIO
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.ground import assemble_ledger
from app.lora_extract import extract, load_lora
from app.main import overlay_rgb
from app.rag import GuidelineIndex
from app.segment import image_findings, load_segmenter, mean_tumor_dice, predict_mask_mlp
from app.synth import load_case, load_manifest

OUT = ROOT / "spaces_static"


def png_b64(arr) -> str:
    if arr.ndim == 2:
        im = Image.fromarray((arr.clip(0, 1) * 255).astype("uint8"), mode="L")
    else:
        im = Image.fromarray(arr.astype("uint8"), mode="RGB")
    buf = BytesIO()
    im.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    mlp, _ = load_segmenter(ROOT / "models" / "segmenter.joblib")
    lora = load_lora(ROOT / "models")
    index = GuidelineIndex()
    samples = []
    manifest = load_manifest(ROOT / "data" / "test")
    for meta in manifest[:4]:
        img, labels, m = load_case(ROOT / "data" / "test", meta["case_id"])
        pred = predict_mask_mlp(mlp, img)
        findings = image_findings(pred)
        extracted = extract(lora, m["note"], findings)
        hits = index.search(index.query_for_case(m["note"], findings, extracted), k=3)
        ledger = assemble_ledger(extracted, findings, hits)
        samples.append(
            {
                "case_id": m["case_id"],
                "note": m["note"],
                "dice": round(mean_tumor_dice(pred, labels), 3),
                "source_png": png_b64(img),
                "mask_png": png_b64(overlay_rgb(img, pred)),
                "ledger": ledger,
                "retrieval": [{"title": h.title, "score": round(h.score, 3), "text": h.text} for h in hits],
            }
        )
    metrics = json.loads((ROOT / "models" / "metrics.json").read_text(encoding="utf-8"))
    bench = json.loads((ROOT / "models" / "bench.json").read_text(encoding="utf-8"))
    payload = {"metrics": metrics, "bench": bench, "samples": samples}
    def _json(o):
        if hasattr(o, "item"):
            return o.item()
        raise TypeError(type(o))

    (OUT / "results.js").write_text(
        "window.GLIOMA = " + json.dumps(payload, default=_json) + ";",
        encoding="utf-8",
    )
    print("wrote", OUT / "results.js", "bytes", (OUT / "results.js").stat().st_size)


if __name__ == "__main__":
    main()
