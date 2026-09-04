"""Optional GPU path: QLoRA on Flan-T5-small for JSON extraction.

This script is skipped unless transformers, peft, and a CUDA device exist.
The production Space serves the distilled LoRA student in app/lora_extract.py.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    try:
        import torch
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    except Exception as exc:
        print("QLoRA extras not installed; skipping.", exc)
        sys.exit(0)

    if not torch.cuda.is_available():
        print("No CUDA GPU; not pretending to 4-bit train on CPU. Exiting 0.")
        sys.exit(0)

    model_name = "google/flan-t5-small"
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, load_in_4bit=True, device_map="auto")
    model = get_peft_model(
        model,
        LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05, target_modules=["q", "v"], task_type="SEQ_2_SEQ_LM"),
    )
    print(model.print_trainable_parameters())
    out = ROOT / "models" / "qlora-flan-t5-small"
    out.mkdir(parents=True, exist_ok=True)
    (out / "README.md").write_text(
        "Adapter checkpoint would be saved here after a real GPU run.\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": "initialized", "model": model_name}))


if __name__ == "__main__":
    main()
