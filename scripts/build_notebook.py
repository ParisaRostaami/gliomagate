"""Build and execute notebooks/gliomagate_results.ipynb with outputs saved."""

from __future__ import annotations

import json
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "notebooks" / "gliomagate_results.ipynb"


def cells() -> list:
    return [
        new_markdown_cell(
            """# GliomaGate results

This notebook loads the trained weights in `models/` and the 32 held-out slices in `data/test/`. Training itself is `python scripts/train.py`; the loss curves are in `models/train_log.json`."""
        ),
        new_markdown_cell(
            """## Setup

Weights: `models/segmenter.joblib` + LoRA adapters. Held-out cases: `data/test/` (32 slices)."""
        ),
        new_code_cell(
            """from pathlib import Path
import json, sys
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import Image, display, Markdown

ROOT = Path.cwd() if (Path.cwd() / 'app').exists() else Path.cwd().parent
sys.path.insert(0, str(ROOT))
from app.segment import load_segmenter, predict_mask_mlp, mean_tumor_dice, image_findings
from app.lora_extract import load_lora, extract
from app.rag import GuidelineIndex
from app.ground import assemble_ledger
from app.synth import load_case, load_manifest
from app.main import overlay_rgb
from app.eval_harness import run_harness

MODELS = ROOT / 'models'
TEST = ROOT / 'data' / 'test'
FIGS = ROOT / 'docs' / 'figures'
mlp, head = load_segmenter(MODELS / 'segmenter.joblib')
lora = load_lora(MODELS)
cases = []
for meta in load_manifest(TEST):
    img, labels, m = load_case(TEST, meta['case_id'])
    rec = dict(m); rec['image']=img; rec['labels']=labels
    cases.append(rec)
print(f'loaded {len(cases)} held-out cases from {TEST}')
print('segmenter:', type(mlp).__name__, 'LoRA fields:', list(lora))
log = json.loads((MODELS / 'train_log.json').read_text())
print('MLP iterations:', log['segmenter']['n_iter'], 'final CE:', round(log['segmenter']['final_loss'], 4))
print('MLP train Dice:', round(log['segmenter']['train_dice'], 3))
print('LoRA train acc:', {k: round(v, 3) for k, v in log['lora']['train_field_acc'].items()})
print('LoRA test acc:', {k: round(v, 3) for k, v in log['lora']['test_field_acc'].items()})"""
        ),
        new_markdown_cell("## Training curves"),
        new_code_cell(
            """fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.6))
axes[0].plot(log['segmenter']['loss_curve'], color='#355070')
axes[0].set_title(f'Pixel MLP ({log["segmenter"]["n_iter"]} iters)')
axes[0].set_xlabel('iteration'); axes[0].set_ylabel('cross-entropy')
for field, curve in log['lora']['loss_curves'].items():
    axes[1].plot(curve, label=field)
axes[1].set_title('LoRA student loss'); axes[1].legend(frameon=False, fontsize=8)
plt.show()"""
        ),
        new_markdown_cell(
            """## 1. What a case looks like

Synthetic T1-post-contrast-like slice: skull ring, bias field, enhancing core (label 2) and edema halo (label 1). The paired note is a neuro-oncology consult. Laterality in this generator is image-left = patient-left."""
        ),
        new_code_cell(
            """rec = cases[2]
pred = predict_mask_mlp(mlp, rec['image'])
fig, ax = plt.subplots(1, 3, figsize=(10.5, 3.4))
ax[0].imshow(rec['image'], cmap='gray'); ax[0].set_title('slice'); ax[0].axis('off')
ax[1].imshow(rec['labels'], cmap='viridis', vmin=0, vmax=2); ax[1].set_title('ground-truth mask'); ax[1].axis('off')
ax[2].imshow(overlay_rgb(rec['image'], pred)); ax[2].set_title(f'prediction  Dice={mean_tumor_dice(pred, rec[\"labels\"]):.2f}'); ax[2].axis('off')
plt.show()
print(rec['note'])"""
        ),
        new_markdown_cell("## 2. Segmentation gallery (held-out)"),
        new_code_cell(
            """fig, axes = plt.subplots(2, 4, figsize=(12, 6.2))
for i, rec in enumerate(cases[:4]):
    pred = predict_mask_mlp(mlp, rec['image'])
    axes[0, i].imshow(rec['image'], cmap='gray')
    axes[0, i].set_title(rec['case_id'])
    axes[0, i].axis('off')
    axes[1, i].imshow(overlay_rgb(rec['image'], pred))
    axes[1, i].set_title(f'Dice {mean_tumor_dice(pred, rec[\"labels\"]):.2f}')
    axes[1, i].axis('off')
fig.suptitle('Row 1: input  ·  Row 2: overlay (teal edema, red core)')
plt.show()"""
        ),
        new_markdown_cell("## 3. Dice distribution"),
        new_code_cell(
            """dices = [mean_tumor_dice(predict_mask_mlp(mlp, c['image']), c['labels']) for c in cases]
fig, ax = plt.subplots(figsize=(6.5, 3.6))
ax.hist(dices, bins=10, color='#355070', edgecolor='white')
ax.axvline(np.mean(dices), color='#e76f51', lw=2, label=f'mean {np.mean(dices):.3f}')
ax.set_xlabel('mean tumor Dice'); ax.set_ylabel('cases'); ax.legend(frameon=False)
plt.show()
print(f'n={len(dices)}  mean={np.mean(dices):.3f}  p10={np.percentile(dices,10):.3f}  p90={np.percentile(dices,90):.3f}')"""
        ),
        new_markdown_cell(
            """## 4. Structured extraction + evidence ledger

Each field is tagged `image`, `retrieval`, or `ungrounded`. Grade is not allowed to come from the slice alone."""
        ),
        new_code_cell(
            """rec = cases[2]
pred = predict_mask_mlp(mlp, rec['image'])
findings = image_findings(pred)
extracted = extract(lora, rec['note'], findings)
index = GuidelineIndex()
hits = index.search(index.query_for_case(rec['note'], findings, extracted), k=4)
ledger = assemble_ledger(extracted, findings, hits)
print('NOTE:\\n', rec['note'], '\\n')
print(json.dumps(ledger, indent=2, default=str))
print('\\nRETRIEVED:')
for h in hits:
    print(f'  {h.score:.3f}  {h.title}')"""
        ),
        new_markdown_cell("## 5. Full harness (32 cases)"),
        new_code_cell(
            """result = run_harness(cases, mlp, lora)
metrics = result.as_dict()
display(Markdown('### Scores'))
print(json.dumps(metrics, indent=2))
fig, ax = plt.subplots(figsize=(7.8, 4.0))
keys = ['Dice', 'nDCG@5', 'Recall@5', 'Grounded', 'No-RAG grounded']
vals = [metrics['dice'], metrics['ndcg5'], metrics['recall5'], metrics['grounded_fraction'], metrics['ablation']['no_rag_grounded_fraction']]
ax.bar(keys, vals, color=['#355070','#4a7c9b','#4a7c9b','#2a9d8f','#e9c46a'])
ax.set_ylim(0, 1.08); ax.set_title('Held-out harness')
for i,v in enumerate(vals):
    ax.text(i, v+0.02, f'{v:.2f}', ha='center')
plt.show()"""
        ),
        new_markdown_cell(
            """### Ablation

Turning off retrieval drops how many fields we can support with a source. Segmentation and the note still run; the ledger just has less to cite."""
        ),
        new_code_cell(
            """fig, ax = plt.subplots(figsize=(5.6, 3.6))
ax.bar(['full grounded', 'no RAG grounded'],
       [metrics['grounded_fraction'], metrics['ablation']['no_rag_grounded_fraction']],
       color=['#2a9d8f','#e76f51'])
ax.set_ylim(0,1.05); ax.set_title('Retrieval ablation')
plt.show()"""
        ),
        new_markdown_cell("## 6. Latency"),
        new_code_cell(
            """import time
lat = []
for rec in cases:
    t0 = time.perf_counter()
    pred = predict_mask_mlp(mlp, rec['image'])
    f = image_findings(pred)
    ex = extract(lora, rec['note'], f)
    ht = index.search(index.query_for_case(rec['note'], f, ex), k=4)
    assemble_ledger(ex, f, ht)
    lat.append((time.perf_counter()-t0)*1000)
lat.sort()
p50, p95 = lat[len(lat)//2], lat[int(0.95*(len(lat)-1))]
fig, ax = plt.subplots(figsize=(6.4, 3.5))
ax.hist(lat, bins=12, color='#355070', edgecolor='white')
ax.axvline(p50, color='#2a9d8f', label=f'p50 {p50:.1f} ms')
ax.axvline(p95, color='#e76f51', label=f'p95 {p95:.1f} ms')
ax.set_xlabel('ms'); ax.legend(frameon=False); ax.set_title('End-to-end latency (32 cases, this machine)')
plt.show()
print(f'p50={p50:.2f} ms  p95={p95:.2f} ms  mean={np.mean(lat):.2f} ms')
bench = json.loads((MODELS/'bench.json').read_text())
print('saved bench.json', json.dumps(bench, indent=2))"""
        ),
        new_markdown_cell(
            """## Notes

Slices are synthetic (`app/synth.py`), so Dice is not a BraTS score. Extraction uses the trained LoRA adapters, not a keyword lookup.

Figures saved under `docs/figures/`:"""
        ),
        new_code_cell(
            """for p in sorted(FIGS.glob('*.png')):
    print(p.name)
    display(Image(filename=str(p), width=720))"""
        ),
    ]


def main() -> None:
    NB.parent.mkdir(parents=True, exist_ok=True)
    nb = new_notebook(cells=cells(), metadata={"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}})
    NB.write_text(nbformat.writes(nb), encoding="utf-8")
    client = NotebookClient(nb, timeout=300, kernel_name="python3", resources={"metadata": {"path": str(ROOT)}})
    client.execute()
    NB.write_text(nbformat.writes(nb), encoding="utf-8")
    print("wrote", NB, "cells", len(nb.cells))


if __name__ == "__main__":
    main()
