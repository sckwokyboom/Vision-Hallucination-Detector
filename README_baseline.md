# SHROOM-visions baseline & eval harness

Model-agnostic pipeline for SHROOM-visions 2026 (hallucination-span detection in
image-conditioned text). See the design in
`docs/superpowers/specs/2026-07-21-shroom-visions-baseline-design.md` and the plan in
`docs/superpowers/plans/2026-07-21-shroom-visions-baseline.md`.

## What runs where

- **Core (data / split / align / aggregate / metrics / prompt-parse)** — pure Python, no model.
  Fully unit-tested; runs on a Mac.
- **VLM backends** — `hf` (transformers / Qwen2.5-VL on A100) and `mlx` (Qwen2.5-VL on Mac) behind
  one `VLMBackend.generate()` interface. Swapped via `configs/*.yaml`.

## Setup

```bash
python3.13 -m venv .venv && . .venv/bin/activate
pip install -r requirements-core.txt        # core + tests (Mac/anywhere)
# on the A100 cluster, additionally:
pip install -r requirements-hf.txt
# for the Mac smoke path, additionally:
pip install -r requirements-mlx.txt
```

Run the tests: `python -m pytest`  (39 tests, ~0.3s, no model needed).

## 1. Freeze the dev split (image-grouped, per language)

```bash
python -m shroom.make_split          # writes splits/{lang}.json and splits/dev.{lang}.jsonl
```

The split hashes `image_name`, so all questions about an image stay on the same side (no leakage).
Deterministic given `seed`/`dev_frac` in `configs/base.yaml`.

## 2. Extract images (once, before any VLM run)

```bash
bash scripts/extract_images.sh       # -> ../Shroom-Vision/images/  (2.4 GB archive)
```

## 3. Predict + evaluate

Cluster (open VLM on A100), per language:
```bash
python -m shroom.run_predict --config configs/base.yaml \
  --input splits/dev.en.jsonl --image_dir ../Shroom-Vision/images --output preds/dev.en.jsonl
python -m shroom.run_eval --gold splits/dev.en.jsonl --pred preds/dev.en.jsonl
```

Mac smoke (small model, few items) — set `backend: mlx` in a config and use `--max_samples`.

`run_eval` prints, per language: Char-IoU, the two trivial baselines, calibration
(Pearson/Spearman), and per-label IoU.

## The numbers to beat (predict-nothing on our frozen dev split)

| lang | dev n | Char-IoU baseline | predict-all |
|------|-------|-------------------|-------------|
| en   | 359   | 0.2117            | 0.1736      |
| fr   | 357   | 0.2969            | 0.1929      |
| it   | 329   | 0.2888            | 0.1572      |
| zh   | 352   | 0.3551            | 0.1559      |

A real system must clear the "Char-IoU baseline" column per language.

## Tuning

Edit `configs/base.yaml`:
- `n_samples` — self-consistency samples; higher = smoother per-char probs (better calibration), more cost.
- `tau` — per-char probability threshold for the IoU metric. Sweep on dev to maximize mean IoU.
- `max_pixels` — image resolution cap; raise for OCR / counting.
- `model_id` — `Qwen/Qwen2.5-VL-7B-Instruct` → `32B`/`72B` as dev IoU justifies.

## Status

Core + harness: done, tested, Milestone 1 verified on real dev. VLM backends: implemented; their
end-to-end smoke run (cluster `hf`, Mac `mlx`) and the full dev prediction run are the next steps.
