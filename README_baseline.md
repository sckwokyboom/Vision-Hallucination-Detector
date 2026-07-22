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

## Mac self-consistency workflow

```bash
# n>1 sampling, saving raw outputs so tau/alignment can be swept offline (no re-inference)
python scripts/mlx_predict.py --model_id mlx-community/Qwen2.5-VL-7B-Instruct-4bit \
  --input splits/dev.en.jsonl --image_dir ../Shroom-Vision/images \
  --output preds/dev.en.jsonl --n_samples 5 --temperature 0.7 --max_side 768 --save_raw
python scripts/sweep_tau.py --gold splits/dev.en.jsonl --pred preds/dev.en.jsonl
```

`--max_side` downscales images (fixes Qwen2.5-VL degenerate "!!!!" output on large images).
`sweep_tau.py` re-aggregates the saved raw samples across a tau grid instantly.

## First-pass results (Mac, Qwen2.5-VL-7B-4bit via MLX, ~3–5 s/item on M4 Pro)

Representative random 100-item en-dev sample (floor 0.210, i.e. 21% clean — matches full dev 0.212):

| config | Char-IoU | vs floor |
|--------|----------|----------|
| predict-nothing (floor) | 0.210 | — |
| n=5, tau=0.20 | 0.152 | −0.058 |
| n=5, tau=0.50 | 0.185 | −0.025 |
| **n=5, tau=0.70 (best)** | **0.200** | −0.010 |

**Conclusion:** tuned zero-shot 7B-4bit essentially ties the predict-nothing floor; it does not beat it.
Self-consistency + high tau gives a real gain (0.152 → 0.200) but only offsets the over-prediction
penalty. The ceiling is the small quantized model's recall/precision on this clean-heavy task.

Two empirical must-dos discovered here: (1) resize images (`--max_side`), or Qwen2.5-VL emits "!!!!";
(2) a **skeptical** prompt — the original permissive prompt made even 7B return `[]` everywhere.

## Model comparison & clean-gate (representative 100 en-dev, floor 0.210)

| | Qwen2.5-VL-7B-4bit | Gemma-3-12b-4bit |
|--------|--------------------|-------------------|
| Char-IoU (best) | 0.200 (n=5, tau=0.7) | 0.202 (n=1 + gate) |
| Recall (dirty items flagged) | low (~24% fire) | **76 / 79** |
| Median predicted span | 94 chars | **29 chars** |
| Calibration (Pearson) | ~0 / negative | **+0.144** |
| Failure mode | under-fires (defaults "clean") | over-fires (flags all incl. clean) |

A two-stage **clean-gate** (`scripts/mlx_gate.py` + `sweep_gate.py`: a YES/NO "is anything
hallucinated?" question that forces empty spans on NO) **does not help** either model: neither tells
clean from dirty as a global yes/no (Qwen says NO on 19/21 clean answers, Gemma says YES on 19/21).

**Claim-level (Method B — `scripts/mlx_claim.py` + `sweep_claim.py`) is the first approach to beat
the floor.** Split the answer into sentence claims and ask Gemma per claim "is this supported by the
image?" with a filler-aware skeptical prompt. Result on the representative 100 (floor 0.210):

| approach | Char-IoU | vs floor | calibration (Pearson) |
|----------|----------|----------|-----------------------|
| Qwen extraction (n=5, tau=0.7) | 0.200 | −0.010 | ~0 |
| Gemma extraction | 0.187 | −0.023 | +0.144 |
| Gemma + clean-gate | 0.202 | −0.008 | +0.144 |
| **Gemma claim-level** | **0.2147** | **+0.005** | +0.109 |

Claim-level wins by protecting clean items: 14/21 clean answers correctly get `[]` (vs 0/21 for
extraction), while catching 35/79 dirty. Its ceiling is (a) recall — 44/79 dirty missed (subtle
hallucinations), and (b) coarse spans — whole sentences are marked, but gold spans are short words.

## Best Mac solution: HYBRID E (`scripts/hybrid_eval.py`)

Combine the two: use claim-level as an **item-level clean-gate** (predict `[]` when it flags no claim)
and take **Gemma's extraction spans** on the flagged (dirty) items. One model (Gemma-12b), two passes.
On **200** representative en-dev items (42 clean, floor 0.210):

| system | Char-IoU | Pearson | clean protected |
|--------|----------|---------|-----------------|
| predict-nothing (floor) | 0.2100 | 0.000 | 42/42 |
| predict-everything | 0.1607 | 0.000 | 0/42 |
| Gemma extraction | 0.1907 | 0.125 | 2/42 |
| claim-level | 0.2493 | 0.163 | 33/42 |
| **HYBRID E** | **0.2728** | **0.151** | 33/42 |

HYBRID E beats predict-nothing by **+0.063** mean per-item IoU (95% bootstrap CI **[+0.018, +0.106]**,
significant; better on 64 items, worse on 9). It wins on **both** official metrics at once: IoU well
above the floor *and* a positive calibration signal.

```bash
# two Gemma passes over the same items, then the hybrid:
python scripts/mlx_predict.py --model_id mlx-community/gemma-3-12b-it-4bit --input splits/dev.en.jsonl \
  --image_dir ../Shroom-Vision/images --output preds/gemma_extr.jsonl --max_side 768 --save_raw
python scripts/mlx_claim.py --model_id mlx-community/gemma-3-12b-it-4bit --input splits/dev.en.jsonl \
  --image_dir ../Shroom-Vision/images --output preds/gemma_claim.jsonl --max_side 768
python scripts/hybrid_eval.py --gold splits/dev.en.jsonl \
  --extraction preds/gemma_extr.jsonl --claim preds/gemma_claim.jsonl --out preds/hybrid.jsonl
```

**Takeaway:** on Mac, HYBRID E (claim clean-gate + Gemma spans) is the working baseline — significantly
above the predict-nothing floor and positively calibrated. It still misses subtle hallucinations
(recall-bound); further gains need a stronger detector (A100 32B/72B) or fine-tuning.

## Status & next levers

Core + harness: done, tested (39 unit tests), Milestone 1 verified, first-pass Mac baseline measured.
To clear the floor, in rough order of expected payoff:
- **Stronger model on the A100 cluster** (`hf` backend, Qwen2.5-VL-32B/72B or unquantized 7B).
- **Fine-tune a token classifier** on the 15k labelled train set (design §4 Method C).
- **Claim-level verification** (design §4 Method B) to raise recall.
- The cluster `hf` smoke run + full multi-language test prediction remain to be executed on the cluster.
