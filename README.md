# Vision Hallucination Detector

Detecting and classifying **hallucination spans in image-conditioned VLM outputs** — a
reproducible evaluation harness and a set of model experiments for the
[SHROOM-visions 2026](https://helsinki-nlp.github.io/shroom/2026) shared task.

![python](https://img.shields.io/badge/python-3.11%2B-blue)
![license](https://img.shields.io/badge/license-MIT-green)
![tests](https://img.shields.io/badge/tests-39%20passing-brightgreen)

---

## The task

Given an **image**, a **question**, and a model-generated **answer**, mark the character
spans of the answer that are *hallucinated* — not supported by, or contradicting, the image —
and classify each into one of five types (`invention`, `mischaracterization`, `OCR`,
`miscounting`, `other`). Systems output, per character, (a) a probability of being
hallucinated and (b) a category. Two official metrics, ranked per language (en/fr/it/zh):

- **Span identification** — character-level IoU of hallucinated characters.
- **Confidence calibration** — correlation between the predicted per-character probability
  and the empirical (multi-annotator) probability.

This repo also implements the organizers' fuller scorer: `roc_auc`, `pr_auc`,
`best_threshold`, `span_iou`, `calib_corr`, and per-class ROC (`class_roc`).

## What's here

- A **model-agnostic evaluation harness** (`shroom/`) — data loading, an image-grouped dev
  split, robust span alignment, self-consistency aggregation, and the full metric suite.
  The deterministic core is covered by **39 unit tests** and runs with no model or GPU.
- **Swappable VLM backends** — HF/transformers (GPU) and MLX (Apple Silicon) behind one
  `generate()` interface, plus a `MockBackend` for testing.
- **Experiment drivers** (`scripts/`) — zero-shot extraction, claim-level verification, a
  clean-gate, a claim-gate × extraction **hybrid**, and a **multimodal few-shot gate** with a
  continuous logprob score and visual controls.
- A written **[design](docs/design.md)**, **[implementation plan](docs/plan.md)**, and an
  honest **[findings report](docs/findings.md)**.

## Key results (representative en dev, character-level)

The character-IoU metric has a **strong "predict-nothing" floor** (~0.21), because ~21% of
answers are clean and empty-vs-empty scores 1.0. Progression of systems (full suite in
[docs/findings.md](docs/findings.md)):

| system | span_iou | roc_auc | calib_corr |
|--------|----------|---------|------------|
| predict-nothing (floor) | 0.210 | 0.500 | 0.000 |
| starter (Qwen2-VL-2B, permissive prompt) | 0.199 | 0.498 | −0.02 |
| Gemma-3-12B extraction (self-consistency) | 0.189 | 0.604 | 0.142 |
| claim-level verification | 0.249 | 0.576 | 0.163 |
| **HYBRID (claim clean-gate + extraction spans)** | **0.273** | 0.600 | 0.164 |

Those zero-shot rows come from an earlier 200-item sample. For a like-for-like table —
every system, including the trained decoders, re-scored on **one** eval set with **one**
scorer and bootstrap CIs against the floor — see [docs/comparison.md](docs/comparison.md).
On the frozen tune-202 split the starter baseline lands at 0.2005 span_iou (below the 0.2129
floor, roc_auc 0.495) while the trained decoders reach 0.320.

And a controlled study of a **binary hallucination gate** (Gemma-4-12B, multimodal
few-shot), scored threshold-free via `logP(YES) − logP(NO)`:

| condition | ROC-AUC |
|-----------|---------|
| correct image | 0.694 |
| shuffled image (control) | 0.629 |
| no image (control) | 0.566 |

The monotonic drop shows a **real but partial visual signal** — the correct image adds
~0.13 AUC over text-only, with a linguistic floor above chance. Honest takeaway: naive
prompting gives a modest signal; the ceiling is detector recall, not the pipeline.

## Repository layout

```
shroom/            # model-free core: data, split, align, aggregate, metrics, predict, backends, CLIs
scripts/           # experiment runners (MLX/HF drivers, gate, hybrid, official scorer, sweeps)
configs/           # example run config
docs/              # design.md, plan.md, findings.md
tests/             # unit tests for the deterministic core (+ a synthetic fixture)
data/              # how to obtain the SHROOM-visions dataset (not shipped)
requirements/      # core.txt (+ mlx.txt, hf.txt)
```

## Quickstart

```bash
python3.13 -m venv .venv && . .venv/bin/activate
pip install -r requirements/core.txt
python -m pytest                                  # 39 tests, no model needed

bash scripts/get_data.sh                           # download + unpack the dataset (see data/README.md)
python -m shroom.make_split                        # image-grouped dev split per language
python -m shroom.run_eval --gold splits/dev.en.jsonl --pred <predictions.jsonl>
```

Training on a GPU box, pinned to one card:

```bash
bash scripts/connector/run_train_h100.sh --gpu 0   # --dry-run first to see the resolved plan
```

Run a VLM on Apple Silicon (MLX):

```bash
pip install -r requirements/mlx.txt
python scripts/mlx_predict.py --model_id mlx-community/gemma-3-12b-it-4bit \
  --input splits/dev.en.jsonl --image_dir ../Shroom-Vision/images \
  --output preds/dev.en.jsonl --n_samples 5 --temperature 0.7 --max_side 768 --save_raw
python scripts/official_metrics.py --gold splits/dev.en.jsonl --pred sys=preds/dev.en.jsonl
```

See [docs/findings.md](docs/findings.md) for the hybrid, gate, and ablation commands.

## Method in one paragraph

A VLM reads image + question + answer and outputs hallucinated phrases; robust alignment maps
phrases to character spans; **self-consistency** over N samples turns per-character frequency
into a calibrated probability (for calibration) which a threshold τ converts into spans (for
IoU). A **hybrid** uses claim-level verification as an item-level clean-gate and takes
extraction spans on the flagged items, winning span-IoU while staying positively calibrated.

## Status & roadmap

- ✅ Harness, metrics, dev split, backends, unit tests.
- ✅ Mac (MLX) baselines and the full gate/ablation study.
- ⬜ Stronger / fine-tuned detectors on GPU (the recall ceiling), all four languages, and
  confirmation on a natural random sample with a held-out final split.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). The core stays deterministic and tested; experiments
live in `scripts/`; results are reported with confidence intervals and against trivial
baselines. Please don't commit dataset content.

## Acknowledgements

Data and task: [SHROOM-visions 2026](https://helsinki-nlp.github.io/shroom/2026)
(Helsinki-NLP). Licensed under [MIT](LICENSE).
