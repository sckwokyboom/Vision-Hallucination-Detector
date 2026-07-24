# SHROOM-visions 2026 — Baseline & Evaluation Pipeline

**Date:** 2026-07-21
**Status:** Approved design, pre-implementation
**Task page:** https://helsinki-nlp.github.io/shroom/2026
**Evaluation deadline:** 2026-07-31

---

## 1. Goal

Build a **model-agnostic evaluation harness** plus a **strong zero-shot VLM baseline** for
SHROOM-visions 2026 (fine-grained hallucination-span detection in image-conditioned text).

Two concrete outcomes:

1. A reproducible **eval pipeline** (both official metrics, per-language, fixed dev split) that lets
   us measure any future system on the same footing and know immediately whether a change helped.
2. A **baseline** that, out of the box, clearly beats the trivial "predict nothing" score and
   produces the per-character probabilities the calibration metric needs.

The harness is the primary deliverable; the baseline is the first system run through it.

---

## 2. Task summary (from the official page + data inspection)

- **Task:** detect and classify character spans of hallucination in a model's answer about an image.
- **Per character**, a system must output: (a) probability the character belongs to a hallucinated
  span, (b) the hallucination category.
- **Two primary metrics (character-level, ranked separately per language):**
  1. **Span Identification** — Intersection-over-Union (IoU) of characters marked hallucinated
     (gold vs. predicted). Category-agnostic.
  2. **Confidence Calibration** — correlation between the predicted per-character hallucination
     probability and the empirical probability observed in the multi-annotator gold data.
- **Categories (5 label strings present in the data):** `invention`, `mischaracterization`,
  `OCR`, `miscounting`, `other`.
- **Languages:** `en`, `fr`, `it`, `zh` — separate rankings/submissions per language.
- **Rules:** any approach and external resources allowed; may target any subset of languages.
- **Data volume:** ~15.1k labeled train, ~4.9k unlabeled (closed) test.

### Data schema

Train item (test item is identical minus `labels`):

```json
{
  "id": "train-en-413",
  "split": "train",
  "language": "en",
  "prompt": "Does this desk have legs? Please elaborate.",
  "image_name": "desk_Floating_desk_....jpg",
  "response": "No, this desk does not have legs...",
  "labels": [
    {"start": 148, "end": 154, "prob": 0.3333333333, "label": "mischaracterization"},
    {"start": 221, "end": 228, "prob": 0.3333333333, "label": "mischaracterization"}
  ]
}
```

- `start`/`end` are **character offsets into `response`** (verified: `response[start:end]` returns the
  exact hallucinated substring).
- **`prob`** = annotator agreement fraction for that span (values like 0.333, 0.667, 1.0 ⇒ ~3
  annotators). This is the target signal for the calibration metric and is **ignored by the current
  starter code**.
- Images live in `../Shroom-Vision/shroom-visions-images.tar.gz` (2.4 GB, not yet extracted);
  `image_name` is the file name inside the archive.

### Key statistics (train)

| lang | items | % with hallucination | % clean | avg response len | avg spans/item |
|------|-------|----------------------|---------|------------------|----------------|
| en   | 3799  | 74.7%                | 25.3%   | 517 chars        | 3.39           |
| fr   | 3767  | 74.4%                | 25.6%   | 333              | 3.24           |
| it   | 3746  | 73.7%                | 26.3%   | 298              | 2.50           |
| zh   | 3790  | 63.5%                | 36.5%   | 181              | 1.99           |

- `invention` + `mischaracterization` ≈ 85% of spans; `OCR`/`miscounting`/`other` are rare.
- Images are **reused**: within a language ~2.4 questions per image; **867 images shared across all 4
  languages**. → dev split must group by image (see §6).

### The number to beat (trivial baselines)

The IoU metric scores `empty gold + empty pred = 1.0`. So **predicting "no hallucination"
everywhere** yields IoU = fraction of clean items:

| lang | "predict-nothing" IoU | "predict-all-hallucinated" IoU (approx) |
|------|-----------------------|-----------------------------------------|
| en   | **0.253**             | ~0.18                                   |
| fr   | **0.256**             | ~0.22                                   |
| it   | **0.263**             | ~0.17                                   |
| zh   | **0.365**             | ~0.16                                   |

Average predict-nothing ≈ **0.28**. Any real system must beat this per language; it is also the
baseline the harness prints on every run for context.

---

## 3. Analysis of the existing starter code (`vision-detector/`)

`label_with_gemma.py` prompts a VLM (Qwen2-VL-2B / Gemma) with image+question+answer, asks for a JSON
list of hallucinated phrases, and maps phrases → char spans via `response.find(phrase)`.
`evaluate.py` computes char-IoU. Problems, in priority order:

1. **Calibration ignored** — no per-character probability, so the 2nd official metric is unaddressed.
2. **3 of 5 categories** offered in the prompt (`mischaracterization`/`miscounting`/`invention`);
   `OCR` and `other` can never be predicted.
3. **`.find()` takes the first occurrence only** — repeats / paraphrased quotes are silently mislocated
   or dropped (`idx == -1`).
4. **2B model is unreliable** at exact-substring JSON, especially for fr/it/zh → parse errors → dropped
   items.
5. **No clean train/dev split** and no per-language harness for iterating or tuning thresholds.
6. **`max_pixels=512²`** downscales images, hurting OCR and object-counting.

Good parts we keep/reuse: `evaluate.py`'s char-IoU logic (matches the official definition of the
"predict-nothing = 1.0" edge case), the transformers image-text-to-text plumbing, and the
`run_cluster.sh` environment setup.

---

## 4. Chosen approach — Method A: generative span extraction + self-consistency

The VLM sees `image + prompt + response` and returns the offending substrings, each with a category
from all 5 classes. We run **N samples** at temperature > 0 and aggregate:

- **Per-character probability** `p_i = (# samples whose spans cover char i) / N`. This is a directly
  calibratable estimate for the **calibration metric**.
- **Binary span membership for the IoU metric** = `p_i ≥ τ`, where `τ` is a single threshold tuned on
  the dev split. Contiguous runs of `p_i ≥ τ` become output spans.
- **Category** per span = the majority category among the samples that covered it.

**Why the two metrics need different treatment:** calibration wants a well-spread *continuous* `p_i`;
IoU wants a *binary* set. Self-consistency gives both from the same N samples (raw frequency for
calibration, thresholded frequency for IoU). This is the central design idea.

Rejected/deferred alternatives (recorded for later phases):

- **B — claim-level verification** (split answer into clauses, verify each against the image). More
  robust (no exact-quote requirement) and naturally calibrated, but marks whole clauses ⇒ larger union
  ⇒ lower IoU, since gold spans are often single words. Kept as a **fallback** for hard languages.
- **C — fine-tuned multimodal token classifier** (image+text encoders → per-token BIO + category head,
  trained on the 15k labeled set). Highest ceiling and closest to the metric, but needs training and is
  slower to a first number. **Phase 2 upgrade** once the harness is locked.

---

## 5. Architecture

Model-agnostic pipeline; the only thing that changes between Mac and the A100 cluster is the backend.

```
shroom/
  backends/
    base.py         VLMBackend.generate(image, text, n, temperature) -> list[str]
    hf_backend.py   transformers on A100: Qwen2.5-VL-7B → 32B/72B  (primary)
    mlx_backend.py  Mac (MLX / Ollama): Qwen2.5-VL-3B/7B           (smoke tests only)
  data.py           load JSONL, resolve image paths, dataclasses
  split.py          fixed-seed group split by image_name, per language
  predict.py        build prompt (5 classes + few-shot from train), sample N, parse JSON
  align.py          phrase -> char span(s): normalize markdown/whitespace, ALL occurrences, fuzzy
  aggregate.py      N raw samples -> per-char prob + dominant category -> spans {start,end,prob,label}
  metrics.py        char-IoU (agnostic + per-label) + calibration (Pearson/Spearman) + trivial baselines
  run_predict.py    CLI: predict on a dev or test file -> predictions JSONL
  run_eval.py       CLI: score predictions vs gold, per language, with trivial baselines
configs/
  base.yaml         model id, N, temperature, tau, image resolution, few-shot settings
```

Existing `label_with_gemma.py`, `evaluate.py`, `run_cluster.sh` are **kept**; `metrics.py` extends
`evaluate.py`'s IoU with the calibration metric and per-language reporting.

### Module contracts

- **`VLMBackend` (base.py):** single method `generate(image: PIL.Image, text: str, n: int,
  temperature: float) -> list[str]`. Returns `n` raw model outputs. HF and MLX implementations are
  interchangeable; all downstream code depends only on this interface.
- **`align.py`:** `phrase_to_spans(phrase: str, response: str) -> list[(start, end)]`. Normalizes
  whitespace/markdown, finds **all** occurrences, falls back to fuzzy (token-overlap) matching when no
  exact match exists. Never silently drops a non-empty phrase without recording why.
- **`aggregate.py`:** `aggregate(samples: list[parsed], response: str, tau: float) -> list[span]`
  where each span is `{start, end, prob, label}` in the gold schema. Produces the per-char prob array
  internally and exposes it for the calibration metric.
- **`metrics.py`:** `char_iou(gold, pred, resp_len)`, `calibration(gold_probs, pred_probs)`,
  `evaluate_file(gold_path, pred_path) -> per-language report` including trivial baselines.

---

## 6. Data & dev-split strategy

- **Group split by `image_name`, per language, single fixed seed.** Because images recur across
  questions and languages, a random per-item split would leak. Grouping by image prevents an image
  appearing in both train and dev.
- **Dev size:** ~10% of items per language (≈ 380 items/lang), frozen. All experiments report on this
  exact dev set so numbers are comparable across runs.
- Remaining train is available for few-shot exemplars and (Phase 2) fine-tuning.
- `split.py` writes the id lists to disk so the split is reproducible and inspectable.

---

## 7. Evaluation harness

Implements **both** official metrics; run on the frozen dev split, reported per language.

- **Span Identification (IoU):** char-agnostic. Gold hallucinated set = union of all gold-span chars.
  Predicted hallucinated set = chars with `p_i ≥ τ`. Per-item IoU (empty/empty = 1.0), averaged.
  Also report **per-category IoU** as a diagnostic.
- **Confidence Calibration:** for each char, gold empirical prob `g_i` = max `prob` over gold spans
  covering it (0 if uncovered); predicted `p_i` = self-consistency frequency. Report **Pearson** and
  **Spearman** correlation, pooled over all chars per language. (Exact aggregation matches our reading
  of the task; kept configurable pending the official scorer.)
- **Context lines printed every run:** predict-nothing IoU and predict-all IoU per language, so every
  result is framed against the trivial baselines.
- **Threshold `τ`** is swept on dev to maximize mean IoU; the chosen `τ` is stored in the config used
  for the test-set run.

---

## 8. Output / submission format

- Internally and on disk, predictions use the **gold `labels` schema**: a list of
  `{start, end, prob, label}` per item. This makes both `evaluate.py` and `metrics.py` work without
  translation, and preserves the continuous `prob` for calibration.
- The pipeline can expand spans → per-character `(prob, category)` arrays if the official submission
  requires that shape. **Open item:** confirm the exact submission format and whether calibration is
  Pearson or Spearman once the CodaBench/participant-kit link is available (not present in the local
  snapshot). Building to the gold schema is the safe default meanwhile.

---

## 9. Model & compute plan

- **Primary:** `Qwen2.5-VL` via HF transformers on the A100 cluster. Start at **7B**; scale to
  **32B / 72B** if dev IoU justifies it. Chosen for strong multilingual coverage (incl. zh), OCR and
  grounding ability, and continuity with the existing Qwen2-VL code.
- **Secondary (Mac):** `Qwen2.5-VL-3B/7B` via MLX (or Ollama) — used only to run the full pipeline on a
  tiny subset and validate parsing/alignment/metrics end-to-end before cluster runs.
- Backends are swapped by config; nothing else in the pipeline changes.
- Image handling: raise the resolution cap above 512² (tune on dev) since OCR/miscounting depend on it.

---

## 10. Milestones (phased)

1. **Harness + dev split + trivial baselines** (runs on Mac, no model). Immediately shows the ~0.28
   plank and proves the eval loop.
2. **Smoke test Method A** on 20–50 items on Mac (small VLM) — verify JSON parsing, span alignment, and
   both metrics work end-to-end.
3. **Full dev run on A100** (Qwen2.5-VL-7B, N=5) — first honest number. Target: clearly > 0.28;
   realistically ~0.4–0.5 IoU on en (confirmed only by the run — not promised).
4. **Tune on dev:** `τ`, few-shot content, image resolution, model size.
5. **Test-set run** for all 4 languages → submission files.

Phase 2 (post-baseline): Method C fine-tuned token classifier; Method B fallback for weak languages.

---

## 11. Risks & open questions

- **Submission format / scorer details** not confirmed locally (see §8) — resolve before the test run.
- **Exact-quote reliability** of the VLM drives IoU; mitigated by robust + fuzzy alignment and by N-sample
  aggregation, but is the main baseline risk. Method B is the fallback if alignment loss is severe.
- **Calibration definition** (Pearson vs Spearman, char aggregation) is our interpretation; kept
  configurable.
- **Cost of N× inference** on ~4.9k test × 4 languages — acceptable on A100s; tune N on dev.
- **Image extraction** (2.4 GB tar) is a required one-time step before any VLM run.
```
