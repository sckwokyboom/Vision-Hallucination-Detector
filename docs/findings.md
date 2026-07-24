# Findings

An honest account of what we built and measured. All numbers are character-level, on a
frozen **image-grouped dev split** carved from the labeled English training data (the official
test set is closed). Few-shot examples, when used, are drawn from the training portion only.

> **Caveat on samples.** Two evaluation sets appear below. `en-dev-200` is a representative
> random sample (~21% clean, matching the natural rate). `strat-100` is a *stratified
> diagnostic* sample that over-samples rare hallucination types so per-class metrics are
> measurable — it is **not** representative for prevalence and should not be read as a
> generalization estimate. A held-out slice is kept untouched for a final evaluation.

---

## 1. The task has a strong "do-nothing" floor

Character-IoU scores `empty gold + empty pred = 1.0`, and 25–37% of answers are clean. So
**predicting "no hallucination" everywhere** scores IoU = the clean fraction:

| lang | en | fr | it | zh |
|------|----|----|----|----|
| predict-nothing IoU | 0.25 | 0.26 | 0.26 | 0.37 |

Two hallucination types dominate the data: `invention` + `mischaracterization` ≈ 85% of all
spans; `OCR`, `miscounting`, `other` are rare (≤ ~8% each). Any real system must clear the
predict-nothing floor on span-IoU — but note this floor is **only** strong for span-IoU. On
the probability-ranking metrics (`roc_auc`, `pr_auc`, `class_roc`) the trivial baselines score
0.5, since they carry no probability signal.

## 2. Evaluation harness

`shroom/` is model-free and unit-tested (39 tests). It provides: JSONL loading, a
deterministic **image-grouped** dev split (all questions about one image stay on the same
side, avoiding leakage), robust phrase→char-span alignment (exact / whitespace-and-markdown
normalized / fuzzy), self-consistency aggregation (N samples → per-char frequency → spans at
threshold τ), and the full official scorer (`scripts/official_metrics.py`).

## 3. Span-detection experiments (en-dev-200, floor 0.210)

`nX` = X-sample self-consistency; `n1` = greedy.

| system | span_iou | roc_auc | pr_auc | calib_corr | class_roc(macro) |
|--------|----------|---------|--------|------------|------------------|
| predict-nothing | 0.210 | 0.500 | 0.153 | 0.000 | 0.500 |
| predict-everything | 0.161 | 0.500 | 0.153 | 0.000 | 0.500 |
| starter (Qwen2-VL-2B, permissive prompt) | 0.199 | 0.498 | 0.153 | −0.020 | 0.505 |
| Qwen2.5-VL-7B `n5` | 0.205 | 0.484 | 0.151 | −0.010 | 0.516 |
| Gemma-3-12B `n1` | 0.191 | 0.576 | 0.180 | 0.125 | 0.553 |
| Gemma-3-12B `n5` | 0.189 | 0.604 | 0.193 | 0.142 | 0.555 |
| claim-level verification | 0.249 | 0.576 | 0.193 | 0.163 | 0.559 |
| HYBRID `n1` | 0.273 | 0.569 | 0.186 | 0.151 | 0.523 |
| **HYBRID `n5`** | **0.273** | 0.600 | **0.199** | **0.164** | 0.527 |

**Observations.**

- The **starter** (a 2B model with a permissive 3-category prompt) sits *at or below* the
  floor with no probability signal — it barely differs from predict-nothing. Two empirical
  fixes were needed to get any signal from small VLMs: (1) resize images (large images make
  Qwen2.5-VL emit degenerate `!!!!`); (2) a **skeptical** prompt (a permissive prompt makes
  even a 7–12B model answer `[]` everywhere).
- **Qwen is a poor detector** here (roc_auc < 0.5); **Gemma-3-12B** is the better base.
- **Self-consistency (`n5`)** lifts the ranking metrics (Gemma roc_auc 0.576 → 0.604) — graded
  per-char probabilities help everything AUC-shaped.
- **span-IoU vs the AUC metrics disagree.** Gemma-extraction has good char-level F1 but poor
  span-IoU because it fires on *every* clean item; IoU heavily rewards protecting clean items
  (predict `[]`), which char-F1 does not see.

### The hybrid

Use **claim-level verification as an item-level clean-gate** (predict `[]` when it flags no
claim) and take **extraction spans** on the flagged items. This is the only configuration
that clears the floor *and* stays calibrated: on `en-dev-200`, HYBRID-`n5` beats
predict-nothing by **+0.063 mean per-item IoU** (95% bootstrap CI **[+0.018, +0.106]**; better
on 64 items, worse on 9), while winning `pr_auc` and `calib_corr` and near-tying `roc_auc`.

### The ceiling is recall

On the dirty items, the extraction "locator" (Gemma-3-12B `n5`) has span-detection recall
**0.61** — it misses ~39% of gold spans entirely (and when it does hit one, it usually hits it
fully, so this is a *detection* limit, not a boundary limit). Character precision is ~0.24
(it over-fires). This is a model-capability ceiling; post-processing cannot manufacture recall
that the detector does not have.

## 4. A binary hallucination gate (does the model know *whether* something is wrong?)

Motivation: span-IoU is dominated by clean-item protection, so a reliable "is there any
hallucination?" gate would be valuable. We studied it carefully because naive gates are
deceptive on an imbalanced set.

### Naive yes/no gates barely discriminate

A single "is anything hallucinated?" question is near-useless: Qwen-7B's version has MCC ≈ 0
(random); Gemma-3-12B's fires `YES` on ~everything (specificity 0.10). A **claim-level** gate
(per-sentence verification) is the first with real signal.

### Multimodal few-shot gate (Gemma-4-12B)

We built a proper multimodal few-shot gate: 8 real dataset samples, **each as its own chat
turn with its own image** (manual per-turn interleaving — the high-level `apply_chat_template`
dumps all images at the end), assistant labels `NO` / `YES: <types>`, target thinking-primed.
Ablation on `strat-100` (greedy operating point; Wilson CIs, bootstrap MCC, Fisher *p*):

| variant | prec | rec | spec | NPV | MCC | MCC 95% CI | Fisher p |
|---------|------|-----|------|-----|-----|------------|----------|
| A · 0-shot, classify | 0.854 | 0.519 | 0.667 | 0.269 | 0.151 | [−0.03, +0.33] | 0.15 |
| B · 0-shot, no-classify | 0.895 | 0.430 | 0.810 | 0.274 | 0.201 | [+0.03, +0.35] | 0.05 |
| C · 8-shot, no-classify | 0.907 | 0.494 | 0.810 | 0.298 | **0.249** | [+0.07, +0.41] | **0.01** |
| D · 8-shot, classify | 0.836 | 0.646 | 0.524 | 0.282 | 0.141 | [−0.05, +0.33] | 0.21 |

Only the **no-classify** variants reach MCC significance; even the best has NPV ≈ 0.30 (the
`NO` branch is still ~70% dirty) and recall < 0.5. The naive gate does **not** solve the task.

### Continuous score + visual controls

Scoring the gate threshold-free via `s = logP(YES) − logP(NO)` at the first token (YES/NO are
single tokens; the full logprob vector is read from `stream_generate`) gives ROC-AUC and lets
us run clean controls on the best config (C):

| condition | ROC-AUC | PR-AUC | spec@recall≥0.9 |
|-----------|---------|--------|-----------------|
| correct image | 0.694 | 0.908 | 0.333 |
| shuffled image (within-set derangement) | 0.629 | 0.881 | 0.238 |
| no image (text-only) | 0.566 | 0.826 | 0.238 |

- The monotonic drop **correct > shuffled > text-only** means the correct image contributes a
  **real visual signal** (~+0.13 AUC over text-only) — but a linguistic floor (0.566 > 0.5)
  shows the answer text alone carries part of it. The signal is **partly visual, partly
  linguistic.**
- **Classify vs no-classify is a wash on ROC-AUC** (0.70 vs 0.69). The earlier greedy result
  that "classify hurts" was an *operating-point artifact* — classification just shifts the
  threshold. (A good reminder to compare curves, not greedy points.)

### Type classification (diagnostic, `strat-100`)

Per-type F1 over all items (a type predicted on a clean answer counts as a false positive):

| type | \|gold\| | precision | recall | F1 |
|------|----------|-----------|--------|-----|
| invention | 45 | 0.49 | 0.44 | 0.47 |
| mischaracterization | 53 | 0.82 | 0.26 | 0.40 |
| OCR | 16 | 1.00 | 0.12 | 0.22 |
| miscounting | 19 | 0.67 | 0.32 | 0.43 |
| other | 17 | — | 0.00 | 0.00 |

The classifier is **biased to `invention` as a catch-all**: when the true type is OCR it says
`invention` in 12/16 cases; `mischaracterization → invention` happens 23 times. `other` is
**never predicted** (by 0-shot or 8-shot) — a structural blind spot, not a missing-example
issue. Interestingly, **0-shot classifies types better than 8-shot** — the demonstrations
introduced an invention bias rather than helping.

## 5. Honest conclusions

1. The **evaluation harness and the honest metric suite are the durable deliverable** — they
   make every subsequent claim measurable and falsifiable.
2. On span-IoU, the **hybrid (claim clean-gate + extraction)** is the only zero-shot Mac
   configuration that significantly clears the predict-nothing floor, while staying calibrated.
3. Naive prompting — bigger prompts, few-shot, classification requests, even multimodal
   few-shot — yields only a **modest, partly-linguistic gate signal (ROC-AUC ≈ 0.69)**. The
   binding constraint is **detector recall / discrimination**, not the pipeline.
4. Method choices that *looked* decisive at a greedy threshold (e.g. "classification hurts")
   dissolve under threshold-free analysis. Confidence intervals and ROC curves matter on
   small samples.

## 6. Next steps

- Stronger / fine-tuned detectors (larger or MoE VLMs, or a supervised token classifier on
  the 15k labeled spans) — the recall lever.
- Confirm the best configurations on a **natural random sample** and evaluate once, at the
  end, on the held-out slice.
- All four languages (`en/fr/it/zh`), ranked separately per the task.
