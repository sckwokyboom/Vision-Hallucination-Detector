# Working results log (durable)

Raw numbers from the investigation, kept in git so they survive `/tmp` wipes / restarts.
All character-level, English dev (image-grouped split from labeled train). Two samples:
`en-dev-200` (representative, ~21% clean) and `strat-100` (stratified diagnostic — rare types
over-sampled, NOT representative).

## 1. Span detection (en-dev-200, floor 0.210)

| system | span_iou | roc_auc | pr_auc | calib | class_roc(macro) |
|---|---|---|---|---|---|
| predict-nothing | 0.210 | 0.500 | 0.153 | 0.000 | 0.500 |
| predict-everything | 0.161 | 0.500 | 0.153 | 0.000 | 0.500 |
| starter (Qwen2-VL-2B, permissive) | 0.199 | 0.498 | 0.153 | −0.020 | 0.505 |
| Qwen2.5-VL-7B n5 | 0.205 | 0.484 | 0.151 | −0.010 | 0.516 |
| Gemma-3-12B n1 | 0.191 | 0.576 | 0.180 | 0.125 | 0.553 |
| Gemma-3-12B n5 | 0.189 | 0.604 | 0.193 | 0.142 | 0.555 |
| claim-level | 0.249 | 0.576 | 0.193 | 0.163 | 0.559 |
| HYBRID n1 | 0.273 | 0.569 | 0.186 | 0.151 | 0.523 |
| HYBRID n5 | 0.273 | 0.600 | 0.199 | 0.164 | 0.527 |

HYBRID-n5 vs predict-nothing: +0.063 mean per-item IoU, 95% CI [+0.018, +0.106] (better on 64
items, worse on 9). Locator (Gemma-3-12B n5) on dirty: span-detection recall 0.61 (~39% missed).

## 2. Gate ablation — Gemma-4-12B, strat-100 (greedy point; Wilson CI, bootstrap MCC, Fisher p)

| variant | prec | rec | spec | NPV | MCC | MCC 95% CI | Fisher p |
|---|---|---|---|---|---|---|---|
| A 0-shot classify | 0.854 | 0.519 | 0.667 | 0.269 | 0.151 | [−0.03,+0.33] | 0.15 |
| B 0-shot no-classify | 0.895 | 0.430 | 0.810 | 0.274 | 0.201 | [+0.03,+0.35] | 0.05 |
| C 8-shot no-classify | 0.907 | 0.494 | 0.810 | 0.298 | 0.249 | [+0.07,+0.41] | 0.01 |
| D 8-shot classify | 0.836 | 0.646 | 0.524 | 0.282 | 0.141 | [−0.05,+0.33] | 0.21 |

Only no-classify variants reach MCC significance. NPV ≈ 0.30 (NO branch ~70% dirty).

## 3. Gate — continuous score logP(YES)−logP(NO), strat-100

| condition (Gemma-4-12B) | ROC-AUC | PR-AUC | sp@rec≥.9 | sp@rec≥.8 |
|---|---|---|---|---|
| C-mm (correct img) | 0.694 | 0.908 | 0.333 | 0.381 |
| C-shuffle_perm | 0.629 | 0.881 | 0.238 | 0.238 |
| C-textonly | 0.566 | 0.826 | 0.238 | 0.238 |
| D-mm (classify) | 0.702 | 0.910 | 0.333 | 0.381 |

Monotonic mm>shuffle>textonly ⇒ signal partly visual (correct image +~0.13 AUC over text-only),
partly linguistic (textonly 0.566 > 0.5). classify≈no-classify on ROC (0.70 vs 0.69): the greedy
"classify hurts" was a threshold artifact.

## 4. Type classification — Gemma-4-12B D (per-type over all 100)

| type | \|gold\| | prec | rec | F1 |
|---|---|---|---|---|
| invention | 45 | 0.49 | 0.44 | 0.47 |
| mischaracterization | 53 | 0.82 | 0.26 | 0.40 |
| OCR | 16 | 1.00 | 0.12 | 0.22 |
| miscounting | 19 | 0.67 | 0.32 | 0.43 |
| other | 17 | — | 0.00 | 0.00 |

Biased to `invention` as catch-all (OCR→invention 12/16, mischar→invention 23×). `other` never
predicted (0-shot or 8-shot). 0-shot classifies types better than 8-shot.

## 5. Capacity: Gemma-4-26B (MoE) vs 12B — gate C, strat-100

id `mlx-community/gemma-4-26b-a4b-it-4bit`, ~15.3 GB, ~11 s/item on M4 Pro.

| | ROC-AUC | PR-AUC | sp@rec≥.8 | sp@rec≥.9 | sp@rec≥.95 |
|---|---|---|---|---|---|
| 26B C-mm | 0.738 | 0.921 | 0.476 | 0.190 | 0.190 |
| 12B C-mm | 0.694 | 0.908 | 0.381 | 0.333 | 0.286 |

Paired bootstrap ΔAUC = **+0.044, 95% CI [−0.033, +0.123], P(ΔAUC>0)=0.865** → NOT significant.
26B better at recall≥0.8 but WORSE at safety recall ≥0.9/0.95 (ROC curves cross): 26B assigns a
few dirty items extreme-negative scores (−25) below the most-negative clean (−24.7) → confident
false negatives collapse high-recall specificity. Greedy points and score-gap not cross-model
comparable.

**Honest status:** capacity is *evidence in favor* (86%), not confirmed.

### 5b. 26B visual controls (ROC-AUC, strat-100)

| condition | 26B | 12B |
|---|---|---|
| correct image | 0.738 | 0.694 |
| shuffled image | 0.524 (≈chance) | 0.629 |
| text-only | 0.610 | 0.566 |

- **Visual lift (correct − text-only) is identical: +0.128 for both** ⇒ the larger model did NOT
  add visual grounding; the image contributes the same ~0.13 AUC regardless of size.
- 26B's overall edge (+0.044) comes entirely from a **higher text-only floor** (0.610 vs 0.566) —
  more capacity improved *linguistic* reasoning, not *visual* discrimination.
- A **wrong** image collapses 26B to chance (0.524) but not 12B (0.629): 26B conditions on the
  image more genuinely; 12B leans more on text.
- Caveat: n=100 (21 clean), wide CIs — suggestive, not significant.

**Takeaway:** the bottleneck is *visual discrimination*, and it did not improve with model size —
"needs more cognition" is only half-right. The lever is stronger visual grounding (task
fine-tuning / a stronger vision stack), not more text capacity alone.

## Reproduce

Score files are in this folder (`score_*.jsonl`: id, gold_dirty, score, first_token, argmax_yes).
Ablation prediction files: `{A,B,C,D,F}_*.jsonl`. Sample: `stratified_100.jsonl`.
Scorers: `scripts/mm_score_eval.py` (ROC/PR/spec@recall), `scripts/mm_gate_eval.py` (binary+type).

## 6. Connector smoke (frozen Gemma-4-12b features, Mac MLX) — SIGNAL CONFIRMED

Cache: 350 items (266 visual tokens @768px, layers {24,32,40,47}, ~3s/item). Train 319 / dev 31
(image-grouped). 12 epochs, spans from the soft-prob head.

| variant | span_iou | floor | roc_auc | calib | calib_lbl |
|---|---|---|---|---|---|
| linear probe, with-image H | **0.3267** | 0.3226 | **0.684** | **0.355** | **0.305** |
| linear probe, text-only H (true control: separate no-image extraction) | 0.3257 | 0.3226 | 0.677 | 0.276 | 0.219 |
| cross-attn connector, with-image | 0.3226 | 0.3226 | 0.674 | 0.258 | 0.197 |

Findings: (1) training on frozen-backbone features works — calib 0.355 from just 319 examples,
~2x the best zero-shot; (2) the visual contribution flows through the backbone's own fusion into H
(with-image > text-only on every metric; the earlier "no_image" control that only zeroed our V copy
was invalid — H had already seen the image inside Gemma); (3) cross-attention over V adds nothing at
this scale — a simple linear readout of H wins. Next: full 3.4k cache + longer training.
Caveat: dev=31 — directional, not significant.
