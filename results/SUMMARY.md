# Full experimental summary (all phases)

> **For a head-to-head table, read [COMPARISON.md](COMPARISON.md) instead.** It re-scores the
> SHROOM starter baseline and every trained system on ONE eval set (tune_dev-202) with ONE scorer,
> with paired bootstrap CIs against the floor. This file is the phase-by-phase history.

Three eval samples (NOT directly comparable): en-dev-200 (representative, floor 0.210),
strat-100 (stratified diagnostic, floor 0.210), conn-dev-31 (connector smoke, floor 0.323).

## I. Span detection, zero-shot (en-dev-200)
| system | span_iou | roc_auc | pr_auc | calib | class_roc |
|---|---|---|---|---|---|
| predict-nothing | 0.210 | 0.500 | 0.153 | 0.000 | 0.500 |
| predict-everything | 0.161 | 0.500 | 0.153 | 0.000 | 0.500 |
| starter (Qwen2-VL-2B) | 0.199 | 0.498 | 0.153 | -0.020 | 0.505 |
| Qwen2.5-VL-7B n5 | 0.205 | 0.484 | 0.151 | -0.010 | 0.516 |
| Gemma-3-12B n1 | 0.191 | 0.576 | 0.180 | 0.125 | 0.553 |
| Gemma-3-12B n5 | 0.189 | 0.604 | 0.193 | 0.142 | 0.555 |
| claim-level | 0.249 | 0.576 | 0.193 | 0.163 | 0.559 |
| HYBRID n1 | 0.273 | 0.569 | 0.186 | 0.151 | 0.523 |
| **HYBRID n5 (best zero-shot)** | **0.273** | 0.600 | 0.199 | 0.164 | 0.527 |

HYBRID vs floor: +0.063, 95% CI [+0.018,+0.106] (significant). Zero-shot ceiling = locator
span-detection recall 0.61. Required fixes: image resize (else "!!!!"), skeptical prompt (else []).

## II. Binary gate (strat-100)
Greedy (Gemma-4-12B mm few-shot): best = 8-shot no-classify, MCC 0.249 [0.07,0.41], p=0.01;
naive yes/no gates ~random (Qwen MCC 0.001) or fire-always (Gemma-3 spec 0.095).
Score-based (logP(YES)-logP(NO)) ROC-AUC: 12B correct 0.694 / shuffled 0.629 / text 0.566;
26B-MoE: 0.738 / 0.524 / 0.610. Visual lift +0.128 IDENTICAL for both sizes; 26B edge is
linguistic-only and not significant (dAUC CI [-0.033,+0.123]); 26B worse at recall>=0.9.
As a safety gate: weak everywhere (NPV~0.28, spec@rec0.9<=0.33).

## III. Type classification (strat-100)
All-into-invention bias (OCR->invention 12/16); `other` never predicted (it's generation
garbage - regexable); few-shot HURT type F1 vs 0-shot; best per-type F1 ~0.43-0.49.

## IV. Trained readout on frozen Gemma-4-12B features (conn-dev-31, SMOKE)
| variant | span_iou | roc | calib | calib_lbl |
|---|---|---|---|---|
| linear readout, with-image H | **0.327** (floor 0.323) | **0.684** | **0.355** | **0.305** |
| linear readout, text-only H (true control) | 0.326 | 0.677 | 0.276 | 0.219 |
| cross-attn connector | 0.323 | 0.674 | 0.258 | 0.197 |

calib 0.355 from 319 examples (~2x best zero-shot). Visual contribution flows through the
backbone's own fusion (with-image > text-only everywhere). Cross-attention over V unnecessary
at smoke scale. Caveat: dev=31, directional.

## V. Lessons
1. Floor is strong only for span_iou; AUC metrics have no trivial floor.
2. Starter was indistinguishable from nothing.
3. Zero-shot prompting plateaued at HYBRID 0.273/0.164; ceiling = detector recall.
4. Model size improved text, NOT vision (visual lift +0.128 at both 12B and 26B).
5. Compare curves + CIs, not greedy points (several early conclusions died under honest analysis).
6. Training on frozen features is the most promising path; infra ready, signal confirmed.
