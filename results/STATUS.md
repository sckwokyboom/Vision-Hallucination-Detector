# Project status — SHROOM-visions 2026 (en, tune-202, internal scorer, floor 0.2129)

Updated: 2026-07-30. Held-out-157 untouched. All numbers seed 13 unless noted.

## The ladder

| system | iou | dirty | cleanOK | note |
|---|---:|---:|---:|---|
| predict-nothing floor        | 0.2129 | 0     | 1.00 | |
| starter (Qwen2-VL-2B)        | 0.2005 | —     | —    | below floor |
| zero-shot HYBRID             | 0.273  | —     | —    | claim-gate + extraction |
| Mac 4-bit v1 / v3-BIO        | 0.304 / 0.320 | | | |
| A0: H100 bf16 v3-BIO (3 seeds) | 0.3519/0.3466/0.3464 | 0.195–0.295 | 0.53–0.93 | operating point unstable |
| A2: + LoRA(24–47), cold, 5 ep | **0.3925** best | 0.322 | 0.65 | biggest single jump |
| V4.1: dirty-only locator + weak gate | 0.3532 | 0.318 ungated | 0.65 | **gold-gate ceiling 0.4631** |

## What we established (and how)

1. **Representation, not decode.** With H100 features the 6-family decoder sweep
   collapsed (minus-bio 0.3504 ~ full 0.3519). Decoder work is closed.
2. **Direct visual memory is dead, twice.** Frozen: A1 = A0, ReZero gate never opened
   (eval-shuffle identical to 1e-8; instrument verified both ways on CPU). Adapted:
   A3 < A2. Everything the decoder uses flows through answer states H.
3. **LoRA moved the ceiling** (+0.04, single seed, cold start — the warm-start
   confound never actually existed). a2_frozen control pending to isolate it from
   live-pipeline differences.
4. **Splitting clean/dirty objectives moved localization as much as LoRA did**:
   dirty 0.195 -> 0.318 on the SAME frozen cache, purely from training the locator
   on dirty answers only. The two gains came by different mechanisms — possibly additive.
5. **The bottleneck is now the gate**: cascade 0.3532 real vs 0.4631 with a perfect
   gate. A dedicated focal/balanced gate reached only AUC 0.794 with no usable
   high-recall operating point.
6. Quantization is a non-issue (4bit ~ 8bit ~ bf16, paired). Type head (Cor_lbl) is
   dead everywhere — two-stage training still pending. Zero-shot grounding was real
   but partial (0.694/0.629/0.566); for the trained decoder the no-image control is
   still pending.

## In flight / queued

1. A0 rerun (restores joint ckpt lost to the pre-sync trainer; prints joint gate AUC).
2. Cascade re-score with the A0 joint gate (--epochs 0, 2 min).
3. a2_frozen (cold, LoRA frozen): isolates the LoRA effect. Decides the backbone for V4.
4. A2 seeds 42/77 overnight.
5. Parked: probe_vbranch, no-image control, official re-scoring of best ckpts.

## Where the next IoU comes from (expected value, order)

1. **Gate v2: up to +0.11** (the ceiling gap). First free shot: A0's joint gate.
   Then: gate on locator statistics (top-k probs, peak counts) — needs out-of-fold
   care because the locator saw the dirty train items.
2. **LoRA x dirty-only: dirty 0.318 -> 0.36–0.40 if additive** (one flag in
   train_lora once a2_frozen confirms LoRA).
3. **V4.2 candidate ranker** over BIO proposals (SegmentScorer+DP already exist).
4. **fr/it/zh**: ranked separately and currently have NO system at all; extraction
   + the same recipe is mostly compute, and multilingual training may help en too.

## Honest risks

- Operating points are tuned on tune-202; cascade trades cleanOK (0.93 -> 0.65) for
  dirty — if the closed test has a higher clean share, this hurts. The g-threshold
  transfer is the single biggest generalization risk.
- Best-checkpoint numbers carry tune-202 selection bias; expect the held-out to land
  a few points lower.
- All V4/A2 numbers are single-seed until tonight.
- Moving to the LoRA backbone multiplies iteration cost ~20x; only worth it if
  a2_frozen confirms the effect.
- Cor is healthy (0.44 vs 0.213 floor) but Cor_lbl is below floor — the type head
  must be trained (two-stage) before any submission.
