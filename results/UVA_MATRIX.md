# Unified Visual Adaptation — factorial results (en, tune-202, seed 13)

2x2: backbone {frozen, LoRA} x decoder input {H-only, H+V}. Same v3-BIO decoder and
protocol everywhere; A2/A3 train without the contrastive term (live batch = 1), so
their tightest frozen baseline is the minus-contrastive ablation. All numbers are
internal scorer on the frozen tune-202 split; floor = 0.213.

| cell | variant | iou (final) | iou (best ep) | dirty | cleanOK | corR |
|------|---------|------------:|--------------:|------:|--------:|-----:|
| A0   | frozen, H-only (full v3)        | 0.3519 | 0.3519 (ep12) | 0.195 | 0.93 | 0.444 |
| A0'  | frozen, H-only, minus-contrast  | 0.3423 | —             | 0.240 | 0.72 | 0.437 |
| A1   | frozen, +V (connector)          | 0.3380 | 0.3585 (ep10) | 0.247 | 0.67 | 0.443 |
| A1c  | frozen, +V, shuffled-V control  | 0.3386 | —             | 0.242 | 0.70 | 0.411 |
| A2   | LoRA(24-47), H-only             | 0.3703 | **0.3925** (ep4) | 0.322 @best | 0.65 @best | 0.468 @best |
| A3   | LoRA(24-47), +V                 | 0.3630 | 0.3882 (ep3)  | 0.298 @best | 0.72 @best | 0.475 @best |

Factorial decomposition (best-checkpoint basis):

- **LoRA gain (A2−A0): +0.041** — the single biggest jump of the whole project
  (previous ladder: floor 0.213 -> zero-shot hybrid 0.273 -> v1 0.304 -> v3 0.352 -> 0.392).
- **Frozen visual memory (A1−A0): ~0** — and the eval-time shuffled-image control on A1
  was identical to GPU noise (1e-8), i.e. the trained ReZero gate never opened.
- **Visual memory after adaptation (A3−A2): −0.004** — nothing, even with a LoRA'd
  backbone and a warm-started connector decoder.
- **Adaptation given memory (A3−A1): +0.030** — consistent with the row above.

Read-out: everything the decoder uses flows through the answer-token states H; the
direct V route is dead in both halves of the matrix. The lever is task-adapting the
backbone, not widening the decoder's access to it.

Caveats before believing the +0.04:
1. Single seed; A0's seed spread was ~0.006, LoRA's is unknown -> seeds 42/77 pending.
2. Best-checkpoint numbers are selected on the same tune-202 they are reported on;
   the frozen held-out-157 stays untouched until the final shot.
3. Within-run instability is real (A2: 0.3925 at ep4 -> 0.3712 at ep5), so best-iou
   checkpointing (already in place) matters.
4. Not yet re-scored with the official scorer; internal iou has matched official
   closely so far but the final table must be official.
5. The lexical-vs-visual question (does ANY of this use the image, or is H itself
   mostly linguistic?) is NOT answered by this matrix — no-image control pending.
