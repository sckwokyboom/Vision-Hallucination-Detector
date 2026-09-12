# Camera-ready ablations (2026-09-12) — reviewer-requested controls

Official scorer everywhere; batch-1 inference through `predict_lora`/`eval_ablation`
(the submission path). "raw" = trainer decode at the submitted thresholds, "pp" = +
submission post-processing (trim, min_len 3, merge_gap 2). Single seed (13).
Dumps: `results/ablation/<split>/{correct,shuffled,no_image}.raw.jsonl` (+ `sweeps.json`,
`summary.json`); reproduce every number with
`python scripts/official/ablation_sweeps.py --raw_dir results/ablation/<split> --gold splits/dev.<lang>.jsonl [--transfer 0.0,0.5,3,2]`.

## 1. Echo prompt vs single copy (en, identical recipe, best epoch on tune-202)

| split | metric | echo (τ=0, g=0.5) | single copy (τ=0.4, g=0.4) | Δ |
|---|---|---:|---:|---:|
| tune-202 | IoU raw / pp | 0.4151 / 0.4221 | 0.3860 / 0.3867 | −0.029 / −0.035 |
| tune-202 | Cor raw / pp | 0.4426 / 0.4527 | 0.4104 / 0.3993 | −0.032 / −0.053 |
| tune-202 | dirty IoU (raw) | 0.326 | 0.283 | −0.043 |
| held-out-157 | IoU raw / pp | 0.3850 / 0.3810 | 0.3764 / 0.3753 | −0.009 / −0.006 |
| held-out-157 | Cor raw / pp | 0.4267 / 0.4226 | 0.4172 / 0.4077 | −0.010 / −0.015 |
| held-out-157 | dirty IoU (raw) | 0.294 | 0.235 | −0.059 |

Single-copy training log: `results/ablation/logs/single_copy_train_en.log` (epochs:
0.3253 → 0.3548 → … → best 0.3829 at ep 4). Checkpoint (not committed):
`results/ablation/single_copy_en/best_iou_lora_linear_single_f24_r16_s13.pt`.

## 2. Image dependence of the submitted checkpoints (pp, IoU / Cor)

| split | floor | correct | shuffled | ΔIoU [95% CI] | no image | ΔIoU [95% CI] |
|---|---:|---:|---:|---:|---:|---:|
| EN tune-202 | 0.213 | 0.422 / 0.453 | 0.183 / 0.130 | −0.239 [−0.311, −0.177] | 0.297 / 0.332 | −0.125 [−0.185, −0.068] |
| EN held-out-157 | 0.210 | 0.381 / 0.423 | 0.211 / 0.169 | −0.171 [−0.242, −0.111] | 0.271 / 0.311 | −0.110 [−0.161, −0.064] |
| FR dev | 0.297 | 0.450 / 0.437 | 0.224 / 0.143 | −0.225 [−0.273, −0.176] | 0.347 / 0.339 | −0.103 [−0.143, −0.060] |
| IT dev | 0.289 | 0.416 / 0.450 | 0.299 / 0.318 | −0.116 [−0.163, −0.072] | 0.366 / 0.403 | −0.049 [−0.082, −0.019] |
| ZH dev | 0.355 | 0.500 / 0.523 | 0.225 / 0.223 | −0.275 [−0.336, −0.214] | 0.398 / 0.414 | −0.102 [−0.145, −0.059] |

Shuffled images: clean answers left empty 0.02–0.14 (IT 0.40) vs 0.75–0.89 with the
correct image. Oracle re-sweep of (τ, g) on the degraded condition recovers only part
of the loss (EN tune-202 no-image: 0.325 vs 0.422).

## 3. Post-processing sensitivity (pp IoU at merge_gap 2; min_len 1 … 5)

| split | 1 | 2 | 3 (submitted) | 4 | 5 | merge_gap range @3 |
|---|---:|---:|---:|---:|---:|---:|
| EN tune-202 | 0.416 | 0.421 | **0.422** | 0.405 | 0.397 | 0.0005 |
| EN held-out-157 | 0.386 | **0.388** | 0.381 | 0.373 | 0.364 | 0.0007 |
| FR dev | 0.448 | 0.449 | 0.450 | **0.451** | 0.440 | 0.0003 |
| IT dev | 0.418 | **0.419** | 0.416 | 0.415 | 0.404 | 0.0011 |
| ZH dev | **0.515** | 0.513 | 0.500 | 0.477 | 0.458 | 0.0020 |

ZH: min_len 3 costs 0.015 IoU / 0.020 Cor vs min_len 1 — the reviewers' concern holds.

## 4. Threshold transfer (fr/it/zh scored with en's operating point τ=0, g=0.5, min_len 3, gap 2)

| track | own | transferred | ΔIoU [95% CI] |
|---|---:|---:|---:|
| FR | 0.4495 | 0.4179 | −0.032 [−0.058, −0.008] |
| IT | 0.4158 | 0.4143 | −0.002 [−0.024, +0.019] |
| ZH | 0.4996 | 0.4853 | −0.014 [−0.039, +0.010] |

## Note on fr/it dev numbers vs SYSTEM_DESCRIPTION.md

`SYSTEM_DESCRIPTION.md` §10 quotes trainer-time dev IoU for fr/it (0.4513 / 0.4335)
from the batched (16, padded) trainer evaluation; the batch-1 submission path used
here gives 0.4495 / 0.4158 at the same thresholds. All camera-ready tables use the
batch-1 path consistently, which is also the path the test submissions went through.
