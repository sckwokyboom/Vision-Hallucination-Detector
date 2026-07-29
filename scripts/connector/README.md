# Task-specific decoder over frozen Gemma 4 12B

Train a small span decoder on cached hidden states of a **frozen** Gemma 4 12B
(encoder-free `gemma4_unified`). Two stages: extract features once, then iterate on the
decoder in minutes. Works fully on Apple Silicon via MLX.

## Pipeline (Mac / MLX)

```bash
# 1) cache features (H of the review-copy answer; layers 24,32,40,47; ~2.7s/item on M4 Pro)
python scripts/connector/extract_features_mlx.py \
  --train_file ../Shroom-Vision/distrib/shroom-vision.train.en.labeled.jsonl \
  --image_dir ../Shroom-Vision/images --out_dir results/cache --h_only
#    --no_image   -> true text-only control cache
#    --probe 2    -> validate token alignment first
#    atomic writes; resumable (existing files skipped)

# 2) train / ablate the decoder (minutes per run on the cache)
python scripts/connector/train_connector.py \
  --train_file ../Shroom-Vision/distrib/shroom-vision.train.en.labeled.jsonl \
  --eval_ids splits/en.eval_protocol.json --cache_dir results/cache \
  --out_dir results/run --arch linear --seed 13 --epochs 12
#    --decoder simple|gate|hyst|gate_hyst   (span decoding ablations)
#    --no_gru | --max_train N | --eval_only --init_from model.pt

# full protocol (3 seeds + decoder ablations + learning curve + controls):
bash scripts/connector/run_fullscale.sh

# 3) manual error analysis: self-contained HTML (image + gold vs model highlighting)
python scripts/connector/make_inspection.py --gold <gold.jsonl> --pred <pred.jsonl> \
  --image_dir ../Shroom-Vision/images --out inspect.html
```

## Pipeline (CUDA / H100)

```bash
bash scripts/get_data.sh --data-dir /scratch/$USER/Shroom-Vision   # login node: needs network
bash scripts/connector/run_train_h100.sh --gpu 0 --data-dir /scratch/$USER/Shroom-Vision
```

`--gpu N` exports `CUDA_VISIBLE_DEVICES=N` **and** passes `--device cuda:0` down, so both
stages land on that one card — masking also pins accelerate/bitsandbytes, which `--device`
alone does not. `--gpu all` keeps the old sharding behaviour, `--dry-run` prints the resolved
plan and exits. The runner downloads the dataset itself if it is missing. Same flags on the
precision ladder (`run_quant_h100.sh --gpu 1`).

Budget ~15 GB of scratch for the bf16 feature cache (3.3 MB/item × ~3800 items) and keep it
off `$HOME`. Gemma 4 12B is license-gated on HF: either `huggingface-cli login` or pass a
local weights path via `--model`.

## Method

Prompt repeats the answer (`Review token by token:`) so every review token attends to the
full answer + image inside frozen Gemma; hidden states of the review copy are cached
(`.npz`: H [T,L,D], tok->char offsets). The decoder: learnable layer mix -> (linear readout
| cross-attention connector) -> char scatter + CNN + residual BiGRU -> heads: soft prob
(trained on annotator probabilities -> Cor), hard-span, 5 types, clean/dirty gate.
Decoding: gate + hysteresis thresholds, swept on the tune split.

## Evaluation protocol

`splits/en.eval_protocol.json` freezes an **image-disjoint** split of the dev set:
`tune_dev` (202 items) for all tuning, `heldout` (157) untouched for a single final run.
Confidence intervals should be cluster-bootstrapped by image.

## Status (en, tune-202, floor 0.213)

- Trained linear readout + gated hysteresis decoder: **span_iou 0.306 / 0.288 (two seeds),
  Cor_raw ~0.39-0.41** - vs the best zero-shot HYBRID (0.273 IoU / 0.164 Cor) and the
  predict-nothing floor (cluster-bootstrap CI [0.247, 0.362], P(>floor)=1.0).
- Oracle decomposition (same probability fields): per-item threshold 0.46, multi-span 0.71
  -> the dominant loss is thresholding + span structure, not representations.
- Known model issues: spans ~7x longer than gold (median 66 vs 9), type head collapses to
  `invention`, long answers dilute the signal. Next iteration: character-level multi-span
  boundary heads, precision-oriented (Tversky) loss, soft token-derived gate, contrastive
  pairs within same-image groups.
