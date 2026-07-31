#!/bin/bash
# Unified Visual Adaptation — LoRA half of the matrix (A2/A3) on ONE H100.
#
#   bash scripts/connector/run_lora_h100.sh [--gpu N] [--data-dir DIR] [--model M]
#
#   A2: --arch linear     LoRA backbone, answer states only
#   A3: --arch connector  LoRA backbone + visual-token cross-attention
#
# Live forward per step (no cache), batch 1 x accum 16, 1 decoder-warm-up epoch +
# 4 joint epochs. ~35 min/epoch => ~3 h per variant, ~6 h total. Decoders warm-start
# from the frozen-half checkpoints when present (A2 <- A0, A3 <- A1).
#
# LoRA default is --lora_from 24: the decoder taps layers {24,32,40,47}, and adapting
# only the top 8 (from 40) would leave the 24/32 taps frozen by construction. Pass
# LORA_FROM=40 to reproduce the plan's top-8-only configuration.
set -euo pipefail
cd "$(dirname "$0")/../.."

GPU="${GPU:-0}"
LORA_FROM="${LORA_FROM:-24}"
DD_SET=0; M_SET=0
DATA_DIR="${DATA_DIR:-../Shroom-Vision}"
MODEL="${MODEL:-google/gemma-4-12B-it}"
POS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --gpu) GPU="$2"; shift 2 ;;
    --data-dir) DATA_DIR="$2"; DD_SET=1; shift 2 ;;
    --model) MODEL="$2"; M_SET=1; shift 2 ;;
    --lora-from) LORA_FROM="$2"; shift 2 ;;
    -h|--help) sed -n '2,16p' "$0"; exit 0 ;;
    -*) echo "unknown option: $1 (try --help)" >&2; exit 2 ;;
    *) POS+=("$1"); shift ;;
  esac
done
if [ "${#POS[@]}" -ge 1 ]; then
  if [ "$DD_SET" -eq 0 ] && [ -d "${POS[0]}/distrib" ]; then DATA_DIR="${POS[0]}"
  elif [ "$M_SET" -eq 0 ]; then MODEL="${POS[0]}"
  else echo "warn: ignoring positional '${POS[0]}' (flags already set)" >&2; fi
fi
export CUDA_VISIBLE_DEVICES="$GPU"
echo "[gpu ] CUDA_VISIBLE_DEVICES=$GPU -> cuda:0"

T="$DATA_DIR/distrib/shroom-vision.train.en.labeled.jsonl"
IMG="$DATA_DIR/images"
P=splits/en.eval_protocol.json
OUT=results/lora_h100
[ -f "$T" ] || { echo "ERROR: $T missing" >&2; exit 1; }

source scripts/connector/_env.sh
setup_cluster_env
mkdir -p "$OUT"

TRAIN="python scripts/connector/train_lora.py --model_id $MODEL --train_file $T \
  --image_dir $IMG --eval_ids $P --device cuda:0 --seed 13 --lora_from $LORA_FROM \
  --epochs 5 --freeze_lora_epochs 1 --accum 16"

A0_CK=results/v3_h100/best_iou_linear_bio_tv_bio_gc_ctr_notype_s13.pt
A1_CK=results/uva_h100/best_iou_connector_bio_tv_bio_gc_ctr_notype_s13.pt
A2_INIT=""; A3_INIT=""
[ -f "$A0_CK" ] && A2_INIT="--init_from $A0_CK"
[ -f "$A1_CK" ] && A3_INIT="--init_from $A1_CK"

echo "=== A2: LoRA + H-only (warm start: ${A2_INIT:-none}) ==="
$TRAIN --arch linear --out_dir "$OUT/a2" $A2_INIT

echo "=== A3: LoRA + visual memory (warm start: ${A3_INIT:-none}) ==="
$TRAIN --arch connector --out_dir "$OUT/a3" $A3_INIT

echo "=== summary (matrix so far) ==="
python - <<'PY'
import json, glob
print(f"{'variant':52}{'iou':>7}{'dirty':>7}{'clnOK':>6}{'corR':>7}")
for pat, hdr in [("results/v3_h100/summary_linear_bio*_s13.json", "A0 (frozen, H-only)"),
                 ("results/uva_h100/summary_connector*_s13.json", "A1 (frozen, +V)"),
                 ("results/lora_h100/a2/summary_*.json", "A2 (LoRA, H-only)"),
                 ("results/lora_h100/a3/summary_*.json", "A3 (LoRA, +V)")]:
    files = sorted(glob.glob(pat))
    if files:
        print(f"--- {hdr} ---")
    for f in files:
        d = json.load(open(f)); m = d["metrics"]
        print(f"{d['variant']:52}{m['span_iou']:>7.4f}{m['dirty_iou']:>7.3f}"
              f"{m['clean_empty']:>6.2f}{m['pooled_pearson_debug']:>7.3f}")
print("""
factorial read-out: A2-A0 = LoRA gain | A1-A0 = visual-memory gain
                    A3-A2 = memory after adaptation | A3-A1 = adaptation given memory
NB: A2/A3 train without the contrastive term (batch=1) — the tightest frozen
    baseline for them is the minus-contrastive ablation in results/v3_h100.
""")
PY
echo "done -> $OUT"
