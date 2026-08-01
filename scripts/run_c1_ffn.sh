#!/bin/bash
# C1 + FFN: cascade LoRA с адаптерами на self-attention + FFN (gate/up/down).
#
#   bash scripts/run_c1_ffn.sh --gpu N
#
# Отличается от оригинального C1 только LoRA-таргетами (добавлен FFN).
# Остальное идентично: cascade-лосс, 5 эпох, warm-start из A0 frozen.
set -euo pipefail
cd "$(dirname "$0")/.."

GPU="${GPU:-0}"
TRAIN_FILE="${TRAIN_FILE:-/userspace/srm/text_dataset/shroom-vision.train.en.labeled.jsonl}"
IMAGE_DIR="${IMAGE_DIR:-/userspace/srm/shroom-vis-images}"
MODEL="${MODEL:-google/gemma-4-12B-it}"
OUT=results/cascade_h100/c1_ffn
EVAL_IDS=splits/en.eval_protocol.json

while [ $# -gt 0 ]; do
  case "$1" in
    --gpu) GPU="$2"; shift 2 ;;
    -h|--help) sed -n '2,6p' "$0"; exit 0 ;;
    *) echo "unknown: $1" >&2; exit 2 ;;
  esac
done

export CUDA_VISIBLE_DEVICES="$GPU"
echo "[run ] GPU=$GPU | train=$TRAIN_FILE | images=$IMAGE_DIR"

A0_CK=results/v3_h100/best_iou_linear_bio_tv_bio_gc_ctr_notype_s13.pt
DEC_INIT=""
[ -f "$A0_CK" ] && DEC_INIT="--init_from $A0_CK" && echo "[run ] warm-start from A0"

mkdir -p "$OUT"

python scripts/connector/train_lora.py --model_id "$MODEL" \
    --train_file "$TRAIN_FILE" --image_dir "$IMAGE_DIR" --eval_ids "$EVAL_IDS" \
    --device cuda:0 --seed 13 --arch linear --cascade_train \
    --epochs 5 --freeze_lora_epochs 1 --accum 16 \
    --lambda_gate 1.0 --focal_gamma 2.0 --l_bio 0.5 \
    --lora_from 24 --out_dir "$OUT" $DEC_INIT

echo "done -> $OUT"
