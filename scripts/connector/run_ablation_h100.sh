#!/bin/bash
# Camera-ready ablations on ONE GPU (reviewer requests): single-copy control,
# eval-time image controls (shuffled / no-image), raw dumps for the offline sweeps.
#
#   bash scripts/connector/run_ablation_h100.sh --phase single-copy-train [--gpu 0]
#   bash scripts/connector/run_ablation_h100.sh --phase eval-final-en --ck results/lora_h100/a2/best_iou_lora_linear_f24_r16_s13.pt
#   bash scripts/connector/run_ablation_h100.sh --phase eval-single-en
#   bash scripts/connector/run_ablation_h100.sh --phase eval-lang --lang zh --ck <zh ckpt> --tau 0.0 --g 0.7
#
# Every phase is resumable/re-runnable; outputs land under results/ablation/.
# Offline analysis afterwards (CPU, seconds, can run anywhere with the dumps):
#   python scripts/official/ablation_sweeps.py --raw_dir results/ablation/en_tune \
#       --gold splits/dev.en.jsonl --min_len 3 --merge_gap 2
set -euo pipefail
cd "$(dirname "$0")/../.."

GPU="${GPU:-0}"
PHASE="${PHASE:-}"
LANG_="${LANG_:-en}"
DATA_DIR="${DATA_DIR:-../Shroom-Vision}"
MODEL="${MODEL:-/workspace/data/models/gemma-4-12B-it}"
CK="${CK:-}"
TAU="${TAU:-}"
GTHR="${GTHR:-}"
SPLIT_ARG=""
LIMIT="${LIMIT:-}"

while [ $# -gt 0 ]; do
  case "$1" in
    --gpu) GPU="$2"; shift 2 ;;
    --phase) PHASE="$2"; shift 2 ;;
    --lang) LANG_="$2"; shift 2 ;;
    --data-dir) DATA_DIR="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --ck) CK="$2"; shift 2 ;;
    --tau) TAU="$2"; shift 2 ;;
    --g) GTHR="$2"; shift 2 ;;
    --limit) LIMIT="$2"; shift 2 ;;
    -h|--help) sed -n '2,15p' "$0"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done
[ -n "$PHASE" ] || { echo "ERROR: --phase required" >&2; exit 2; }

export CUDA_VISIBLE_DEVICES="$GPU"
DEVICE=cuda:0
TRAIN_EN="$DATA_DIR/distrib/shroom-vision.train.en.labeled.jsonl"
IMG_DIR="$DATA_DIR/images"
PROTO=splits/en.eval_protocol.json
OUT=results/ablation
SINGLE_CK="$OUT/single_copy_en/best_iou_lora_linear_single_f24_r16_s13.pt"
mkdir -p "$OUT/logs"

source scripts/connector/_env.sh
setup_cluster_env
[ -f splits/dev.en.jsonl ] || python -m shroom.make_split --distrib "$DATA_DIR/distrib" --out_dir splits

thr_args() {
  local a=()
  [ -n "$TAU" ] && a+=(--tau "$TAU")
  [ -n "$GTHR" ] && a+=(--gate_threshold "$GTHR")
  [ -n "$LIMIT" ] && a+=(--limit "$LIMIT")
  echo ${a[@]+"${a[@]}"}
}

run_logged() {
  local log="$1"; shift
  echo "=== $* ==="
  "$@" 2>&1 | tee "$log"
}

case "$PHASE" in
  single-copy-train)
    # Identical to the final en run (cold start, seed 13, 5 epochs, LoRA 24-47) except
    # for --single_copy. Best epoch by tune-202 IoU, like every other ablation row.
    run_logged "$OUT/logs/single_copy_train_en.log" \
      python scripts/connector/train_lora.py \
        --model_id "$MODEL" --train_file "$TRAIN_EN" --image_dir "$IMG_DIR" \
        --eval_ids "$PROTO" --out_dir "$OUT/single_copy_en" \
        --arch linear --seed 13 --epochs 5 --lora_from 24 --single_copy --device "$DEVICE"
    ;;
  eval-final-en)
    [ -f "$CK" ] || { echo "ERROR: --ck <final en checkpoint> missing: $CK" >&2; exit 1; }
    for split in tune_dev heldout; do
      run_logged "$OUT/logs/eval_final_en_${split}.log" \
        python scripts/connector/eval_ablation.py \
          --checkpoint "$CK" --model_id "$MODEL" --gold splits/dev.en.jsonl \
          --eval_ids "$PROTO" --split "$split" --image_dir "$IMG_DIR" \
          --out_dir "$OUT/en_${split}" --conditions correct,shuffled,no_image \
          --device "$DEVICE" $(thr_args)
    done
    ;;
  eval-single-en)
    [ -f "${CK:-$SINGLE_CK}" ] || { echo "ERROR: single-copy checkpoint missing: ${CK:-$SINGLE_CK}" >&2; exit 1; }
    for split in tune_dev heldout; do
      run_logged "$OUT/logs/eval_single_en_${split}.log" \
        python scripts/connector/eval_ablation.py \
          --checkpoint "${CK:-$SINGLE_CK}" --model_id "$MODEL" --gold splits/dev.en.jsonl \
          --eval_ids "$PROTO" --split "$split" --image_dir "$IMG_DIR" \
          --out_dir "$OUT/single_en_${split}" --conditions correct \
          --device "$DEVICE" $(thr_args)
    done
    ;;
  eval-lang)
    [ -f "$CK" ] || { echo "ERROR: --ck <$LANG_ checkpoint> missing: $CK" >&2; exit 1; }
    [ -f "splits/dev.$LANG_.jsonl" ] || { echo "ERROR: splits/dev.$LANG_.jsonl missing" >&2; exit 1; }
    # fr/it/zh thresholds were swept on the ids listed in their protocol file; score the
    # same ids. Without a protocol file the whole hash-dev split is used.
    SPLIT_ARGS=(--split all)
    [ -f "splits/$LANG_.eval_protocol.json" ] && \
      SPLIT_ARGS=(--eval_ids "splits/$LANG_.eval_protocol.json" --split tune_dev)
    run_logged "$OUT/logs/eval_${LANG_}_dev.log" \
      python scripts/connector/eval_ablation.py \
        --checkpoint "$CK" --model_id "$MODEL" --gold "splits/dev.$LANG_.jsonl" \
        "${SPLIT_ARGS[@]}" --image_dir "$IMG_DIR" --out_dir "$OUT/${LANG_}_dev" \
        --conditions correct,shuffled,no_image --device "$DEVICE" $(thr_args)
    ;;
  *) echo "ERROR: unknown --phase $PHASE" >&2; exit 2 ;;
esac
echo "done: $PHASE -> $OUT"
