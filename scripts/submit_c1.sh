#!/bin/bash
# Submit C1 cascade LoRA results for English.
#
#   bash scripts/submit_c1.sh
#
# Uses the best checkpoint from C1, predicts on test, postprocesses,
# format-checks, and writes the final submission to results/cascade_h100/c1/submit_en.jsonl
set -euo pipefail
cd "$(dirname "$0")/.."

CKPT="results/cascade_h100/c1/best_iou_lora_linear_cascade_f24_r16_s13.pt"
TEST="${TEST_FILE:-/userspace/srm/text_dataset/shroom-vision.test.en.unlabeled.jsonl}"
IMAGE_DIR="${IMAGE_DIR:-/userspace/srm/shroom-vis-images}"
MODEL="${MODEL:-google/gemma-4-12B-it}"
DEVICE="${DEVICE:-cuda:0}"
OUTDIR="results/cascade_h100/c1"

[ -f "$CKPT" ] || { echo "ERROR: checkpoint $CKPT not found" >&2; exit 1; }
[ -f "$TEST" ] || { echo "ERROR: test file $TEST not found" >&2; exit 1; }
[ -d "$IMAGE_DIR" ] || { echo "ERROR: image dir $IMAGE_DIR not found" >&2; exit 1; }

mkdir -p "$OUTDIR"

echo "=== Step 1: predict on test set ==="
python scripts/connector/predict_lora.py \
    --checkpoint "$CKPT" \
    --model_id "$MODEL" \
    --input "$TEST" \
    --image_dir "$IMAGE_DIR" \
    --output "$OUTDIR/test_pred_raw.jsonl" \
    --decoder bio \
    --device "$DEVICE"

echo ""
echo "=== Step 2: postprocess (trim edges, filter, force invention label) ==="
python scripts/official/postprocess_submission.py \
    --pred "$OUTDIR/test_pred_raw.jsonl" \
    --items "$TEST" \
    --out "$OUTDIR/submit_en.jsonl" \
    --trim_edges \
    --min_len 4 \
    --min_prob 0.20 \
    --label_policy constant \
    --default_label invention

echo ""
echo "=== Step 3: format check ==="
python scripts/official/format_checker.py \
    "$OUTDIR/submit_en.jsonl" \
    --reference-file "$TEST"

echo ""
echo "=== done ==="
echo "submit file: $OUTDIR/submit_en.jsonl"
wc -l "$OUTDIR/submit_en.jsonl"
cksum "$OUTDIR/submit_en.jsonl"
