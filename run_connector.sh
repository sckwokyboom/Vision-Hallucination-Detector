#!/bin/bash
# Task-specific connector over frozen Gemma 4 12B — full experiment grid.
#
#   bash run_connector.sh [DATA_DIR] [MODEL_ID]
#
# Stage 1 (expensive, once): cache frozen-Gemma features (V at visual tokens, H of the
# review-copy answer) for train+dev. Stage 2 (minutes each): train connector variants.
#
# Quick-signal path: stage 1 on a subset first (--max_items 1200) ~ tens of minutes on
# one A100; the full cache can be resumed later (already-cached items are skipped).
set -euo pipefail
cd "$(dirname "$0")"

DATA_DIR="${1:-../Shroom-Vision}"
MODEL_ID="${2:-google/gemma-4-12B-it}"
TRAIN_FILE="$DATA_DIR/distrib/shroom-vision.train.en.labeled.jsonl"
IMG_DIR="$DATA_DIR/images"
CACHE="results/connector_cache"
OUT="results/connector"
[ -f "$TRAIN_FILE" ] || { echo "ERROR: $TRAIN_FILE not found"; exit 1; }

source .venv-cluster/bin/activate 2>/dev/null || {
  python3 -m venv .venv-cluster && source .venv-cluster/bin/activate
  pip install -q --upgrade pip && pip install -q -r requirements/cluster.txt accelerate
}
python -c "import torch; assert torch.cuda.is_available()" || { echo "no CUDA"; exit 1; }

if [ ! -d "$IMG_DIR" ] || [ -z "$(ls -A "$IMG_DIR" 2>/dev/null)" ]; then
  bash scripts/extract_images.sh "$DATA_DIR/shroom-visions-images.tar.gz" "$IMG_DIR"
fi

echo "=== STAGE 1a: probe (validate token counts / alignment) ==="
python scripts/connector/extract_features.py --model_id "$MODEL_ID" \
  --train_file "$TRAIN_FILE" --image_dir "$IMG_DIR" --out_dir "$CACHE" --probe 3

echo "=== STAGE 1b: cache features (subset first for quick signal) ==="
python scripts/connector/extract_features.py --model_id "$MODEL_ID" \
  --train_file "$TRAIN_FILE" --image_dir "$IMG_DIR" --out_dir "$CACHE" --max_items 1200

echo "=== STAGE 2: experiment grid (each run takes minutes on cached features) ==="
COMMON="--train_file $TRAIN_FILE --cache_dir $CACHE --out_dir $OUT"
echo "--- (1) frozen linear readout baseline ---"
python scripts/connector/train_connector.py $COMMON --arch linear
echo "--- (2) MAIN: cross-attention connector (+ shuffled-image eval control) ---"
python scripts/connector/train_connector.py $COMMON --arch connector --eval_shuffle
echo "--- (3) text-only control ---"
python scripts/connector/train_connector.py $COMMON --arch connector --no_image

echo ""
echo "=== SUMMARIES ==="
for f in "$OUT"/summary_*.json; do echo "--- $f"; cat "$f"; echo; done
echo "SUCCESS CRITERIA: connector > linear on span_iou/calib/calib_lbl; connector >"
echo "no_image and > shuffled-image (visual grounding); span_iou > floor."
echo "Next: rerun stage 1b WITHOUT --max_items to cache the full train set, then rerun stage 2."
