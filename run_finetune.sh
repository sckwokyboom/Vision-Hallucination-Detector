#!/bin/bash
# One-command fine-tune pilot on a CUDA cluster.
#
#   bash run_finetune.sh [DATA_DIR]
#
# DATA_DIR must contain distrib/shroom-vision.train.en.labeled.jsonl and either
# images/ or shroom-visions-images.tar.gz. Default: ../Shroom-Vision
#
# What it does: venv + deps -> extract images if needed -> quick pilot run
# (capped train, 2 epochs, ~10-15 min) -> full run WITH image and NO-image control
# (~30-40 min total on one A100). All outputs land in results/ft_pilot*/.
set -euo pipefail
cd "$(dirname "$0")"

DATA_DIR="${1:-../Shroom-Vision}"
TRAIN_FILE="$DATA_DIR/distrib/shroom-vision.train.en.labeled.jsonl"
IMG_DIR="$DATA_DIR/images"
[ -f "$TRAIN_FILE" ] || { echo "ERROR: $TRAIN_FILE not found"; exit 1; }

# 1) environment
if [ ! -d .venv-cluster ]; then
  python3 -m venv .venv-cluster
fi
source .venv-cluster/bin/activate
pip install -q --upgrade pip
pip install -q -r requirements/cluster.txt
python - <<'PY'
import torch
assert torch.cuda.is_available(), "CUDA not available — check the torch wheel for your cluster"
print(f"CUDA OK: {torch.cuda.get_device_name(0)}")
PY

# 2) images
if [ ! -d "$IMG_DIR" ] || [ -z "$(ls -A "$IMG_DIR" 2>/dev/null)" ]; then
  echo "[setup] extracting images..."
  bash scripts/extract_images.sh "$DATA_DIR/shroom-visions-images.tar.gz" "$IMG_DIR"
fi

# 3) quick signal run (capped train, 2 epochs) — should finish in ~10-15 min
echo "=== QUICK PILOT (capped 1500 train items, 2 epochs) ==="
python scripts/train_token_classifier.py --train_file "$TRAIN_FILE" --image_dir "$IMG_DIR" \
  --out_dir results/ft_quick --epochs 2 --max_train 1500

# 4) full pilot: with image, then the no-image control (vision ablation)
echo "=== FULL PILOT (all train, 3 epochs, with image) ==="
python scripts/train_token_classifier.py --train_file "$TRAIN_FILE" --image_dir "$IMG_DIR" \
  --out_dir results/ft_pilot --epochs 3

echo "=== CONTROL (no image) ==="
python scripts/train_token_classifier.py --train_file "$TRAIN_FILE" --image_dir "$IMG_DIR" \
  --out_dir results/ft_pilot --epochs 3 --no_image

echo ""
echo "=== DONE — summaries ==="
cat results/ft_pilot/summary_with_image.json
cat results/ft_pilot/summary_no_image.json
echo ""
echo "READ: signal exists if span_iou > floor (printed in metrics) and the with_image"
echo "run beats no_image on roc_auc/span_iou (the vision contribution)."
