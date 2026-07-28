#!/bin/bash
# Precision ladder on one H100: BF16 (reference) vs INT8 vs NF4 — same subset, same
# decoder, paired report.
#
#   bash scripts/connector/run_quant_h100.sh [DATA_DIR] [MODEL]
#
#   DATA_DIR — folder with distrib/*.jsonl and shroom-visions-images.tar.gz
#              (default ../Shroom-Vision)
#   MODEL    — HF id OR a LOCAL PATH to already-downloaded Gemma 4 12B weights,
#              e.g. /models/gemma-4-12B-it. Default google/gemma-4-12B-it (the HF id is
#              license-gated -> needs `huggingface-cli login`; a local path does not).
set -euo pipefail
cd "$(dirname "$0")/../.."
DATA_DIR="${1:-../Shroom-Vision}"
MODEL="${2:-google/gemma-4-12B-it}"
T="$DATA_DIR/distrib/shroom-vision.train.en.labeled.jsonl"
IMG="$DATA_DIR/images"
SUB=splits/subset_quant.jsonl
P=splits/en.eval_protocol.json
[ -f "$T" ] || { echo "ERROR: $T not found — put the dataset next to the repo (see data/README.md)"; exit 1; }

# --- environment (handles both repo layouts) ---
source .venv-cluster/bin/activate 2>/dev/null || {
  python3 -m venv .venv-cluster && source .venv-cluster/bin/activate
  pip install -q --upgrade pip
  REQ=requirements/cluster.txt; [ -f "$REQ" ] || REQ=requirements-cluster.txt
  pip install -q -r "$REQ" bitsandbytes accelerate; }
python -c "import torch; assert torch.cuda.is_available(), 'no CUDA torch'"
echo "[env] OK, model=$MODEL"

# --- data: images + dev gold for evaluation ---
if [ ! -d "$IMG" ] || [ -z "$(ls -A "$IMG" 2>/dev/null)" ]; then
  bash scripts/extract_images.sh "$DATA_DIR/shroom-visions-images.tar.gz" "$IMG"
fi
mkdir -p splits
[ -f splits/dev.en.jsonl ] || python -m shroom.make_split --distrib "$DATA_DIR/distrib" --out_dir splits
python scripts/connector/make_subset.py --train_file "$T" --protocol "$P" --out "$SUB"

# --- precision ladder ---
for Q in bf16 int8 nf4; do
  echo "=== [$Q] probe ==="
  python scripts/connector/extract_features.py --model_id "$MODEL" --quant $Q \
    --train_file "$SUB" --image_dir "$IMG" --out_dir results/cache_h100_$Q --h_only --probe 2
  echo "=== [$Q] extract ==="
  python scripts/connector/extract_features.py --model_id "$MODEL" --quant $Q \
    --train_file "$SUB" --image_dir "$IMG" --out_dir results/cache_h100_$Q --h_only
  echo "=== [$Q] train decoder ==="
  python scripts/connector/train_connector.py --train_file "$T" --eval_ids "$P" \
    --cache_dir results/cache_h100_$Q --out_dir results/quant_h100/$Q \
    --arch linear --seed 13 --epochs 12 --batch 16 --max_train 1000
done

echo "=== REPORT (paired, image-cluster CIs) ==="
mkdir -p results/quant_h100
python scripts/connector/quant_report.py --gold splits/dev.en.jsonl \
  --pred bf16=results/quant_h100/bf16/dev_pred_linear_gate_hyst_s13.jsonl \
  --pred int8=results/quant_h100/int8/dev_pred_linear_gate_hyst_s13.jsonl \
  --pred nf4=results/quant_h100/nf4/dev_pred_linear_gate_hyst_s13.jsonl \
  | tee results/quant_h100/REPORT.txt
echo "done -> results/quant_h100/REPORT.txt (+ summary_*.json per variant)"
