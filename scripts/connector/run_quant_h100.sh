#!/bin/bash
# Precision ladder on one H100: BF16 (reference) vs INT8 vs NF4 — same subset, same
# decoder, paired report. Usage:  bash scripts/connector/run_quant_h100.sh [DATA_DIR]
set -euo pipefail
cd "$(dirname "$0")/../.."
DATA_DIR="${1:-../Shroom-Vision}"
T="$DATA_DIR/distrib/shroom-vision.train.en.labeled.jsonl"
IMG="$DATA_DIR/images"
SUB=splits/subset_quant.jsonl
P=splits/en.eval_protocol.json

source .venv-cluster/bin/activate 2>/dev/null || {
  python3 -m venv .venv-cluster && source .venv-cluster/bin/activate
  pip install -q --upgrade pip
  pip install -q -r requirements-cluster.txt bitsandbytes accelerate; }
python -c "import torch; assert torch.cuda.is_available()"
[ -d "$IMG" ] && [ -n "$(ls -A "$IMG")" ] || bash scripts/extract_images.sh "$DATA_DIR/shroom-visions-images.tar.gz" "$IMG"
mkdir -p splits && python scripts/connector/make_subset.py --train_file "$T" --protocol "$P" --out "$SUB"

for Q in bf16 int8 nf4; do
  echo "=== extract $Q ==="
  python scripts/connector/extract_features.py --quant $Q --train_file "$SUB" \
    --image_dir "$IMG" --out_dir results/cache_h100_$Q --h_only --probe 2
  python scripts/connector/extract_features.py --quant $Q --train_file "$SUB" \
    --image_dir "$IMG" --out_dir results/cache_h100_$Q --h_only
  echo "=== train decoder on $Q ==="
  python scripts/connector/train_connector.py --train_file "$T" --eval_ids "$P" \
    --cache_dir results/cache_h100_$Q --out_dir results/quant_h100/$Q \
    --arch linear --seed 13 --epochs 12 --batch 16 --max_train 1000
done

echo "=== REPORT ==="
python scripts/connector/quant_report.py --gold splits/dev.en.jsonl \
  --pred bf16=results/quant_h100/bf16/dev_pred_linear_gate_hyst_s13.jsonl \
  --pred int8=results/quant_h100/int8/dev_pred_linear_gate_hyst_s13.jsonl \
  --pred nf4=results/quant_h100/nf4/dev_pred_linear_gate_hyst_s13.jsonl
echo "NOTE: dev gold file: generate with 'python -m shroom.make_split' if missing."
