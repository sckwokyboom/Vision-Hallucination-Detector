#!/bin/bash
# Precision ladder on one H100: BF16 (reference) vs INT8 vs NF4 — same subset, same
# decoder, paired report.
#
#   bash scripts/connector/run_quant_h100.sh [--gpu N] [--data-dir DIR] [--model M]
#
#   --gpu N  — pin to one card via CUDA_VISIBLE_DEVICES (default 0; 'all' = don't mask)
#   DATA_DIR — folder with distrib/*.jsonl and the images (default ../Shroom-Vision);
#              downloaded by scripts/get_data.sh if missing
#   MODEL    — HF id OR a LOCAL PATH to already-downloaded Gemma 4 12B weights,
#              e.g. /models/gemma-4-12B-it. Default google/gemma-4-12B-it (the HF id is
#              license-gated -> needs `huggingface-cli login`; a local path does not).
set -euo pipefail
cd "$(dirname "$0")/../.."
GPU="${GPU:-0}"
DD_SET=0; M_SET=0
DATA_DIR="${DATA_DIR:-../Shroom-Vision}"
MODEL="${MODEL:-google/gemma-4-12B-it}"
POS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --gpu) GPU="$2"; shift 2 ;;
    --data-dir) DATA_DIR="$2"; DD_SET=1; shift 2 ;;
    --model) MODEL="$2"; M_SET=1; shift 2 ;;
    -h|--help) sed -n '2,12p' "$0"; exit 0 ;;
    -*) echo "unknown option: $1 (try --help)" >&2; exit 2 ;;
    *) POS+=("$1"); shift ;;
  esac
done
# legacy positional form — never overrides an explicit flag. A lone positional
# that looks like a model path (no distrib/ inside) is treated as --model.
if [ "${#POS[@]}" -ge 1 ]; then
  if [ "$DD_SET" -eq 0 ] && [ -d "${POS[0]}/distrib" ]; then DATA_DIR="${POS[0]}"
  elif [ "$M_SET" -eq 0 ]; then MODEL="${POS[0]}"
  else echo "warn: ignoring positional '${POS[0]}' (flags already set)" >&2; fi
fi
[ "${#POS[@]}" -ge 2 ] && [ "$M_SET" -eq 0 ] && MODEL="${POS[1]}"
if [ "$GPU" = "all" ]; then
  DEVICE=auto; TRAIN_DEV=(--device cuda)   # the decoder is tiny — it never needs sharding
else
  export CUDA_VISIBLE_DEVICES="$GPU"; DEVICE=cuda:0; TRAIN_DEV=(--device cuda:0)
  echo "[gpu ] CUDA_VISIBLE_DEVICES=$GPU -> cuda:0"
fi
T="$DATA_DIR/distrib/shroom-vision.train.en.labeled.jsonl"
IMG="$DATA_DIR/images"
SUB=splits/subset_quant.jsonl
P=splits/en.eval_protocol.json
if [ ! -f "$T" ] && [ "$DD_SET" -eq 1 ]; then
  echo "ERROR: --data-dir '$DATA_DIR' has no distrib/shroom-vision.train.en.labeled.jsonl" >&2
  echo "       (not downloading: --data-dir was given explicitly)" >&2; exit 1
fi
if { [ ! -f "$T" ] || [ ! -d "$IMG" ] || [ -z "$(ls -A "$IMG" 2>/dev/null)" ]; } && [ "$DD_SET" -eq 0 ]; then
  bash scripts/get_data.sh --data-dir "$DATA_DIR"
fi
[ -f "$T" ] || { echo "ERROR: $T still missing after scripts/get_data.sh"; exit 1; }

# --- environment (handles both repo layouts) ---
source .venv-cluster/bin/activate 2>/dev/null || {
  python3 -m venv .venv-cluster && source .venv-cluster/bin/activate
  pip install -q --upgrade pip
  REQ=requirements/cluster.txt; [ -f "$REQ" ] || REQ=requirements-cluster.txt
  pip install -q -r "$REQ" bitsandbytes accelerate; }
python -c "import torch; assert torch.cuda.is_available(), 'no CUDA torch'"
echo "[env] OK, model=$MODEL"

# --- data: dev gold for evaluation ---
mkdir -p splits
[ -f splits/dev.en.jsonl ] || python -m shroom.make_split --distrib "$DATA_DIR/distrib" --out_dir splits
python scripts/connector/make_subset.py --train_file "$T" --protocol "$P" --out "$SUB"

# --- precision ladder ---
for Q in bf16 int8 nf4; do
  echo "=== [$Q] probe ==="
  python scripts/connector/extract_features.py --model_id "$MODEL" --quant $Q --device "$DEVICE" \
    --train_file "$SUB" --image_dir "$IMG" --out_dir results/cache_h100_$Q --h_only --probe 2
  echo "=== [$Q] extract ==="
  python scripts/connector/extract_features.py --model_id "$MODEL" --quant $Q --device "$DEVICE" \
    --train_file "$SUB" --image_dir "$IMG" --out_dir results/cache_h100_$Q --h_only
  echo "=== [$Q] train decoder ==="
  python scripts/connector/train_connector.py --train_file "$T" --eval_ids "$P" \
    "${TRAIN_DEV[@]}" \
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
