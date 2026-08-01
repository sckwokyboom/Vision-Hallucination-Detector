#!/bin/bash
# Cascade experiment: LoRA backbone + раздельные gate/locator objectives.
#
#   bash scripts/connector/run_cascade_h100.sh --gpu 0
#
# Окружение: conda с pytorch, transformers, torchvision, scikit-learn.
# Данные уже на кластере:
#   train:  /userspace/srm/text_dataset/shroom-vision.train.en.labeled.jsonl
#   images: /userspace/srm/shroom-vis-images
#
# Тренирует 3 варианта на одной A100 (~5.5 h total):
#   C1: cascade, LoRA from layer 24 (все слои, которые видит декодер)
#   C2: cascade, LoRA from layer 40 (top-8 only — контроль изоляции эффекта)
#   C3: cascade-frozen control (freeze_lora_epochs=99 — decoder-only, без LoRA)
#
# Live forward per step, batch 1 × accum 16, 5 эпох.
# warm-start декодера из A0 frozen чекпоинта (если найден).
set -euo pipefail
cd "$(dirname "$0")/../.."

GPU="${GPU:-0}"
TRAIN_FILE="${TRAIN_FILE:-/userspace/srm/text_dataset/shroom-vision.train.en.labeled.jsonl}"
IMAGE_DIR="${IMAGE_DIR:-/userspace/srm/shroom-vis-images}"
MODEL="${MODEL:-google/gemma-4-12B-it}"
LORA_FROM="${LORA_FROM:-24}"
EVAL_IDS=splits/en.eval_protocol.json
OUT=results/cascade_h100

while [ $# -gt 0 ]; do
  case "$1" in
    --gpu) GPU="$2"; shift 2 ;;
    --train) TRAIN_FILE="$2"; shift 2 ;;
    --images) IMAGE_DIR="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --lora-from) LORA_FROM="$2"; shift 2 ;;
    -h|--help) sed -n '2,16p' "$0"; exit 0 ;;
    *) echo "unknown option: $1 (try --help)" >&2; exit 2 ;;
  esac
done

export CUDA_VISIBLE_DEVICES="$GPU"
echo "[run ] CUDA_VISIBLE_DEVICES=$GPU -> cuda:0"
echo "[run ] train: $TRAIN_FILE"
echo "[run ] images: $IMAGE_DIR"
echo "[run ] model: $MODEL"

[ -f "$TRAIN_FILE" ] || { echo "ERROR: $TRAIN_FILE missing" >&2; exit 1; }
[ -d "$IMAGE_DIR" ] || { echo "ERROR: $IMAGE_DIR missing" >&2; exit 1; }
[ -f "$EVAL_IDS" ] || { echo "ERROR: $EVAL_IDS missing" >&2; exit 1; }

python -c "import torch; assert torch.cuda.is_available(), 'no CUDA torch'" || exit 1
python -c "import torchvision, sklearn" || {
  echo "ERROR: missing torchvision or scikit-learn — install in your conda env" >&2
  exit 1
}

mkdir -p "$OUT"

A0_CK=results/v3_h100/best_iou_linear_bio_tv_bio_gc_ctr_notype_s13.pt
DEC_INIT=""
if [ -f "$A0_CK" ]; then
  DEC_INIT="--init_from $A0_CK"
  echo "[run ] warm-start decoder from $A0_CK"
else
  echo "[run ] no A0 checkpoint found — decoder from scratch"
fi

BASE="python scripts/connector/train_lora.py --model_id $MODEL \
  --train_file $TRAIN_FILE --image_dir $IMAGE_DIR --eval_ids $EVAL_IDS \
  --device cuda:0 --seed 13 --arch linear --cascade_train \
  --epochs 5 --freeze_lora_epochs 1 --accum 16 \
  --lambda_gate 1.0 --focal_gamma 2.0 --l_bio 0.5 $DEC_INIT"

echo ""
echo "=== C1: LoRA cascade (from layer $LORA_FROM) ==="
$BASE --lora_from "$LORA_FROM" --out_dir "$OUT/c1"

echo ""
echo "=== C2: LoRA cascade, top-8 only (from layer 40) ==="
$BASE --lora_from 40 --out_dir "$OUT/c2"

echo ""
echo "=== C3: cascade-frozen control (no LoRA, decoder only) ==="
$BASE --lora_from "$LORA_FROM" --freeze_lora_epochs 99 --out_dir "$OUT/c3_frozen"

echo ""
echo "=== summary ==="
python - <<'PY'
import json, glob
print(f"{'variant':60}{'iou':>7}{'dirty':>7}{'clnOK':>6}{'corR':>7}{'gAUC':>7}")
for pat, label in [
    ("results/cascade_h100/c1/summary_*.json",      "C1 · LoRA cascade"),
    ("results/cascade_h100/c2/summary_*.json",      "C2 · LoRA cascade top-8"),
    ("results/cascade_h100/c3_frozen/summary_*.json","C3 · cascade frozen ctrl"),
    ("results/v3_h100/summary_linear_bio*_s13.json", "A0 · frozen v3 (baseline)"),
    ("results/lora_h100/a2/summary_*.json",          "A2 · LoRA v3 (for reference)"),
]:
    files = sorted(glob.glob(pat))
    if files:
        print(f"--- {label} ---")
    for f in files:
        d = json.load(open(f)); m = d["metrics"]
        ga = m.get("gate_auc", float("nan"))
        print(f"{d['variant']:60}{m['span_iou']:>7.4f}{m['dirty_iou']:>7.3f}"
              f"{m['clean_empty']:>6.2f}{m['pooled_pearson_debug']:>7.3f}{ga:>7.3f}")
print("""
expected gains:
  cascade-frozen vs A0 frozen v3  -> dirty-only training effect on same backbone
  cascade-LoRA vs cascade-frozen  -> LoRA adaptation effect on cascade objective
  cascade-LoRA vs A2 joint LoRA   -> cascade vs joint training at same LoRA budget
""")
PY
echo "done -> $OUT"
