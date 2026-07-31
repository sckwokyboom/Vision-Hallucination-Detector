#!/bin/bash
# Unified Visual Adaptation — screening stage (frozen-backbone half of the matrix).
#
#   bash scripts/connector/run_uva_h100.sh [--gpu N] [--data-dir DIR] [--model M]
#
# Runs on ONE H100, seed 13, the same v3 recipe as run_train_h100.sh:
#   A1       --arch connector                (answer states + visual-token memory)
#   A1-shufV --arch connector --shuffle_v    (same capacity, deranged V = grounding control)
# A0 (--arch linear) is NOT retrained: H is identical whether or not V was cached, so the
# existing results/v3_h100 summaries are the paired baseline.
#
# The LoRA half (A2/A3) needs a live-backbone trainer and is a separate script.
#
# Requires the H+V cache (~46 GB: ~12 MB/item x 3799). Extraction reuses stage-1 logic
# but WITHOUT --h_only; it will not touch the h_only cache used by A0.
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
    -h|--help) sed -n '2,15p' "$0"; exit 0 ;;
    -*) echo "unknown option: $1 (try --help)" >&2; exit 2 ;;
    *) POS+=("$1"); shift ;;
  esac
done
if [ "${#POS[@]}" -ge 1 ]; then
  if [ "$DD_SET" -eq 0 ] && [ -d "${POS[0]}/distrib" ]; then DATA_DIR="${POS[0]}"
  elif [ "$M_SET" -eq 0 ]; then MODEL="${POS[0]}"
  else echo "warn: ignoring positional '${POS[0]}' (flags already set)" >&2; fi
fi

if [ "$GPU" = "all" ]; then
  DEVICE=auto
else
  export CUDA_VISIBLE_DEVICES="$GPU"; DEVICE=cuda:0
  echo "[gpu ] CUDA_VISIBLE_DEVICES=$GPU -> cuda:0"
fi

T="$DATA_DIR/distrib/shroom-vision.train.en.labeled.jsonl"
IMG="$DATA_DIR/images"
P=splits/en.eval_protocol.json
CACHE=results/cache_h100_hv          # H + V, unlike cache_h100_bf16 (h_only)
OUT=results/uva_h100

if [ ! -f "$T" ]; then
  echo "ERROR: $T missing — run this after the dataset is in place" >&2; exit 1
fi

source scripts/connector/_env.sh
setup_cluster_env
mkdir -p "$OUT"

echo "=== stage 1: bf16 H+V features (resumable; ~46 GB, ~30 min) ==="
EXTRACT="python scripts/connector/extract_features.py --model_id $MODEL --quant bf16 \
  --train_file $T --image_dir $IMG --out_dir $CACHE --device $DEVICE"
$EXTRACT --probe 2          # expect V=(~270, 4, 3840), NOT (0, ...)
$EXTRACT

echo "=== stage 2: screening, seed 13 (A0 baseline = results/v3_h100, cached) ==="
TRAIN="python scripts/connector/train_connector.py --train_file $T --eval_ids $P \
  --cache_dir $CACHE --out_dir $OUT --epochs 12 --batch 16 --seed 13"
[ "$DEVICE" = auto ] || TRAIN="$TRAIN --device $DEVICE"
V3="--decoder bio --bio --gate_consistency --contrastive --tversky --no_type_loss"

$TRAIN --arch connector $V3 --eval_shuffle       # A1 (+ free inference-shuffle control)
$TRAIN --arch connector $V3 --shuffle_v          # A1-shufV (train-time grounding control)

echo "=== stage 3: summary ==="
python - <<'PY'
import json, glob

def rows(pat):
    for f in sorted(glob.glob(pat)):
        d = json.load(open(f))
        yield d["variant"], d["metrics"], d.get("metrics_shuffled_image")

print(f"{'variant':52}{'iou':>7}{'dirty':>7}{'clnOK':>6}{'corR':>7}")
print("--- A0 reference (results/v3_h100, arch=linear, h_only) ---")
for v, m, _ in rows("results/v3_h100/summary_linear_bio*_s13.json"):
    print(f"{v:52}{m['span_iou']:>7.4f}{m['dirty_iou']:>7.3f}{m['clean_empty']:>6.2f}{m['pooled_pearson_debug']:>7.3f}")
print("--- A1 / controls (this run) ---")
for v, m, ms in rows("results/uva_h100/summary_*.json"):
    print(f"{v:52}{m['span_iou']:>7.4f}{m['dirty_iou']:>7.3f}{m['clean_empty']:>6.2f}{m['pooled_pearson_debug']:>7.3f}")
    if ms:
        print(f"{'  ^ eval-time shuffled-image control':52}{ms['span_iou']:>7.4f}")
print("""
read-out:
  A1 > A0 and A1 > A1-shufV  -> --h_only was discarding signal; visual memory is real
  A1 ~ A1-shufV > A0         -> gain is capacity, not vision (connector arch != visual win)
  A1 ~ A0                    -> frozen V states carry nothing the answer states lack;
                                the LoRA half (A2/A3) is the remaining hypothesis
""")
PY
echo "done -> $OUT"
