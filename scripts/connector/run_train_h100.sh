#!/bin/bash
# FULL-SCALE v3 training on H100 with full-precision (bf16) Gemma 4 12B features.
#
#   bash scripts/connector/run_train_h100.sh [DATA_DIR] [MODEL]
#
#   DATA_DIR — dataset folder (default ../Shroom-Vision)
#   MODEL    — HF id or LOCAL PATH to Gemma 4 12B weights (default google/gemma-4-12B-it)
#
# What runs (all evaluated on the frozen tune split; held-out untouched):
#   stage 1: bf16 feature extraction for ALL en items (~3799; ~20-40 min on H100)
#   stage 2: v3 = BIO boundary head + gate-consistency + contrastive groups + Tversky
#            - main v3, 3 seeds
#            - component ablations (minus-bio / minus-contrastive / minus-consistency)
#            - v1 baseline (gate_hyst) for reference
#   stage 3: summary table
set -euo pipefail
cd "$(dirname "$0")/../.."
DATA_DIR="${1:-../Shroom-Vision}"
MODEL="${2:-google/gemma-4-12B-it}"
T="$DATA_DIR/distrib/shroom-vision.train.en.labeled.jsonl"
IMG="$DATA_DIR/images"
P=splits/en.eval_protocol.json
CACHE=results/cache_h100_bf16
OUT=results/v3_h100
[ -f "$T" ] || { echo "ERROR: $T not found (see data/README.md)"; exit 1; }

source .venv-cluster/bin/activate 2>/dev/null || {
  python3 -m venv .venv-cluster && source .venv-cluster/bin/activate
  pip install -q --upgrade pip
  REQ=requirements/cluster.txt; [ -f "$REQ" ] || REQ=requirements-cluster.txt
  pip install -q -r "$REQ" bitsandbytes accelerate; }
python -c "import torch; assert torch.cuda.is_available(), 'no CUDA torch'"
if [ ! -d "$IMG" ] || [ -z "$(ls -A "$IMG" 2>/dev/null)" ]; then
  bash scripts/extract_images.sh "$DATA_DIR/shroom-visions-images.tar.gz" "$IMG"
fi
mkdir -p splits "$OUT"
[ -f splits/dev.en.jsonl ] || python -m shroom.make_split --distrib "$DATA_DIR/distrib" --out_dir splits

echo "=== stage 1: bf16 features for ALL items (resumable) ==="
python scripts/connector/extract_features.py --model_id "$MODEL" --quant bf16 \
  --train_file "$T" --image_dir "$IMG" --out_dir "$CACHE" --h_only --probe 2
python scripts/connector/extract_features.py --model_id "$MODEL" --quant bf16 \
  --train_file "$T" --image_dir "$IMG" --out_dir "$CACHE" --h_only

TRAIN="python scripts/connector/train_connector.py --train_file $T --eval_ids $P \
  --cache_dir $CACHE --out_dir $OUT --arch linear --epochs 12 --batch 16"
V3="--decoder bio --bio --gate_consistency --contrastive --tversky --no_type_loss"

echo "=== stage 2a: v3 main, 3 seeds ==="
for s in 13 42 77; do $TRAIN --seed $s $V3; done

echo "=== stage 2b: component ablations (seed 13) ==="
$TRAIN --seed 13 --decoder gate_hyst --tversky --no_type_loss --gate_consistency --contrastive   # minus-bio
$TRAIN --seed 13 --decoder bio --bio --gate_consistency --tversky --no_type_loss                 # minus-contrastive
$TRAIN --seed 13 --decoder bio --bio --contrastive --tversky --no_type_loss                      # minus-consistency
$TRAIN --seed 13 --decoder gate_hyst                                                             # v1 reference

echo "=== stage 3: summary ==="
python - <<'PY'
import json, glob
rows = []
for f in sorted(glob.glob("results/v3_h100/summary_*.json")):
    d = json.load(open(f)); m = d["metrics"]
    rows.append((d["variant"], m))
print(f"{'variant':46}{'iou':>7}{'dirty':>7}{'clnOK':>6}{'gateR':>6}{'corR':>7}{'corS':>7}{'corLbl':>7}")
for v, m in rows:
    print(f"{v:46}{m['span_iou']:>7.4f}{m['dirty_iou']:>7.3f}{m['clean_empty']:>6.2f}"
          f"{m['dirty_gate_recall']:>6.2f}{m['cor_raw']:>7.3f}{m['cor_submission']:>7.3f}{m['cor_lbl']:>7.3f}")
print("\nreference points: predict-nothing floor 0.213 | Mac 4-bit v1 full-train 0.306 | "
      "zero-shot HYBRID 0.273 / Cor 0.164")
PY
echo "done -> results/v3_h100/ (summaries, manifests, dev predictions, weights)"
