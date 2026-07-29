#!/bin/bash
# FULL-SCALE v3 training on ONE H100 with full-precision (bf16) Gemma 4 12B features.
#
#   bash scripts/connector/run_train_h100.sh [--gpu N] [--data-dir DIR] [--model M] [--dry-run]
#
#   --gpu N        which card to use (default 0). Sets CUDA_VISIBLE_DEVICES=N, so every
#                  child process sees exactly one GPU and 'cuda:0' means physical GPU N.
#                  --gpu all leaves CUDA_VISIBLE_DEVICES untouched and shards across cards.
#   --data-dir DIR dataset folder (default ../Shroom-Vision); downloaded by
#                  scripts/get_data.sh if missing.
#   --model M      HF id or LOCAL PATH to Gemma 4 12B weights (default google/gemma-4-12B-it;
#                  the HF id is license-gated -> needs `huggingface-cli login`, a path does not)
#   --dry-run      print the resolved plan and exit without loading anything
#
# The legacy positional form `run_train_h100.sh DATA_DIR MODEL` still works.
#
# What runs (all evaluated on the frozen tune split; held-out untouched):
#   stage 0: dataset present? else download (login node only — compute nodes rarely have network)
#   stage 1: bf16 feature extraction for ALL en items (~3799; ~20-40 min on one H100)
#   stage 2: v3 = BIO boundary head + gate-consistency + contrastive groups + Tversky
#            - main v3, 3 seeds
#            - component ablations (minus-bio / minus-contrastive / minus-consistency)
#            - v1 baseline (gate_hyst) for reference
#   stage 3: summary table
#
# Resumable: stage 1 skips already-cached items, stage 2 is minutes per variant.
set -euo pipefail
cd "$(dirname "$0")/../.."

GPU="${GPU:-0}"
DD_SET=0; M_SET=0
DATA_DIR="${DATA_DIR:-../Shroom-Vision}"
MODEL="${MODEL:-google/gemma-4-12B-it}"
DRY=0
POS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --gpu) GPU="$2"; shift 2 ;;
    --data-dir) DATA_DIR="$2"; DD_SET=1; shift 2 ;;
    --model) MODEL="$2"; M_SET=1; shift 2 ;;
    --dry-run) DRY=1; shift ;;
    -h|--help) sed -n '2,26p' "$0"; exit 0 ;;
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

# --- GPU selection -----------------------------------------------------------------
# Masking with CUDA_VISIBLE_DEVICES (rather than only passing --device) also pins
# accelerate, bitsandbytes and any NCCL init to the same card.
if [ "$GPU" = "all" ]; then
  DEVICE=auto
  echo "[gpu ] using every visible GPU (CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>})"
else
  export CUDA_VISIBLE_DEVICES="$GPU"
  DEVICE=cuda:0
  echo "[gpu ] CUDA_VISIBLE_DEVICES=$GPU -> cuda:0"
fi

T="$DATA_DIR/distrib/shroom-vision.train.en.labeled.jsonl"
IMG="$DATA_DIR/images"
P=splits/en.eval_protocol.json
CACHE=results/cache_h100_bf16
OUT=results/v3_h100

if [ "$DRY" -eq 1 ]; then
  printf 'data:   %s\nmodel:  %s\ndevice: %s\ncache:  %s\nout:    %s\n' \
         "$DATA_DIR" "$MODEL" "$DEVICE" "$CACHE" "$OUT"
  exit 0
fi

echo "=== stage 0: dataset ==="
if [ ! -f "$T" ] && [ "$DD_SET" -eq 1 ]; then
  echo "ERROR: --data-dir '$DATA_DIR' has no distrib/shroom-vision.train.en.labeled.jsonl" >&2
  echo "       (not downloading: --data-dir was given explicitly)" >&2; exit 1
fi
if { [ ! -f "$T" ] || [ ! -d "$IMG" ] || [ -z "$(ls -A "$IMG" 2>/dev/null)" ]; } && [ "$DD_SET" -eq 0 ]; then
  bash scripts/get_data.sh --data-dir "$DATA_DIR"
fi
[ -f "$T" ] || { echo "ERROR: $T still missing after scripts/get_data.sh"; exit 1; }

source scripts/connector/_env.sh
setup_cluster_env
python - <<'PY'
import torch
n = torch.cuda.device_count()
print(f"[env ] torch {torch.__version__}, {n} visible GPU(s): " + ", ".join(
    f"{i}:{torch.cuda.get_device_name(i)} "
    f"({torch.cuda.get_device_properties(i).total_memory / 2**30:.0f} GiB)" for i in range(n)))
PY
mkdir -p splits "$OUT"
[ -f splits/dev.en.jsonl ] || python -m shroom.make_split --distrib "$DATA_DIR/distrib" --out_dir splits

echo "=== stage 1: bf16 features for ALL items (resumable) ==="
EXTRACT="python scripts/connector/extract_features.py --model_id $MODEL --quant bf16 \
  --train_file $T --image_dir $IMG --out_dir $CACHE --device $DEVICE --h_only"
$EXTRACT --probe 2
$EXTRACT

TRAIN="python scripts/connector/train_connector.py --train_file $T --eval_ids $P \
  --cache_dir $CACHE --out_dir $OUT --arch linear --epochs 12 --batch 16"
[ "$DEVICE" = auto ] || TRAIN="$TRAIN --device $DEVICE"
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
print("\nreference points on tune-202: floor 0.213 | starter baseline 0.200 | zero-shot HYBRID "
      "0.273 | Mac 4-bit v1 0.304 | Mac 4-bit BIO 0.320  (see results/COMPARISON.md)")
PY
echo "done -> results/v3_h100/ (summaries, manifests, dev predictions, weights)"
cat <<EOF
next: score it against everything else with the same scorer
  python scripts/unified_table.py --gold splits/tune202.en.jsonl \\
    --pred "h100 v3 s13=$OUT/dev_pred_linear_bio_tv_bio_gc_ctr_notype_s13.jsonl" \\
    --out results/COMPARISON_h100.md
EOF
