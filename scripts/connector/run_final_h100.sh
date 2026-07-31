#!/bin/bash
# Final SHROOM Vision EN protocol runner for one H100.
#
# Typical remote usage:
#   bash scripts/connector/run_final_h100.sh --gpu 1 --phase a2-backup
#   bash scripts/connector/run_final_h100.sh --gpu 1 --phase cascade-screen
#   bash scripts/connector/run_final_h100.sh --gpu 1 --phase a2-backup --test-file /path/test.en.jsonl
#
# Other phases:
#   a2-train-s13        train cold-start A2 seed 13 if the old checkpoint is absent
#   a0-rerun            rerun frozen A0 joint model from cached h-only features
#   a2-frozen           frozen-backbone control through the live LoRA trainer
#   a2-finalist-seeds   train A2 seeds 42 and 77
#   old-queue           a0-rerun, a2-frozen, then a2-finalist-seeds
#   all                 a2-backup, then cascade-screen
set -euo pipefail
cd "$(dirname "$0")/../.."

GPU="${GPU:-1}"
PHASE="${PHASE:-all}"
DATA_DIR="${DATA_DIR:-../Shroom-Vision}"
MODEL="${MODEL:-/workspace/data/models/gemma-4-12B-it}"
LORA_FROM="${LORA_FROM:-24}"
A2_CK="${A2_CK:-results/lora_h100/a2/best_iou_lora_linear_f24_r16_s13.pt}"
TEST_FILE="${TEST_FILE:-}"

while [ $# -gt 0 ]; do
  case "$1" in
    --gpu) GPU="$2"; shift 2 ;;
    --phase) PHASE="$2"; shift 2 ;;
    --data-dir) DATA_DIR="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --lora-from) LORA_FROM="$2"; shift 2 ;;
    --a2-ck) A2_CK="$2"; shift 2 ;;
    --test-file) TEST_FILE="$2"; shift 2 ;;
    -h|--help) sed -n '2,23p' "$0"; exit 0 ;;
    -*) echo "unknown option: $1" >&2; exit 2 ;;
    *) echo "unexpected positional argument: $1" >&2; exit 2 ;;
  esac
done

export CUDA_VISIBLE_DEVICES="$GPU"
DEVICE=cuda:0
TRAIN_EN="$DATA_DIR/distrib/shroom-vision.train.en.labeled.jsonl"
TEST_EN="${TEST_FILE:-$DATA_DIR/distrib/shroom-vision.test.en.jsonl}"
IMG_DIR="$DATA_DIR/images"
PROTO=splits/en.eval_protocol.json
FINAL=results/final
LOG_DIR="$FINAL/logs"

mkdir -p "$FINAL/current_a2" "$FINAL/cascade_lora" "$FINAL/heldout" \
  "$FINAL/submission" "$LOG_DIR"

require_file() {
  [ -f "$1" ] || { echo "ERROR: missing required file: $1" >&2; exit 1; }
}

resolve_test_file() {
  if [ -f "$TEST_EN" ]; then
    return 0
  fi
  local -a candidates=()
  if [ -d "$DATA_DIR/distrib" ]; then
    while IFS= read -r path; do
      candidates+=("$path")
    done < <(find "$DATA_DIR/distrib" -maxdepth 1 -type f \
      \( -name '*test*en*.jsonl' -o -name '*test*.en*.jsonl' \) | sort)
  fi
  if [ "${#candidates[@]}" -eq 1 ]; then
    TEST_EN="${candidates[0]}"
    echo "[test] using discovered EN test file: $TEST_EN"
    return 0
  fi
  {
    echo "ERROR: EN test file is missing."
    echo "       Looked for: $TEST_EN"
    if [ "${#candidates[@]}" -gt 1 ]; then
      echo "       Multiple possible test files were found; rerun with --test-file PATH:"
      printf '         %s\n' "${candidates[@]}"
    else
      echo "       Put the official EN test JSONL under DATA_DIR/distrib or rerun with:"
      echo "         --test-file /absolute/or/relative/path/to/shroom-vision.test.en.jsonl"
      echo "       Quick remote check:"
      echo "         find $DATA_DIR -name '*test*en*.jsonl' -o -name '*test*.en*.jsonl'"
    fi
  } >&2
  return 1
}

setup() {
  require_file "$TRAIN_EN"
  require_file "$PROTO"
  [ -d "$IMG_DIR" ] || { echo "ERROR: missing image dir: $IMG_DIR" >&2; exit 1; }
  source scripts/connector/_env.sh
  setup_cluster_env
  [ -f splits/dev.en.jsonl ] || python -m shroom.make_split --distrib "$DATA_DIR/distrib" --out_dir splits
  python - "$MODEL" "$TRAIN_EN" "$PROTO" "$GPU" <<'PY'
import json, os, platform, subprocess, sys
try:
    import torch
except Exception:
    torch = None
try:
    import transformers
except Exception:
    transformers = None
model, train_file, proto_file, gpu = sys.argv[1:]
proto = json.load(open(proto_file))
def cmd(args):
    try:
        return subprocess.check_output(args, text=True).strip()
    except Exception as exc:
        return f"unavailable: {type(exc).__name__}: {exc}"
manifest = {
    "git_sha": cmd(["git", "rev-parse", "HEAD"]),
    "git_status": cmd(["git", "status", "--short"]),
    "model_id": model,
    "dtype": "bf16",
    "gpu_requested": gpu,
    "gpu_visible": (
        torch.cuda.get_device_name(0) if torch is not None and torch.cuda.is_available()
        else "cuda unavailable"
    ),
    "python": sys.version.replace("\n", " "),
    "platform": platform.platform(),
    "torch": getattr(torch, "__version__", None),
    "transformers": getattr(transformers, "__version__", None),
    "train_file": train_file,
    "eval_protocol": proto_file,
    "tune_dev_count": len(proto.get("tune_dev", [])),
    "heldout_count": len(proto.get("heldout", [])),
    "seed": 13,
}
os.makedirs("results/final", exist_ok=True)
json.dump(manifest, open("results/final/RUN_MANIFEST.json", "w"), indent=2)
print(json.dumps(manifest, indent=2))
PY
}

run_logged() {
  local log="$1"
  shift
  echo "=== $* ==="
  "$@" 2>&1 | tee "$log"
}

checkpoint_provenance() {
  local ck="$1"
  python - "$ck" <<'PY'
import json, sys, torch
path = sys.argv[1]
ck = torch.load(path, map_location="cpu", weights_only=False)
out = {
    "checkpoint": path,
    "args.init_from": (ck.get("args") or {}).get("init_from"),
    "epoch": ck.get("epoch"),
    "metrics": ck.get("metrics"),
}
print(json.dumps(out, indent=2, default=str))
PY
}

find_a2_checkpoint() {
  if [ -f "$A2_CK" ]; then
    echo "$A2_CK"
    return
  fi
  local found
  found=$(find results/lora_h100/a2 -maxdepth 1 -name 'best_iou_lora_linear*_s13.pt' 2>/dev/null | sort | tail -n 1 || true)
  [ -n "$found" ] || {
    echo "ERROR: A2 checkpoint not found. Expected $A2_CK" >&2
    echo "Run: bash $0 --gpu $GPU --phase a2-train-s13 --data-dir $DATA_DIR --model $MODEL" >&2
    exit 1
  }
  echo "$found"
}

phase_a2_train_s13() {
  setup
  mkdir -p results/lora_h100/a2
  run_logged "$LOG_DIR/a2_train_s13.log" \
    python scripts/connector/train_lora.py \
      --model_id "$MODEL" \
      --train_file "$TRAIN_EN" \
      --image_dir "$IMG_DIR" \
      --eval_ids "$PROTO" \
      --out_dir results/lora_h100/a2 \
      --arch linear \
      --seed 13 \
      --epochs 5 \
      --lora_from "$LORA_FROM" \
      --device "$DEVICE"
}

phase_a2_backup() {
  setup
  local ck
  ck=$(find_a2_checkpoint)
  checkpoint_provenance "$ck" | tee "$FINAL/current_a2/checkpoint_provenance.json"

  python scripts/connector/predict_lora.py \
    --checkpoint "$ck" \
    --model_id "$MODEL" \
    --input "$TRAIN_EN" \
    --image_dir "$IMG_DIR" \
    --output "$FINAL/current_a2/tune_predictions.jsonl" \
    --decoder bio \
    --device "$DEVICE"

  python scripts/official/eval_official.py \
    --gold splits/dev.en.jsonl \
    --eval_ids "$PROTO" \
    --pred "A2=$FINAL/current_a2/tune_predictions.jsonl" \
    2>&1 | tee "$FINAL/current_a2/official_eval.txt"

  resolve_test_file
  python scripts/connector/predict_lora.py \
    --checkpoint "$ck" \
    --model_id "$MODEL" \
    --input "$TEST_EN" \
    --image_dir "$IMG_DIR" \
    --output "$FINAL/submission/submission_a2_en.jsonl" \
    --decoder bio \
    --device "$DEVICE"

  python scripts/official/format_checker.py \
    "$FINAL/submission/submission_a2_en.jsonl" \
    --reference-file "$TEST_EN" \
    2>&1 | tee "$FINAL/submission/format_check_a2.txt"
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$FINAL/submission/submission_a2_en.jsonl" > "$FINAL/submission/sha256_a2.txt"
  else
    shasum -a 256 "$FINAL/submission/submission_a2_en.jsonl" > "$FINAL/submission/sha256_a2.txt"
  fi
}

phase_cascade_screen() {
  setup
  run_logged "$LOG_DIR/cascade_lora_s13.log" \
    python scripts/connector/train_lora.py \
      --model_id "$MODEL" \
      --train_file "$TRAIN_EN" \
      --image_dir "$IMG_DIR" \
      --eval_ids "$PROTO" \
      --out_dir "$FINAL/cascade_lora" \
      --arch linear \
      --cascade_train \
      --seed 13 \
      --epochs 5 \
      --lora_from "$LORA_FROM" \
      --device "$DEVICE"

  local pred
  pred=$(find "$FINAL/cascade_lora" -maxdepth 1 -name 'dev_pred_lora_linear_cascade*_s13.jsonl' | sort | tail -n 1)
  python scripts/official/eval_official.py \
    --gold splits/dev.en.jsonl \
    --eval_ids "$PROTO" \
    --pred "cascade=$pred" \
    2>&1 | tee "$FINAL/cascade_lora/official_eval.txt"
}

phase_a0_rerun() {
  setup
  run_logged results/a0_rerun.log \
    python scripts/connector/train_connector.py \
      --train_file "$TRAIN_EN" \
      --eval_ids "$PROTO" \
      --cache_dir results/cache_h100_bf16 \
      --out_dir results/v3_h100 \
      --arch linear \
      --epochs 12 \
      --batch 16 \
      --seed 13 \
      --decoder bio \
      --bio \
      --gate_consistency \
      --contrastive \
      --tversky \
      --no_type_loss \
      --device "$DEVICE"
}

phase_a2_frozen() {
  setup
  run_logged results/a2_frozen.log \
    python scripts/connector/train_lora.py \
      --model_id "$MODEL" \
      --train_file "$TRAIN_EN" \
      --image_dir "$IMG_DIR" \
      --eval_ids "$PROTO" \
      --out_dir results/lora_h100/a2_frozen \
      --arch linear \
      --seed 13 \
      --epochs 5 \
      --freeze_lora_epochs 99 \
      --lora_from "$LORA_FROM" \
      --device "$DEVICE"
}

phase_a2_finalist_seeds() {
  setup
  mkdir -p results/lora_h100/a2
  for seed in 42 77; do
    run_logged "$LOG_DIR/a2_seed_${seed}.log" \
      python scripts/connector/train_lora.py \
        --model_id "$MODEL" \
        --train_file "$TRAIN_EN" \
        --image_dir "$IMG_DIR" \
        --eval_ids "$PROTO" \
        --out_dir results/lora_h100/a2 \
        --arch linear \
        --seed "$seed" \
        --epochs 5 \
        --lora_from "$LORA_FROM" \
        --device "$DEVICE"
  done
}

case "$PHASE" in
  a2-train-s13) phase_a2_train_s13 ;;
  a2-backup) phase_a2_backup ;;
  cascade-screen) phase_cascade_screen ;;
  a0-rerun) phase_a0_rerun ;;
  a2-frozen) phase_a2_frozen ;;
  a2-finalist-seeds) phase_a2_finalist_seeds ;;
  old-queue) phase_a0_rerun; phase_a2_frozen; phase_a2_finalist_seeds ;;
  all) phase_a2_backup; phase_cascade_screen ;;
  *) echo "ERROR: unknown --phase $PHASE" >&2; exit 2 ;;
esac
