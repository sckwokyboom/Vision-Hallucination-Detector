#!/bin/bash
# Full-scale connector protocol on the Mac. Waits for the feature caches, then runs
# everything on the TUNE split only (held-out is never touched here).
# All outputs -> results/fullscale/ (durable). Resumable: completed steps are skipped.
set -uo pipefail
cd "$(dirname "$0")/../.."
source .venv/bin/activate
T=../Shroom-Vision/distrib/shroom-vision.train.en.labeled.jsonl
C=results/connector_cache_mac
CN=results/connector_cache_mac_noimg
P=splits/en.eval_protocol.json
O=results/fullscale
mkdir -p "$O"
LOG=results/fullscale/RUN.log
step() { echo "[$(date +%H:%M:%S)] $1" >> "$LOG"; }

# ---- wait for the with-image cache ----
step "waiting for with-image cache..."
until grep -q "WITHIMG_FULL_DONE" /tmp/fullcache.log 2>/dev/null || ! pgrep -f extract_features_mlx >/dev/null; do sleep 60; done
step "with-image cache ready ($(ls $C | wc -l) files)"

TRAIN="python scripts/connector/train_connector.py --train_file $T --eval_ids $P --batch 8 --epochs 12"

# ---- 1) main: 3 seeds, linear + full decoder ----
for s in 13 42 77; do
  [ -f "$O/summary_linear_gate_hyst_s$s.json" ] && continue
  step "main linear seed $s"
  $TRAIN --cache_dir $C --out_dir $O --arch linear --seed $s >> "$LOG" 2>&1
done

# ---- 2) decoder ablation (seed 13) ----
for dec in simple gate hyst; do
  [ -f "$O/summary_linear_${dec}_s13.json" ] && continue
  step "decoder ablation: $dec"
  $TRAIN --cache_dir $C --out_dir $O --arch linear --seed 13 --decoder $dec >> "$LOG" 2>&1
done
[ -f "$O/summary_linear_gate_hyst_nogru_s13.json" ] || { step "ablation: no-gru"; \
  $TRAIN --cache_dir $C --out_dir $O --arch linear --seed 13 --no_gru >> "$LOG" 2>&1; }

# ---- 3) learning curve (nested, seed 13) ----
for n in 100 300 1000; do
  [ -f "$O/lc_$n/summary_linear_gate_hyst_s13.json" ] && continue
  step "learning curve n=$n"
  $TRAIN --cache_dir $C --out_dir $O/lc_$n --arch linear --seed 13 --max_train $n >> "$LOG" 2>&1
done

# ---- 4) residual/full connector ablation (seed 13, smoke-scale V cache only covers 350) ----
[ -f "$O/summary_connector_gate_hyst_s13.json" ] || { step "connector ablation"; \
  $TRAIN --cache_dir $C --out_dir $O --arch connector --seed 13 >> "$LOG" 2>&1; }

# ---- 5) text-only control (needs the no-image cache) ----
step "waiting for text-only cache..."
until grep -q "ALL_CACHE_DONE" /tmp/fullcache.log 2>/dev/null || ! pgrep -f extract_features_mlx >/dev/null; do sleep 60; done
step "text-only cache ready ($(ls $CN 2>/dev/null | wc -l) files)"
for s in 13 42; do
  [ -f "$O/textonly/summary_linear_gate_hyst_s$s.json" ] && continue
  step "text-only control seed $s"
  $TRAIN --cache_dir $CN --out_dir $O/textonly --arch linear --seed $s >> "$LOG" 2>&1
done

step "ALL_FULLSCALE_DONE"
echo "ALL_FULLSCALE_DONE" >> "$LOG"
