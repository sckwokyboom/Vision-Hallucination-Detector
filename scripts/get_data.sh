#!/bin/bash
# Download + unpack the SHROOM-visions dataset. Idempotent and resumable: re-running it
# after an interrupted download continues the transfer, and already-unpacked data is left
# alone unless you pass --force.
#
#   bash scripts/get_data.sh [--data-dir DIR] [--no-images] [--force] [--keep-archives]
#
#   --data-dir DIR    where the dataset lives (default ../Shroom-Vision, i.e. next to the repo)
#   --no-images       annotations only (16 MB) — enough for splits, metrics and the tables,
#                     NOT enough for feature extraction or any VLM run
#   --force           re-download and re-unpack even if the data is already there
#   --keep-archives   keep the .zip/.tar.gz after unpacking (default: keep, see NOTE)
#
# Produces:
#   DIR/distrib/shroom-vision.{train,test}.{en,fr,it,zh}.*.jsonl     (16 MB)
#   DIR/images/<image_name>.jpg                                      (1607 files, ~2.5 GB)
#
# The image names in the JSONL are relative to DIR/images, so the tarball's internal
# `shroom-vis-images/` prefix is stripped on extraction.
#
# NOTE: on a cluster, run this on a login node — compute nodes usually have no outbound
# network. The archives are kept by default so a re-extract needs no network at all.
set -euo pipefail
cd "$(dirname "$0")/.."

DATA_URL="https://a3s.fi/mickusti-2007780-pub/shroom-visions-data.zip"
IMG_URL="https://a3s.fi/mickusti-2007780-pub/shroom-visions-images.tar.gz"
# The task page also advertises participant-kit.zip and extra-info.tar.gz under
# a3s.fi/shroom-visions/ — both are marked BROKEN LINK there and 404 at the time of writing.

DATA_DIR="../Shroom-Vision"
WANT_IMAGES=1
FORCE=0
NEED_GB=6                                  # 2.4 GB archive + ~2.5 GB extracted + headroom
EXPECT_JSONL=8
MIN_IMAGES=1000                            # the real count is 1607

while [ $# -gt 0 ]; do
  case "$1" in
    --data-dir) DATA_DIR="$2"; shift 2 ;;
    --no-images) WANT_IMAGES=0; NEED_GB=1; shift ;;
    --force) FORCE=1; shift ;;
    --keep-archives) shift ;;              # accepted for symmetry; archives are kept anyway
    -h|--help) sed -n '2,25p' "$0"; exit 0 ;;
    *) echo "unknown option: $1 (try --help)" >&2; exit 2 ;;
  esac
done

mkdir -p "$DATA_DIR"
DATA_DIR="$(cd "$DATA_DIR" && pwd)"
ZIP="$DATA_DIR/shroom-visions-data.zip"
TAR="$DATA_DIR/shroom-visions-images.tar.gz"

free_gb() { df -Pk "$1" | awk 'NR==2 {printf "%d", $4/1024/1024}'; }
# Counting things that may not exist yet: `ls`/`find` fail on a missing path, and under
# `set -e` + `pipefail` a failing command substitution would kill the script.
n_jsonl_now() { ls "$DATA_DIR"/distrib/*.jsonl 2>/dev/null | wc -l | tr -d ' ' || true; }
n_img_now()   { find "$DATA_DIR/images" -type f 2>/dev/null | wc -l | tr -d ' ' || true; }
have=$(free_gb "$DATA_DIR")
if [ "$have" -lt "$NEED_GB" ]; then
  echo "ERROR: only ${have} GB free on $DATA_DIR, need ~${NEED_GB} GB." >&2
  echo "       Point --data-dir at a bigger filesystem (scratch, not \$HOME)." >&2
  exit 1
fi

fetch() {                                   # fetch URL DEST — resumable, atomic-ish
  local url="$1" dest="$2"
  if [ -s "$dest" ] && [ "$FORCE" -eq 0 ]; then
    echo "[skip] $(basename "$dest") already downloaded ($(du -h "$dest" | cut -f1))"
    return
  fi
  [ "$FORCE" -eq 1 ] && rm -f "$dest"
  echo "[get ] $url"
  if command -v curl >/dev/null 2>&1; then
    curl -fL --retry 5 --retry-delay 5 -C - -o "$dest" "$url"
  elif command -v wget >/dev/null 2>&1; then
    wget -c -O "$dest" "$url"
  else
    echo "ERROR: neither curl nor wget available" >&2; exit 1
  fi
}

# ---------------------------------------------------------------- annotations (16 MB)
n_jsonl=$(n_jsonl_now)
if [ "${n_jsonl:-0}" -ge "$EXPECT_JSONL" ] && [ "$FORCE" -eq 0 ]; then
  echo "[skip] distrib/ already unpacked ($n_jsonl jsonl files)"
else
  fetch "$DATA_URL" "$ZIP"
  unzip -tq "$ZIP" >/dev/null || { echo "ERROR: $ZIP is corrupt — rerun with --force" >&2; exit 1; }
  unzip -oq "$ZIP" -d "$DATA_DIR"
  n_jsonl=$(n_jsonl_now)
  echo "[ok  ] distrib/ -> $n_jsonl jsonl files"
fi

# ---------------------------------------------------------------- images (2.4 GB)
if [ "$WANT_IMAGES" -eq 1 ]; then
  n_img=$(n_img_now)
  if [ "${n_img:-0}" -ge "$MIN_IMAGES" ] && [ "$FORCE" -eq 0 ]; then
    echo "[skip] images/ already unpacked ($n_img files)"
  else
    fetch "$IMG_URL" "$TAR"
    echo "[test] verifying the archive (2.4 GB, ~30 s)..."
    gzip -t "$TAR" || { echo "ERROR: $TAR is corrupt — rerun with --force" >&2; exit 1; }
    bash "$(dirname "$0")/extract_images.sh" "$TAR" "$DATA_DIR/images"
    n_img=$(n_img_now)
  fi
else
  echo "[skip] images (--no-images)"
fi

# ---------------------------------------------------------------- report
echo
echo "dataset ready at $DATA_DIR"
echo "  distrib/  $(n_jsonl_now) jsonl"
[ "$WANT_IMAGES" -eq 1 ] && echo "  images/   $(n_img_now) files" || true
cat <<EOF

next:
  bash scripts/connector/run_train_h100.sh --data-dir "$DATA_DIR" --gpu 0
EOF
