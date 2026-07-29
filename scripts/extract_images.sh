#!/bin/bash
# Extract the image archive into ../Shroom-Vision/images/ (flat: one file per image_name).
#
#   bash scripts/extract_images.sh [ARCHIVE] [DEST]
#
# The tarball wraps everything in a `shroom-vis-images/` directory, but the JSONL
# `image_name` fields are bare filenames — so that prefix has to go. This detects the
# wrapper instead of assuming it, and is a no-op if the images are already extracted.
set -eu
ARCHIVE="${1:-../Shroom-Vision/shroom-visions-images.tar.gz}"
DEST="${2:-../Shroom-Vision/images}"
[ -f "$ARCHIVE" ] || { echo "ERROR: $ARCHIVE not found — run: bash scripts/get_data.sh" >&2; exit 1; }
mkdir -p "$DEST"

# Peek at the first entries: if they all share one top-level directory, strip it.
TOP=$(tar -tzf "$ARCHIVE" 2>/dev/null | head -20 | cut -d/ -f1 | sort -u || true)
STRIP=0
if [ "$(printf '%s\n' "$TOP" | wc -l | tr -d ' ')" = "1" ] && [ -n "$TOP" ]; then
  STRIP=1
  echo "Stripping wrapper directory: $TOP/"
fi

echo "Extracting $ARCHIVE -> $DEST ..."
tar -xzf "$ARCHIVE" -C "$DEST" --strip-components="$STRIP"
echo "Done. Image count: $(find "$DEST" -type f | wc -l | tr -d ' ')"
