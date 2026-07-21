#!/bin/bash
# Extract the image archive once into ../Shroom-Vision/images/
set -eu
ARCHIVE="${1:-../Shroom-Vision/shroom-visions-images.tar.gz}"
DEST="${2:-../Shroom-Vision/images}"
mkdir -p "$DEST"
echo "Extracting $ARCHIVE -> $DEST ..."
tar -xzf "$ARCHIVE" -C "$DEST" --strip-components=0
echo "Done. Image count: $(find "$DEST" -type f | wc -l)"
