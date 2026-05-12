#!/usr/bin/env bash
# Download the full ProteinGym substitutions DMS database (~1 GB, 217 proteins).
#
# Source: https://proteingym.org/download
# Citation: Notin et al., NeurIPS 2023, "ProteinGym: Large-Scale Benchmarks..."
#
# Usage:
#   bash scripts/download_proteingym.sh
#
# Result: extracts to data/proteingym_full/

set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="$REPO_ROOT/data/proteingym_full"
TMP_ZIP="$OUT_DIR.zip"

# Official ProteinGym substitutions URL — update if the hosting changes.
URL="https://marks.hms.harvard.edu/proteingym/ProteinGym_substitutions.zip"

echo "Target: $OUT_DIR"
mkdir -p "$OUT_DIR"

if [ ! -f "$TMP_ZIP" ]; then
  echo "Downloading from $URL ..."
  if command -v curl >/dev/null; then
    curl -L -o "$TMP_ZIP" "$URL"
  else
    wget -O "$TMP_ZIP" "$URL"
  fi
fi

echo "Extracting ..."
unzip -q "$TMP_ZIP" -d "$OUT_DIR"
rm -f "$TMP_ZIP"

echo "Done."
ls "$OUT_DIR" | head -5
echo "..."
echo "Total files: $(ls "$OUT_DIR" | wc -l)"
