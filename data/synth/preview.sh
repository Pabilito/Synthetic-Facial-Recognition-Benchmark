#!/bin/bash
# Preview internal structure of all dataset archives.

BASE="$(cd "$(dirname "$0")" && pwd)"
LOG="$BASE/structure_preview.txt"

> "$LOG"

for archive in "$BASE"/*.zip; do
    [ -f "$archive" ] || continue
    name=$(basename "$archive" .zip)
    echo "=== $name (zip) ===" | tee -a "$LOG"
    unzip -l "$archive" | awk '{print $4}' | grep -v '^$' | head -20 | tee -a "$LOG"
    echo "" | tee -a "$LOG"
done

for archive in "$BASE"/*.tar.gz; do
    [ -f "$archive" ] || continue
    name=$(basename "$archive" .tar.gz)
    echo "=== $name (tar.gz) ===" | tee -a "$LOG"
    tar -tf "$archive" | head -20 | tee -a "$LOG"
    echo "" | tee -a "$LOG"
done

echo "Preview saved to $LOG"
