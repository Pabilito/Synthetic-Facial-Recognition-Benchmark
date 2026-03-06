#!/bin/bash
# Extract all identities from each dataset into DatasetName/Identity/Image format.

BASE="$(cd "$(dirname "$0")" && pwd)"

# ── Helper: extract all identities from a zip, flatten to Identity/Image ──
extract_zip() {
    local archive="$1" prefix="$2" name="$3"
    local dest="$BASE/$name"
    mkdir -p "$dest"

    local ids
    ids=$(unzip -l "$archive" | awk '{print $4}' \
          | grep "^${prefix}[^/]*/\$" \
          | sed "s|^${prefix}||;s|/\$||" \
          | sort -u)
    echo "[$name] $(echo "$ids" | wc -l) identities"

    while IFS= read -r id; do
        mkdir -p "$dest/$id"
        unzip -q "$archive" "${prefix}${id}/*" -d "$dest/$id" 2>/dev/null
        # Flatten: move files up if unzip created subdirs
        find "$dest/$id" -mindepth 2 -type f | while read -r f; do
            mv "$f" "$dest/$id/"
        done
        find "$dest/$id" -mindepth 1 -type d -empty -delete
    done <<< "$ids"
}

# ── Helper: extract all identities from a tar.gz, flatten to Identity/Image ──
extract_tar() {
    local archive="$1" prefix="$2" name="$3"
    local dest="$BASE/$name"
    mkdir -p "$dest"

    local ids
    ids=$(tar -tf "$archive" \
          | grep "^${prefix}[^/]*/\$" \
          | sed "s|^${prefix}||;s|/\$||" \
          | sort -u)
    echo "[$name] $(echo "$ids" | wc -l) identities"

    while IFS= read -r id; do
        mkdir -p "$dest/$id"
        tar -xf "$archive" -C "$dest/$id" --strip-components=$(echo "$prefix" | tr -cd '/' | wc -c) \
            "${prefix}${id}/" 2>/dev/null
    done <<< "$ids"
}

echo "[CemiFace]"
extract_zip "$BASE/CemiFace.zip" "pretrained_cos_500k_real/" "CemiFace"

echo "[DCFace]"
extract_zip "$BASE/DCFace.zip" "DCFace/" "DCFace"

echo "[IDiffFace]"
extract_zip "$BASE/IDiffFace.zip" "" "IDiffFace"

echo "[SFace2]"
extract_zip "$BASE/SFace2.zip" "SFace2_Center/" "SFace2"

echo "[Control10k]"
mkdir -p "$BASE/Control10k"
unzip -l "$BASE/Control10k.zip" | awk '{print $4}' \
    | grep -E "^ControlFace10k/[^/]+/[^/]+/[^/]+/identity-[^/]+/$" \
    | sort -u \
    | while IFS= read -r idpath; do
        id=$(basename "$idpath")
        mkdir -p "$BASE/Control10k/$id"
        unzip -q "$BASE/Control10k.zip" "${idpath}*" -d "$BASE/Control10k/$id"
        find "$BASE/Control10k/$id" -mindepth 2 -type f | while read -r f; do
            mv "$f" "$BASE/Control10k/$id/"
        done
        find "$BASE/Control10k/$id" -mindepth 1 -type d -empty -delete
      done
echo "  -> done"

echo "[Vec2Face]"
mkdir -p "$BASE/Vec2Face"
unzip -q "$BASE/Vec2Face.zip" -d "$BASE/Vec2Face"
echo "  -> extracted (flat archive, verify structure manually)"

echo "[IDNet]"
mkdir -p "$BASE/IDNet"
unzip -q "$BASE/IDNet.zip" -d "$BASE/IDNet"
echo "  -> extracted (verify manually)"

echo "[Digi2Real]"
extract_tar "$BASE/Digi2Real.tar.gz" "Digi2Real/digi2real/" "Digi2Real"

echo "[HyperFace]"
extract_tar "$BASE/HyperFace.tar.gz" "HyperFace/HyperFace_10k_StyleGAN/" "HyperFace"

echo "[LangevinDisco]"
extract_tar "$BASE/LangevinDisco.tar.gz" \
    "Synthetics-DisCo/sg2_n10k_arc_r14_disco_n64_rl14/images_arcface_112x112/" "LangevinDisco"

echo "[MorphFace]"
extract_tar "$BASE/MorphFace.tar.gz" \
    "3dg_50pose_ca_casia_250k_test_4_std18_kpshape_tc05/images/" "MorphFace"

echo "[Synface]"
extract_tar "$BASE/Synface.tar.gz" \
    "CASIA-Generated-10k-100-01-MixID-Align/" "Synface"

echo "[SynMultiPie]"
extract_tar "$BASE/SynMultiPie.tar.gz" \
    "SynMulti-PIE/images_arcface_112x112_regen/" "SynMultiPie"

echo ""
echo "All done. Output in $BASE"
