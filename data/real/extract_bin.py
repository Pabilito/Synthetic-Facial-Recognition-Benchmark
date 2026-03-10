"""
extract_bin.py
─────────────────────────────────────────────────────────────────────────────
Converts an InsightFace .bin evaluation file (cfp_fp.bin, lfw.bin,
agedb_30.bin, calfw.bin, cplfw.bin, …) into:

  • A flat folder of 112×112 JPEG images
  • An annotation .txt file compatible with eval_real.py / eval_utils.py

RUN FROM:  data/real/   (same directory as your other ann files)

USAGE
─────
    # CFP-FP (most common case)
    python extract_bin.py cfp_fp.bin

    # Explicit output folder and ann file
    python extract_bin.py cfp_fp.bin --img-dir cfp_fp_112x112 --out cfp_fp_ann.txt

    # Re-extract any other bin in the same format
    python extract_bin.py lfw.bin --img-dir lfw_112x112 --out lfw_ann.txt

ANNOTATION FORMAT  (identical to your existing lfw_ann.txt etc.)
─────────────────
    <label> <img1_rel_path> <img2_rel_path>

    Paths are relative to data/real/ (= CWD when running eval_real.py).

    Example:
        1 cfp_fp_112x112/00000_0.jpg cfp_fp_112x112/00000_1.jpg
        0 cfp_fp_112x112/00350_0.jpg cfp_fp_112x112/01234_1.jpg

BIN FORMAT  (InsightFace standard)
───────────
    pickle.load(f) → (img_list, label_list)
      img_list   : list of bytes objects  (length = 2 × N pairs)
      label_list : list/array of int      (length = N pairs)

    img_list[2*i]   = JPEG bytes for image 1 of pair i
    img_list[2*i+1] = JPEG bytes for image 2 of pair i
    label_list[i]   = 1 (genuine) or 0 (impostor)

REGISTER IN eval_real.py
─────────────────────────
    DATASETS: dict[str, Path] = {
        ...
        "CFP-FP": VAL_DIR / "cfp_fp_ann.txt",
    }
"""

import argparse
import io
import pickle
import sys
from pathlib import Path

from PIL import Image


def load_bin(bin_path: Path):
    """
    Load an InsightFace .bin file.

    Returns (imgs, labels) where:
      imgs   : list of bytes  (length 2*N)
      labels : list of int    (length N)
    """
    with open(bin_path, "rb") as f:
        imgs, labels = pickle.load(f, encoding="bytes")

    # labels may be a numpy array — normalise to plain list of ints
    try:
        labels = [int(x) for x in labels]
    except TypeError:
        raise ValueError(f"Unexpected label format in {bin_path.name}")

    if len(imgs) != 2 * len(labels):
        raise ValueError(
            f"Mismatch: {len(imgs)} images but {len(labels)} labels "
            f"(expected {2*len(labels)} images). "
            f"This .bin may use a different format."
        )

    return imgs, labels


def extract(
    bin_path:   Path,
    img_dir:    Path,   # absolute path where images will be written
    ann_prefix: str,    # prefix used in ann file (relative to data/real/)
    out_path:   Path,   # output annotation file
    overwrite:  bool,
) -> None:

    print(f"Loading {bin_path.name} …")
    imgs, labels = load_bin(bin_path)
    n_pairs = len(labels)
    print(f"  {n_pairs} pairs  ({sum(labels)} genuine + {n_pairs-sum(labels)} impostor)")

    img_dir.mkdir(parents=True, exist_ok=True)

    ann_lines: list[str] = []
    n_written = 0

    for i, label in enumerate(labels):
        name1 = f"{i:05d}_0.jpg"
        name2 = f"{i:05d}_1.jpg"
        path1 = img_dir / name1
        path2 = img_dir / name2

        # Write images (skip if already exist and not overwriting)
        for raw, path in ((imgs[2*i], path1), (imgs[2*i+1], path2)):
            if overwrite or not path.exists():
                try:
                    img = Image.open(io.BytesIO(raw)).convert("RGB")
                    img.save(path, "JPEG", quality=95)
                    n_written += 1
                except Exception as e:
                    print(f"  [WARN] pair {i}: failed to decode image — {e}")
                    continue

        ann_lines.append(f"{label} {ann_prefix}/{name1} {ann_prefix}/{name2}")

        if (i + 1) % 1000 == 0:
            print(f"  {i+1}/{n_pairs} pairs processed …")

    # Write annotation file
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(ann_lines) + "\n", encoding="utf-8")

    n_genuine  = sum(labels)
    n_impostor = n_pairs - n_genuine
    print(f"\nImages written : {n_written}  (in {img_dir})")
    print(f"Annotation     : {out_path}")
    print(f"  {n_pairs} pairs  ({n_genuine} genuine + {n_impostor} impostor)")
    print(f"\nAdd to DATASETS in eval_real.py:")
    ds_key = out_path.stem.upper().replace("_", "-")  # cfp_fp_ann → CFP-FP-ANN, cleaned below
    # Derive a clean dataset key from filename: cfp_fp_ann.txt → CFP-FP
    stem = out_path.stem                      # e.g. cfp_fp_ann
    if stem.endswith("_ann"):
        stem = stem[:-4]                      # cfp_fp
    ds_key = stem.upper().replace("_", "-")   # CFP-FP
    print(f'    "{ds_key}": VAL_DIR / "{out_path.name}",')


def main() -> None:
    cwd = Path.cwd()

    parser = argparse.ArgumentParser(
        description="Extract InsightFace .bin → images + annotation txt. "
                    "Run from data/real/.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("bin", metavar="BIN_FILE",
                        help="Path to the .bin file (e.g. cfp_fp.bin). "
                             "Relative to CWD or absolute.")
    parser.add_argument("--img-dir", default=None, metavar="DIR",
                        help="Output image folder. Relative to CWD. "
                             "Default: <bin_stem>_112x112  "
                             "(e.g. cfp_fp_112x112/)")
    parser.add_argument("--out", default=None, metavar="PATH",
                        help="Output annotation file. Relative to CWD. "
                             "Default: <bin_stem>_ann.txt  "
                             "(e.g. cfp_fp_ann.txt)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Re-extract images even if they already exist.")
    args = parser.parse_args()

    bin_path = Path(args.bin)
    if not bin_path.is_absolute():
        bin_path = cwd / bin_path
    if not bin_path.exists():
        print(f"[ERROR] .bin file not found: {bin_path}")
        sys.exit(1)

    bin_stem = bin_path.stem   # e.g. "cfp_fp"

    img_dir_name = args.img_dir or f"{bin_stem}_112x112"
    img_dir      = (cwd / img_dir_name).resolve()
    ann_prefix   = img_dir_name.rstrip("/")   # relative to CWD = VAL_DIR

    out_name = args.out or f"{bin_stem}_ann.txt"
    out_path = (cwd / out_name).resolve()

    print(f"CWD        : {cwd}")
    print(f"Bin file   : {bin_path}")
    print(f"Image dir  : {img_dir}")
    print(f"Ann prefix : '{ann_prefix}'")
    print(f"Output     : {out_path}")
    print(f"Overwrite  : {args.overwrite}")
    print()

    extract(bin_path, img_dir, ann_prefix, out_path, args.overwrite)


if __name__ == "__main__":
    main()
