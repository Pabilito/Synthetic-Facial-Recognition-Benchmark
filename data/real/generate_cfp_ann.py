"""
generate_cfp_ann.py
─────────────────────────────────────────────────────────────────────────────
Converts the official CFP-FP (or CFP-FF) protocol files into a flat
annotation file compatible with eval_real.py / eval_utils.py.

RUN FROM:  data/real/
─────────────────────
This script is designed to be run from the data/real/ directory, which is
where eval_real.py's VAL_DIR points.  All default paths are relative to CWD.

After downloading the Kaggle dataset (chinafax/cfpw-dataset) and unzipping,
your data/real/ directory should look like:

    data/real/
      cfp-dataset/                     ← unzipped Kaggle archive
        Data/
          Images/
            001/
              frontal/  01.jpg … 10.jpg
              profile/  01.jpg …  4.jpg
            002/ … 500/
        Protocol/
          Pair_list_F.txt
          Pair_list_P.txt
          Split/
            FP/
              1/  same.txt  diff.txt
              2/ … 10/
            FF/
              1/  same.txt  diff.txt
              2/ … 10/
      lfw_ann.txt
      calfw_ann.txt
      …

USAGE (from data/real/)
───────────────────────
    # Standard — uses cfp-dataset/ in CWD, writes cfp_fp_ann.txt in CWD
    python generate_cfp_ann.py

    # Explicit paths (all relative to data/real/ or absolute)
    python generate_cfp_ann.py --cfp-root cfp-dataset --out cfp_fp_ann.txt

    # Frontal-Frontal instead of Frontal-Profile
    python generate_cfp_ann.py --ff

    # Skip image existence check (faster; useful if images are on slow mount)
    python generate_cfp_ann.py --no-verify

    # If your images were pre-aligned into a separate folder:
    python generate_cfp_ann.py --img-dir cfp_fp_112x112
    # annotation paths will then be:  cfp_fp_112x112/001/frontal/01.jpg

ANNOTATION FORMAT (same as lfw_ann.txt / calfw_ann.txt)
────────────────────────────────────────────────────────
    <label> <img1_rel_path> <img2_rel_path>
    label 1 = genuine pair (same identity)
    label 0 = impostor pair (different identities)

    Paths are relative to data/real/ (CWD when running eval_real.py).

    Example lines:
        1 cfp-dataset/Data/Images/049/frontal/06.jpg cfp-dataset/Data/Images/049/profile/01.jpg
        0 cfp-dataset/Data/Images/049/frontal/06.jpg cfp-dataset/Data/Images/103/profile/03.jpg

FOLD ORDERING
─────────────
eval_utils.eval_10fold() uses KFold(n_splits=10, shuffle=False), which cuts
the pair list into 10 consecutive equal-sized blocks.  The CFP protocol also
has 10 equal splits.  For these to agree, pairs are written split-by-split
(split 1 first, then split 2, …, then split 10) so that KFold block i
corresponds exactly to protocol split i+1.

REGISTER IN eval_real.py
─────────────────────────
    DATASETS: dict[str, Path] = {
        "LFW":      VAL_DIR / "lfw_ann.txt",
        "CALFW":    VAL_DIR / "calfw_ann.txt",
        ...
        "CFP-FP":   VAL_DIR / "cfp_fp_ann.txt",   ← add this line
    }
"""

import argparse
import sys
from pathlib import Path


# ── Protocol parsers ──────────────────────────────────────────────────────────

def load_pair_list(path: Path) -> dict[int, str]:
    """
    Parse Pair_list_F.txt or Pair_list_P.txt.

    The Kaggle CFP-W repack uses space as separator and prefixes paths with
    '../Data/Images/', e.g.:
        1 ../Data/Images/001/frontal/01.jpg
        2 ../Data/Images/001/frontal/02.jpg

    The original cfp-dataset.zip used commas with bare relative paths:
        1,001/frontal/01.jpg

    Both formats are handled.  The leading '../Data/Images/' prefix (and any
    other leading '../' components) are stripped so the returned paths are
    always relative to the Data/Images/ root, e.g. '001/frontal/01.jpg'.

    Returns {1-based-integer-index: relative-path-string}.
    """
    mapping: dict[int, str] = {}
    with open(path, encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, 1):
            line = raw.strip()
            if not line:
                continue

            # Detect separator: comma or whitespace
            comma = line.find(",")
            if comma != -1:
                sep_idx  = comma
                rel_part = line[comma + 1:].strip()
                idx_part = line[:comma].strip()
            else:
                parts = line.split(None, 1)   # split on first whitespace
                if len(parts) < 2:
                    print(f"  [WARN] {path.name}:{lineno}: cannot parse — skipping: {line!r}")
                    continue
                idx_part, rel_part = parts[0], parts[1].strip()

            try:
                idx = int(idx_part)
            except ValueError:
                print(f"  [WARN] {path.name}:{lineno}: non-integer index — skipping: {line!r}")
                continue

            if not rel_part:
                print(f"  [WARN] {path.name}:{lineno}: empty path — skipping")
                continue

            # Strip any leading '../' components so the path is relative to
            # Data/Images/, matching what cfp_root/Data/Images/<rel> resolves to.
            # e.g.  '../Data/Images/001/frontal/01.jpg'  →  '001/frontal/01.jpg'
            #        'Data/Images/001/frontal/01.jpg'     →  '001/frontal/01.jpg'
            #        '001/frontal/01.jpg'                 →  '001/frontal/01.jpg'
            rel = rel_part.replace("\\", "/")
            for prefix in ("../Data/Images/", "Data/Images/", "../data/images/", "data/images/"):
                if rel.startswith(prefix):
                    rel = rel[len(prefix):]
                    break
            # Also strip any remaining leading dots/slashes
            while rel.startswith("../") or rel.startswith("./"):
                rel = rel.split("/", 1)[1]

            mapping[idx] = rel
    return mapping


def load_split_file(path: Path) -> list[tuple[int, int]]:
    """
    Parse same.txt or diff.txt.

    Format — one pair per line, whitespace-separated:
        <idx_a>  <idx_b>

    Returns list of (idx_a, idx_b) in file order.
    """
    pairs: list[tuple[int, int]] = []
    with open(path, encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, 1):
            line = raw.strip()
            if not line:
                continue
            parts = line.replace(",", " ").split()
            if len(parts) < 2:
                print(f"  [WARN] {path.name}:{lineno}: expected 2 tokens — skipping: {line!r}")
                continue
            try:
                pairs.append((int(parts[0]), int(parts[1])))
            except ValueError:
                print(f"  [WARN] {path.name}:{lineno}: non-integer indices — skipping: {line!r}")
    return pairs


# ── Core ──────────────────────────────────────────────────────────────────────

def build_ann(
    cfp_root:   Path,    # root of the cfp-dataset folder
    img_dir:    Path,    # where images actually live (for existence checks)
    ann_prefix: str,     # prefix prepended to every path written in the ann file
    out_path:   Path,    # output annotation file
    mode:       str,     # "FP" or "FF"
    verify:     bool,
) -> None:
    """
    Walk all 10 protocol splits in order (1 → 10) and emit one annotation
    line per pair so that KFold(shuffle=False) reconstructs the same folds.

    Within each split all genuine pairs come first (same.txt), then all
    impostor pairs (diff.txt) — matching the protocol file order and keeping
    each KFold block uniform (350 genuine + 350 impostor = 700 per fold).
    """
    proto = cfp_root / "Protocol"

    # ── Load index → image-path maps ─────────────────────────────────────────
    pf_path = proto / "Pair_list_F.txt"
    pp_path = proto / "Pair_list_P.txt"

    if not pf_path.exists():
        print(f"[ERROR] Not found: {pf_path}")
        sys.exit(1)

    frontal_map = load_pair_list(pf_path)
    print(f"  Pair_list_F.txt : {len(frontal_map)} frontal images")

    if mode == "FP":
        if not pp_path.exists():
            print(f"[ERROR] Not found: {pp_path}")
            sys.exit(1)
        profile_map = load_pair_list(pp_path)
        print(f"  Pair_list_P.txt : {len(profile_map)} profile images")
    else:
        # FF: both indices reference frontal images
        profile_map = frontal_map

    # ── Walk splits ───────────────────────────────────────────────────────────
    split_dir = proto / "Split" / mode
    if not split_dir.exists():
        print(f"[ERROR] Split directory not found: {split_dir}")
        sys.exit(1)

    # Detect subdirectory naming: '1'..'10'  OR  'split1'..'split10'
    def _find_sub(n: int):
        for name in (str(n), f"{n:02d}", f"split{n}", f"Split{n}"):
            p = split_dir / name
            if p.is_dir():
                return p
        return None

    split_subdirs = []
    for split_no in range(1, 11):
        sub = _find_sub(split_no)
        if sub is None:
            existing = sorted(p.name for p in split_dir.iterdir()) if split_dir.exists() else []
            print(f"[ERROR] Cannot find split {split_no} under {split_dir}")
            print(f"        Found subdirs: {existing}")
            sys.exit(1)
        split_subdirs.append(sub)

    all_lines   = []
    split_sizes = []
    n_warn = 0

    for split_no, sub in enumerate(split_subdirs, 1):
        same_file = sub / "same.txt"
        diff_file = sub / "diff.txt"

        if not same_file.exists() or not diff_file.exists():
            print(f"[ERROR] Split {split_no}: same.txt or diff.txt missing in {sub}")
            sys.exit(1)

        same_pairs = load_split_file(same_file)
        diff_pairs = load_split_file(diff_file)

        block: list[str] = []

        for label, pairs in ((1, same_pairs), (0, diff_pairs)):
            for idx_a, idx_b in pairs:
                # idx_a = frontal index; idx_b = profile index (or frontal for FF)
                if idx_a not in frontal_map:
                    print(f"  [WARN] Split {split_no}: frontal index {idx_a} "
                          f"not in Pair_list_F — skipping ({idx_a},{idx_b})")
                    n_warn += 1
                    continue
                if idx_b not in profile_map:
                    src = "Pair_list_F" if mode == "FF" else "Pair_list_P"
                    print(f"  [WARN] Split {split_no}: index {idx_b} "
                          f"not in {src} — skipping ({idx_a},{idx_b})")
                    n_warn += 1
                    continue

                rel_a = frontal_map[idx_a]
                rel_b = profile_map[idx_b]

                ann_a = f"{ann_prefix}/{rel_a}" if ann_prefix else rel_a
                ann_b = f"{ann_prefix}/{rel_b}" if ann_prefix else rel_b

                # Guard: whitespace breaks parse_ann (which splits on whitespace)
                for p in (ann_a, ann_b):
                    if any(c in p for c in (" ", "\t")):
                        print(f"[ERROR] Path contains whitespace — "
                              f"this would corrupt parse_ann: {p!r}")
                        sys.exit(1)

                if verify:
                    for rel in (rel_a, rel_b):
                        if not (img_dir / rel).exists():
                            print(f"  [WARN] Image not found: {img_dir / rel}")
                            n_warn += 1

                block.append(f"{label} {ann_a} {ann_b}")

        split_sizes.append(len(block))
        all_lines.extend(block)
        n_same = sum(1 for l in block if l[0] == "1")
        n_diff = len(block) - n_same
        print(f"  Split {split_no:2d}: {len(block):4d} pairs  "
              f"({n_same} genuine + {n_diff} impostor)")

    # ── Fold-size sanity check ────────────────────────────────────────────────
    if len(set(split_sizes)) > 1:
        print(f"\n  [WARN] Unequal split sizes: {split_sizes}")
        print(f"         KFold(shuffle=False) assumes equal-sized blocks.")
        print(f"         Fold boundaries may not match the protocol splits.")
    else:
        print(f"\n  ✓ All 10 splits have {split_sizes[0]} pairs each — "
              f"KFold blocks will align with protocol splits.")

    # ── Write output ──────────────────────────────────────────────────────────
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(all_lines) + "\n", encoding="utf-8")

    n_total    = len(all_lines)
    n_genuine  = sum(1 for l in all_lines if l[0] == "1")
    n_impostor = n_total - n_genuine

    print(f"\n  Total: {n_total} pairs  "
          f"({n_genuine} genuine + {n_impostor} impostor)")
    if n_warn:
        print(f"  [WARN] {n_warn} warning(s) above — review before running eval_real.")
    print(f"  Written → {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    # Default root is relative to CWD (i.e. data/real/) since the Kaggle zip
    # unpacks to a folder named "cfp-dataset" inside wherever you extract it.
    cwd = Path.cwd()

    parser = argparse.ArgumentParser(
        description="Generate cfp_fp_ann.txt for eval_real.py. "
                    "Run from data/real/.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--cfp-root", default="cfp-dataset", metavar="DIR",
        help="Root of the CFP dataset (contains Data/ and Protocol/). "
             "Relative to CWD or absolute. "
             "Default: cfp-dataset  (i.e. data/real/cfp-dataset/)",
    )
    parser.add_argument(
        "--img-dir", default=None, metavar="DIR",
        help="Directory that holds the images Pair_list_*.txt references. "
             "Used for existence checks and to derive the path prefix written "
             "into the annotation file. "
             "Relative to CWD or absolute. "
             "Default: <cfp-root>/Data/Images  (raw dataset, no prefix added).\n"
             "Set to e.g. 'cfp_fp_112x112' if you have pre-aligned images there; "
             "annotation paths will then start with 'cfp_fp_112x112/...'.",
    )
    parser.add_argument(
        "--out", default=None, metavar="PATH",
        help="Output annotation file path. Relative to CWD or absolute. "
             "Default: cfp_fp_ann.txt  (or cfp_ff_ann.txt for --ff).",
    )
    parser.add_argument(
        "--ff", action="store_true",
        help="Generate Frontal-Frontal pairs instead of Frontal-Profile (FP).",
    )
    parser.add_argument(
        "--no-verify", action="store_true",
        help="Skip checking that every referenced image file actually exists.",
    )
    args = parser.parse_args()

    mode   = "FF" if args.ff else "FP"
    suffix = "ff" if args.ff else "fp"

    cfp_root = Path(args.cfp_root)
    if not cfp_root.is_absolute():
        cfp_root = cwd / cfp_root
    cfp_root = cfp_root.resolve()

    if not cfp_root.exists():
        print(f"[ERROR] --cfp-root not found: {cfp_root}")
        print(f"        (CWD is {cwd})")
        sys.exit(1)

    # img_dir and ann_prefix
    if args.img_dir:
        img_dir_arg = Path(args.img_dir)
        if not img_dir_arg.is_absolute():
            img_dir_arg = cwd / img_dir_arg
        img_dir    = img_dir_arg.resolve()
        # ann_prefix is the path relative to CWD (= VAL_DIR in eval_real)
        try:
            ann_prefix = str(img_dir.relative_to(cwd))
        except ValueError:
            # img_dir is outside CWD — use its name and warn
            ann_prefix = img_dir.name
            print(f"  [WARN] --img-dir is outside CWD ({cwd}).")
            print(f"         Using folder name '{ann_prefix}' as annotation prefix.")
            print(f"         Ensure data/real/{ann_prefix}/ is accessible from eval_real.")
    else:
        # Default: images live inside cfp_root/Data/Images/
        img_dir    = cfp_root / "Data" / "Images"
        # Make ann_prefix the path from CWD to the Images folder, so that
        # eval_real (which also runs from data/real/) can resolve them.
        try:
            ann_prefix = str((cfp_root / "Data" / "Images").relative_to(cwd))
        except ValueError:
            ann_prefix = str(cfp_root / "Data" / "Images")
            print(f"  [WARN] cfp-root is outside CWD. Annotation paths will be absolute.")
            print(f"         Consider placing cfp-dataset/ inside data/real/.")

    # out_path
    out_path = Path(args.out) if args.out else cwd / f"cfp_{suffix}_ann.txt"
    if not out_path.is_absolute():
        out_path = cwd / out_path
    out_path = out_path.resolve()

    pose_label = "Frontal-Profile" if mode == "FP" else "Frontal-Frontal"
    print(f"CWD        : {cwd}")
    print(f"CFP root   : {cfp_root}")
    print(f"Mode       : {pose_label} ({mode})")
    print(f"Image dir  : {img_dir}")
    print(f"Ann prefix : '{ann_prefix}'")
    print(f"Output     : {out_path}")
    print(f"Verify     : {'no' if args.no_verify else 'yes'}")
    print()

    build_ann(
        cfp_root   = cfp_root,
        img_dir    = img_dir,
        ann_prefix = ann_prefix,
        out_path   = out_path,
        mode       = mode,
        verify     = not args.no_verify,
    )

    # ── Tell user what to add to eval_real.py ────────────────────────────────
    ann_name = out_path.name
    ds_key   = f"CFP-{'FP' if mode == 'FP' else 'FF'}"
    print(f"""
Add this entry to the DATASETS dict in eval_real.py:

    "{ds_key}": VAL_DIR / "{ann_name}",
""")


if __name__ == "__main__":
    main()
