"""
generate_ann.py
─────────────────────────────────────────────────────────────────────────────
Scans every dataset folder under data/synth/ and produces one annotation file
per dataset saved alongside the datasets in data/synth/<dataset_name>_ann.txt.

Annotation format (same as lfw_ann.txt):
    label img1_relative_path img2_relative_path
    e.g.  1 id_042/img0.jpg id_042/img7.jpg

Two folder conventions are supported and detected automatically:

  Convention A – identity sub-folders (most synthetic FR datasets):
    data/synth/<dataset>/
      <identity_0>/  img0.jpg  img1.png ...
      <identity_1>/  ...
    → same-identity pairs drawn within each sub-folder
    → different-identity pairs drawn across sub-folders

  Convention B – flat folder (mirrors LFW's numeric BMP layout):
    data/synth/<dataset>/
      0.bmp  1.bmp  2.bmp  3.bmp  ...
    → consecutive even/odd files treated as same-identity pairs
    → non-consecutive files treated as different-identity pairs

Reproducibility
───────────────
  • If data/synth/<dataset>_ann.txt already exists it is left untouched.
  • Pass --force to regenerate all annotation files from scratch.
  • A fixed SEED (in synth_utils.py) ensures the same pairs are produced on
    every fresh run.

Usage
─────
  python generate_ann.py            # skip datasets that already have .txt
  python generate_ann.py --force    # regenerate everything
"""

import argparse
import random
from pathlib import Path

from synth_utils import (
    SYNTH_ROOT, N_PAIRS, SEED, IMG_EXTS, parse_ann,
)

# ── File-system helpers ───────────────────────────────────────────────────────

def is_image(path: Path) -> bool:
    return path.suffix.lower() in IMG_EXTS


def make_relative(path: Path, base: Path) -> str:
    """Return a POSIX-style relative path string."""
    return path.relative_to(base).as_posix()


def scan_identity_folders(dataset_dir: Path) -> dict[str, list[Path]]:
    """
    Convention A: return {identity_name: [sorted image paths]}.
    Only sub-directories that contain at least one image are included.
    """
    identities: dict[str, list[Path]] = {}
    for subdir in sorted(dataset_dir.iterdir()):
        if not subdir.is_dir():
            continue
        imgs = sorted(f for f in subdir.iterdir() if f.is_file() and is_image(f))
        if imgs:
            identities[subdir.name] = imgs
    return identities


def scan_flat_folder(dataset_dir: Path) -> list[Path]:
    """Convention B: return sorted list of image files in a flat folder."""
    return sorted(f for f in dataset_dir.iterdir() if f.is_file() and is_image(f))


def detect_convention(dataset_dir: Path) -> str:
    """Return 'A' if the folder contains identity sub-folders, else 'B'."""
    subdirs_with_images = [
        d for d in dataset_dir.iterdir()
        if d.is_dir() and any(is_image(f) for f in d.iterdir() if f.is_file())
    ]
    return "A" if subdirs_with_images else "B"


# ── Pair generators ───────────────────────────────────────────────────────────

def generate_pairs_convention_a(
    identities: dict[str, list[Path]],
    dataset_dir: Path,
    n_pairs: int,
    rng: random.Random,
) -> list[tuple[str, str, int]]:
    """
    Build n_pairs/2 same-identity and n_pairs/2 different-identity pairs.
    Identities with fewer than 2 images are excluded from same-identity
    sampling but are still eligible as one side of a different-identity pair.
    """
    ids_multi = {k: v for k, v in identities.items() if len(v) >= 2}
    all_keys  = list(identities.keys())
    multi_keys = list(ids_multi.keys())

    if len(multi_keys) < 1:
        raise ValueError(
            f"{dataset_dir.name}: need at least 1 identity with ≥2 images "
            f"for same-identity pairs."
        )
    if len(all_keys) < 2:
        raise ValueError(
            f"{dataset_dir.name}: need at least 2 identities for "
            f"different-identity pairs."
        )

    half = n_pairs // 2

    # ── same-identity ────────────────────────────────────────────────────────
    same_pairs: list[tuple[str, str, int]] = []
    attempts = 0
    while len(same_pairs) < half and attempts < half * 20:
        attempts += 1
        iid  = rng.choice(multi_keys)
        a, b = rng.sample(ids_multi[iid], 2)
        same_pairs.append((make_relative(a, dataset_dir),
                           make_relative(b, dataset_dir), 1))

    # ── different-identity ───────────────────────────────────────────────────
    diff_pairs: list[tuple[str, str, int]] = []
    attempts = 0
    while len(diff_pairs) < half and attempts < half * 20:
        attempts += 1
        id1, id2 = rng.sample(all_keys, 2)
        a = rng.choice(identities[id1])
        b = rng.choice(identities[id2])
        diff_pairs.append((make_relative(a, dataset_dir),
                           make_relative(b, dataset_dir), 0))

    pairs = same_pairs + diff_pairs
    rng.shuffle(pairs)
    return pairs


def generate_pairs_convention_b(
    images: list[Path],
    dataset_dir: Path,
    n_pairs: int,
    rng: random.Random,
) -> list[tuple[str, str, int]]:
    """
    Flat folder convention: consecutive even/odd files are same-identity pairs
    (mirrors the LFW flat numbering).  Non-consecutive files are different.
    """
    half = n_pairs // 2

    # Build the pool of natural same-identity pairs
    natural_same = [
        (make_relative(images[i],   dataset_dir),
         make_relative(images[i+1], dataset_dir), 1)
        for i in range(0, len(images) - 1, 2)
    ]

    # Different-identity: any two images that do NOT form a natural pair
    all_indices = list(range(len(images)))
    diff_pairs: list[tuple[str, str, int]] = []
    attempts = 0
    while len(diff_pairs) < half and attempts < half * 20:
        attempts += 1
        a_idx, b_idx = rng.sample(all_indices, 2)
        if abs(a_idx - b_idx) == 1 and min(a_idx, b_idx) % 2 == 0:
            continue  # skip natural same-identity pair
        diff_pairs.append((make_relative(images[a_idx], dataset_dir),
                           make_relative(images[b_idx], dataset_dir), 0))

    same_pairs = rng.sample(natural_same, min(half, len(natural_same)))
    pairs = same_pairs + diff_pairs
    rng.shuffle(pairs)
    return pairs


# ── File existence verification ───────────────────────────────────────────────

def verify_and_filter(
    pairs: list[tuple[str, str, int]],
    dataset_dir: Path,
) -> list[tuple[str, str, int]]:
    """Remove any pair where either image file is missing on disk."""
    verified, skipped = [], 0
    for p1, p2, label in pairs:
        if (dataset_dir / p1).exists() and (dataset_dir / p2).exists():
            verified.append((p1, p2, label))
        else:
            skipped += 1
    if skipped:
        print(f"  [warn] Removed {skipped} pair(s) with missing files")
    return verified


# ── Ann-file writer ───────────────────────────────────────────────────────────

def write_ann(ann_file: Path, pairs: list[tuple[str, str, int]]) -> None:
    with open(ann_file, "w") as f:
        for p1, p2, label in pairs:
            f.write(f"{label} {p1} {p2}\n")


# ── Per-dataset entry point ───────────────────────────────────────────────────

def generate_for_dataset(
    ds_dir: Path,
    ann_file: Path,
    n_pairs: int,
    rng: random.Random,
    force: bool,
) -> None:
    print(f"━━━ {ds_dir.name} ━━━")

    if ann_file.exists() and not force:
        # Quick sanity-check: count lines
        existing = parse_ann(ann_file)
        print(f"  [skip] {ann_file.name} already exists ({len(existing)} pairs) "
              f"– use --force to regenerate\n")
        return

    if ann_file.exists() and force:
        print(f"  [force] Removing existing {ann_file.name}")
        ann_file.unlink()

    # Detect layout and generate pairs
    convention = detect_convention(ds_dir)
    print(f"  [conv] Convention {convention} detected")

    if convention == "A":
        identities = scan_identity_folders(ds_dir)
        print(f"  [info] {len(identities)} identity folder(s) found")
        try:
            pairs = generate_pairs_convention_a(identities, ds_dir, n_pairs, rng)
        except ValueError as e:
            print(f"  [SKIP] {e}\n")
            return
    else:
        images = scan_flat_folder(ds_dir)
        print(f"  [info] {len(images)} image file(s) found (flat)")
        if len(images) < 2:
            print("  [SKIP] Need at least 2 images\n")
            return
        pairs = generate_pairs_convention_b(images, ds_dir, n_pairs, rng)

    # Verify every file really exists
    pairs = verify_and_filter(pairs, ds_dir)

    if len(pairs) < 10:
        print(f"  [SKIP] Only {len(pairs)} valid pair(s) after verification "
              f"– need at least 10\n")
        return

    write_ann(ann_file, pairs)
    same_count = sum(1 for _, _, l in pairs if l == 1)
    diff_count = len(pairs) - same_count
    print(f"  [done] Wrote {len(pairs)} pairs "
          f"({same_count} same / {diff_count} different) → {ann_file}\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate annotation .txt files for synthetic FR datasets."
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Regenerate annotation files even if they already exist."
    )
    parser.add_argument(
        "--n-pairs", type=int, default=N_PAIRS,
        help=f"Number of pairs per dataset (default: {N_PAIRS}, must be divisible by 10)."
    )
    args = parser.parse_args()

    if args.n_pairs % 10 != 0:
        parser.error("--n-pairs must be divisible by 10 for a clean 10-fold split.")

    if not SYNTH_ROOT.exists():
        print(f"[ERROR] Synthetic data root not found: {SYNTH_ROOT}")
        return

    datasets = sorted(d for d in SYNTH_ROOT.iterdir() if d.is_dir())
    if not datasets:
        print(f"[ERROR] No sub-folders found in {SYNTH_ROOT}")
        return

    print(f"Found {len(datasets)} dataset(s): {[d.name for d in datasets]}\n")

    rng = random.Random(SEED)

    for ds_dir in datasets:
        ann_file = SYNTH_ROOT / f"{ds_dir.name}_ann.txt"
        generate_for_dataset(ds_dir, ann_file, args.n_pairs, rng, args.force)

    print("All annotation files are up to date.")


if __name__ == "__main__":
    main()
