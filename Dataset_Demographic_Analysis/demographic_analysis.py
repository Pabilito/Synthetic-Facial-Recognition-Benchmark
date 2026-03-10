"""
demographic_analysis.py
─────────────────────────────────────────────────────────────────────────────
Runs age and gender estimation on every unique image referenced by any
*_ann.txt annotation file found under data/real/ and data/synth/.

Images are de-duplicated per dataset — each file is analysed once even if
it appears in multiple pairs.

Model
─────
  DeepFace with detector_backend="skip" (images are pre-cropped face chips).
  Age: apparent age regressor  (~±4.65 MAE)
  Gender: classifier            (~97.4 % accuracy)

Output  (Evaluation/demographics.csv)
──────────────────────────────────────
  One row per image, all datasets in a single file.
  Columns:
    dataset       – name of the dataset (folder / ann file stem)
    image_path    – relative path as it appears in the ann.txt
    identity      – sub-folder name for Convention A datasets,
                    empty string for flat (LFW-style) datasets
    age           – estimated apparent age (float)
    gender        – "Man" or "Woman"
    gender_confidence – confidence of the gender prediction (0–100)

Usage
─────
  python demographic_analysis.py                   # scan both data/real and data/synth
  python demographic_analysis.py --real-only
  python demographic_analysis.py --synth-only
  python demographic_analysis.py --resume          # skip images already in the CSV

  # Run only selected datasets (mix of real and synthetic, by name):
  python demographic_analysis.py --datasets lfw cemiFace

  # Selected datasets + resume:
  python demographic_analysis.py --datasets lfw cemiFace --resume

  --datasets resolves each name against both data/real/ and data/synth/
  automatically, so there is no need to also pass --real-only / --synth-only.
  Names are matched case-insensitively against the ann.txt stem
  (e.g. "lfw" matches data/real/lfw_ann.txt, "cemiFace" matches
  data/synth/cemiface_ann.txt).
"""

import argparse
import csv
import os
import warnings
from pathlib import Path

from tqdm import tqdm

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

from deepface import DeepFace

# ── Paths ─────────────────────────────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).parent.parent          # project root
REAL_ROOT  = SCRIPT_DIR / "data" / "real"
SYNTH_ROOT = SCRIPT_DIR / "data" / "synth"
EVAL_DIR   = SCRIPT_DIR / "Evaluation"
OUT_CSV    = EVAL_DIR / "demographics.csv"

CSV_COLS   = ["dataset", "image_path", "identity",
              "age", "gender", "gender_confidence"]

# ── Annotation discovery ──────────────────────────────────────────────────────

def find_ann_files(root: Path) -> list[tuple[str, Path, Path]]:
    """
    Scan root for *_ann.txt files.
    Returns list of (dataset_name, ann_path, dataset_img_root).

    For data/real/ the ann file and images share the same folder.
    For data/synth/ each dataset lives in its own sub-folder.
    """
    if not root.exists():
        return []

    entries = []
    for ann_file in sorted(root.glob("*_ann.txt")):
        ds_name = ann_file.stem.replace("_ann", "")

        # synth: images are in root/<ds_name>/
        # real:  images are directly in root/
        img_root = root / ds_name if (root / ds_name).is_dir() else root
        entries.append((ds_name, ann_file, img_root))

    return entries


def unique_images_from_ann(ann_path: Path) -> list[str]:
    """Return deduplicated list of image paths from an ann.txt file."""
    paths = set()
    with open(ann_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 3:
                _, p1, p2 = parts
                paths.add(p1)
                paths.add(p2)
    return sorted(paths)


def identity_from_path(image_path: str) -> str:
    """
    Extract identity for Convention A (sub-folder) datasets.
    Returns the first path component if there are multiple components,
    empty string for flat (single-level) paths like 'lfw_112x112/0.bmp'.
    """
    parts = Path(image_path).parts
    # Convention A: identity/image.jpg  → parts = ('identity', 'image.jpg')
    # Convention B: folder/0.bmp        → parts = ('folder', '0.bmp')
    #               or just  0.bmp      → parts = ('0.bmp',)
    # Heuristic: if there are exactly 2 parts and the first looks like a
    # numeric index or a dataset-folder name it's Convention B.
    if len(parts) >= 2:
        candidate = parts[0]
        # If the top-level folder name ends with a resolution hint (e.g.
        # "lfw_112x112") it is a dataset sub-folder, not an identity.
        if any(candidate.endswith(s) for s in
               ("112x112", "112", "160", "224", "_crop", "_aligned")):
            return ""
        # If it looks purely numeric it's also not an identity label
        if candidate.isdigit():
            return ""
        return candidate
    return ""


# ── DeepFace wrapper ──────────────────────────────────────────────────────────

def analyse_image(img_path: Path) -> dict | None:
    """
    Returns {"age": float, "gender": str, "gender_confidence": float}
    or None if DeepFace cannot process the image.
    """
    try:
        results = DeepFace.analyze(
            img_path=str(img_path),
            actions=["age", "gender"],
            detector_backend="skip",   # images are pre-cropped face chips
            enforce_detection=False,
            silent=True,
        )
        r = results[0] if isinstance(results, list) else results
        return {
            "age":               round(float(r["age"]), 2),
            "gender":            r["dominant_gender"],
            "gender_confidence": round(float(max(r["gender"].values())), 2),
        }
    except Exception:
        return None


# ── CSV helpers ───────────────────────────────────────────────────────────────

def load_already_done(csv_path: Path) -> set[tuple[str, str]]:
    """Return set of (dataset, image_path) pairs already in the CSV."""
    done = set()
    if csv_path.exists():
        with open(csv_path, newline="") as f:
            for row in csv.DictReader(f):
                done.add((row["dataset"], row["image_path"]))
    return done


def open_csv_writer(csv_path: Path):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists()
    f = open(csv_path, "a", newline="")
    writer = csv.DictWriter(f, fieldnames=CSV_COLS)
    if write_header:
        writer.writeheader()
    return f, writer


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Age & gender estimation on all ann.txt-referenced images."
    )
    parser.add_argument("--real-only",  action="store_true")
    parser.add_argument("--synth-only", action="store_true")
    parser.add_argument(
        "--datasets",
        nargs="+",
        metavar="NAME",
        help=(
            "Process only the named dataset(s). Names are matched "
            "case-insensitively against the ann.txt stem and are searched "
            "in both data/real/ and data/synth/ (overrides --real-only / "
            "--synth-only)."
        ),
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Skip images already present in demographics.csv."
    )
    args = parser.parse_args()

    # When --datasets is given, search both roots regardless of --real/synth-only
    if args.datasets:
        roots = [REAL_ROOT, SYNTH_ROOT]
    else:
        roots = []
        if not args.synth_only:
            roots.append(REAL_ROOT)
        if not args.real_only:
            roots.append(SYNTH_ROOT)

    # Collect all (dataset_name, ann_path, img_root) entries
    all_entries: list[tuple[str, Path, Path]] = []
    for root in roots:
        all_entries.extend(find_ann_files(root))

    # Filter to requested datasets when --datasets is supplied
    if args.datasets:
        requested = {name.lower() for name in args.datasets}
        all_entries = [
            e for e in all_entries if e[0].lower() in requested
        ]
        found = {e[0].lower() for e in all_entries}
        missing = requested - found
        if missing:
            print(
                f"[WARN] The following dataset(s) were not found in "
                f"{REAL_ROOT} or {SYNTH_ROOT}: "
                + ", ".join(sorted(missing))
            )

    if not all_entries:
        print("[ERROR] No *_ann.txt files found.")
        return

    print(f"Found {len(all_entries)} dataset(s):")
    for ds_name, ann_path, img_root in all_entries:
        print(f"  {ds_name:<30}  ann: {ann_path.name}  imgs: {img_root}")

    already_done = load_already_done(OUT_CSV) if args.resume else set()
    if args.resume and already_done:
        print(f"\nResuming — {len(already_done)} images already in CSV, skipping.")

    f, writer = open_csv_writer(OUT_CSV)

    total_written = 0
    total_failed  = 0

    try:
        for ds_name, ann_path, img_root in all_entries:
            images = unique_images_from_ann(ann_path)
            print(f"\n━━━ {ds_name} ({len(images)} unique images) ━━━")

            failed = 0
            for rel_path in tqdm(images, desc=f"  {ds_name}", unit="img"):
                if args.resume and (ds_name, rel_path) in already_done:
                    continue

                result = analyse_image(img_root / rel_path)
                if result is None:
                    failed += 1
                    continue

                writer.writerow({
                    "dataset":            ds_name,
                    "image_path":         rel_path,
                    "identity":           identity_from_path(rel_path),
                    "age":                result["age"],
                    "gender":             result["gender"],
                    "gender_confidence":  result["gender_confidence"],
                })
                total_written += 1

            total_failed += failed
            if failed:
                print(f"  [warn] {failed} image(s) failed / no face detected")

    finally:
        f.close()

    print(f"\n{'─'*52}")
    print(f"Done. {total_written} rows written → {OUT_CSV}")
    if total_failed:
        print(f"      {total_failed} image(s) skipped (no face / error)")


if __name__ == "__main__":
    main()
