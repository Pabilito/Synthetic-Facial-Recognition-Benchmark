"""
compute_demographics_summary.py
--------------------------------
Summarises age and gender per dataset, restricted to the subset of
images actually used for evaluation (listed in ann.txt pair files).

Demographics source : demographics.csv
                      columns: dataset, image_path, identity, age, gender, gender_confidence

Annotation files    : data/real/<DatasetName>_ann.txt
                      data/synth/<DatasetName>_ann.txt
                      Each line: <label> <image_path_1> <image_path_2>
                      Unique evaluation images = union of all path1 and path2 values.

The dataset name in demographics.csv (e.g. "vec2face") is matched to the
ann.txt filename stem (e.g. "Vec2Face_ann.txt") case-insensitively.

Usage
-----
    python compute_demographics_summary.py [--demo PATH] [--anndir PATH] [--output PATH]

Defaults (relative to script directory):
    --demo    demographics.csv
    --anndir  .          (searches recursively for *_ann.txt files)
    --output  demographics_summary.csv
"""

import argparse
import os
import glob
import pandas as pd
import numpy as np


# ── Helpers ───────────────────────────────────────────────────────────────────

def find_ann_files(ann_root: str) -> dict:
    """
    Recursively find all *_ann.txt files under ann_root.
    Returns {dataset_stem_lower: filepath}
    e.g. {"vec2face": "/path/data/synth/Vec2Face_ann.txt"}
    """
    pattern = os.path.join(ann_root, "**", "*_ann.txt")
    found = {}
    for path in glob.glob(pattern, recursive=True):
        stem = os.path.basename(path).replace("_ann.txt", "").lower()
        found[stem] = path
    return found


def load_eval_paths(ann_file: str) -> set:
    """
    Load unique image paths from a pair-format ann.txt.
    Each line: <label> <image_path_1> <image_path_2>
    Returns the union of all path1 and path2 values.
    """
    paths = set()
    with open(ann_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3:
                paths.add(parts[1])
                paths.add(parts[2])
    return paths


def age_stats(series: pd.Series) -> dict:
    valid = series.dropna()
    total = len(series)
    if valid.empty:
        return {
            "age_mean": None, "age_std": None, "age_median": None,
            "age_min": None,  "age_max": None,
            "age_missing_pct": 100.0,
        }
    return {
        "age_mean":        round(valid.mean(),   2),
        "age_std":         round(valid.std(),    2),
        "age_median":      round(valid.median(), 1),
        "age_min":         round(valid.min(),    1),
        "age_max":         round(valid.max(),    1),
        "age_missing_pct": round(100 * (total - len(valid)) / total, 1),
    }


def gender_stats(series: pd.Series) -> dict:
    total = len(series)
    counts = series.value_counts(dropna=False)
    result = {}
    for label, count in counts.items():
        safe = str(label).lower().replace(" ", "_")
        result[f"n_{safe}"]   = int(count)
        result[f"pct_{safe}"] = round(100 * count / total, 1)
    return result


# ── Main ──────────────────────────────────────────────────────────────────────

def main(demo_path: str, ann_root: str, output_path: str):
    demo = pd.read_csv(demo_path)
    print(f"Loaded demographics : {len(demo):,} rows, "
          f"{demo['dataset'].nunique()} datasets")

    ann_files = find_ann_files(ann_root)
    print(f"Found {len(ann_files)} ann.txt files: {sorted(ann_files.keys())}\n")

    records = []

    for ds_name, group in demo.groupby("dataset"):
        ds_key    = ds_name.lower()
        ann_path  = ann_files.get(ds_key)

        if ann_path is None:
            print(f"  ⚠  No ann.txt for '{ds_name}' — skipping")
            continue

        eval_paths = load_eval_paths(ann_path)

        # Match on image_path (exact), fallback to basename only
        mask = group["image_path"].isin(eval_paths)
        if mask.sum() == 0:
            mask = group["image_path"].apply(os.path.basename).isin(
                {os.path.basename(p) for p in eval_paths}
            )

        subset = group[mask]
        print(f"  {ds_name:20s}: {len(eval_paths):6,} ann paths → "
              f"{len(subset):6,} matched / {len(group):6,} total")

        row = {
            "dataset":       ds_name,
            "n_images":      len(subset),
            "ann_file":      os.path.basename(ann_path),
        }
        row.update(age_stats(subset["age"]))
        row.update(gender_stats(subset["gender"]))
        records.append(row)

    if not records:
        print("\nNo datasets matched — check that --anndir points to the folder "
              "containing *_ann.txt files.")
        return

    summary = (pd.DataFrame(records)
               .sort_values("dataset")
               .reset_index(drop=True))
    summary.to_csv(output_path, index=False)
    print(f"\nSaved summary ({len(summary)} datasets) to '{output_path}'")

    print("\n" + "=" * 90)
    print("Demographics Summary — evaluation subset only")
    print("=" * 90)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--demo",
        default=os.path.join(script_dir, "demographics.csv"),
        help="Path to demographics.csv",
    )
    parser.add_argument(
        "--anndir",
        default=os.path.join(script_dir, ".."),
        help="Root dir to search recursively for *_ann.txt files",
    )
    parser.add_argument(
        "--output",
        default=os.path.join(script_dir, "demographics_summary.csv"),
        help="Output CSV path",
    )
    args = parser.parse_args()
    main(args.demo, args.anndir, args.output)
