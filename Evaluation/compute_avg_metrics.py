"""
compute_avg_metrics.py
----------------------
Reads the recomputed correlation CSV, averages r, p_r, rho, and p_rho
per (metric, synthetic_dataset) across the 7 real datasets, and
prints four pivot tables.

Usage
-----
    python compute_avg_metrics.py [--input PATH]

Default:
    --input  biometric_corr_all_metrics_recomputed.csv  (same directory as script)
"""

import argparse
import os
import pandas as pd

REAL_DATASETS  = {"AgeDB-30", "CALFW", "CFP-FP", "CPLFW", "IJBB", "IJBC", "LFW"}
SYNTH_DATASETS = {"CemiFace", "Control10k", "DCFace", "Digi2Real", "HyperFace",
                  "IDiffFace", "Langevin-DisCo", "MorphFace", "SFace2",
                  "SynFace", "SynMulti-PIE", "Vec2Face"}
METRIC_ORDER   = ["EER", "FMR100", "FMR1000", "TAR@FAR01", "TAR@FAR1", "ZeroFMR"]


def main(input_path: str):
    df = pd.read_csv(input_path)

    # Keep only real-vs-synthetic pairs (dataset_x = real, dataset_y = synth)
    mask = df["dataset_x"].isin(REAL_DATASETS) & df["dataset_y"].isin(SYNTH_DATASETS)
    pairs = df[mask].copy()
    pairs = pairs.rename(columns={"dataset_y": "synthetic_dataset"})

    # Average all four correlation columns per (metric, synthetic_dataset)
    agg = (
        pairs
        .groupby(["metric", "synthetic_dataset"])[["r", "p_r", "rho", "p_rho"]]
        .mean()
        .reset_index()
    )

    labels = [
        ("r",     "Average Pearson r"),
        ("p_r",   "Average Pearson p-value"),
        ("rho",   "Average Spearman rho"),
        ("p_rho", "Average Spearman p-value"),
    ]

    for value_col, label in labels:
        pivot = agg.pivot(index="synthetic_dataset", columns="metric", values=value_col)
        cols  = [m for m in METRIC_ORDER if m in pivot.columns]
        pivot = pivot[cols].sort_index()

        print("=" * 72)
        print(f"{label} per Metric x Synthetic Dataset")
        print("(averaged over the 7 real datasets)")
        print("=" * 72)
        print(pivot.round(4).to_string())
        print()


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--input",
        default=os.path.join(script_dir, "biometric_corr_all_metrics_recomputed.csv"),
    )
    args = parser.parse_args()
    main(args.input)
