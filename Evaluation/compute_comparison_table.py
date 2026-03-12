"""
compute_comparison_table.py
---------------------------
Produces a correlation table for each metric with rows:
  - Every real vs real dataset pair (excluding self-pairs)
  - MorphFace vs each real dataset
  - Vec2Face  vs each real dataset

Metrics: EER, FMR100, FMR1000, ZeroFMR, TAR@FAR1, TAR@FAR01, accuracy

Sources:
  - EER/FMR*/ZeroFMR/TAR* : biometric_metrics.csv
  - accuracy               : real_summary.csv + synth_summary.csv

Columns per pair: r, 95% CI (Fisher), p_r, rho, 95% CI (Fisher), p_rho

Usage
-----
    python compute_comparison_table.py [--metrics PATH] [--real PATH]
                                       [--synth PATH]  [--output PATH]

Defaults (same directory as script):
    --metrics  biometric_metrics.csv
    --real     real_summary.csv
    --synth    synth_summary.csv
    --output   comparison_table.csv
"""

import argparse
import os
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

REAL_DATASETS = ["AgeDB-30", "CALFW", "CFP-FP", "CPLFW", "IJBB", "IJBC", "LFW"]
FOCUS_SYNTH   = ["MorphFace", "Vec2Face"]
BIOMETRIC_METRICS = ["EER", "FMR100", "FMR1000", "ZeroFMR", "TAR@FAR1", "TAR@FAR01"]
ALL_METRICS   = BIOMETRIC_METRICS + ["accuracy"]


def fisher_ci(corr, n, alpha=0.05):
    """95% CI for a correlation via Fisher z-transformation."""
    z     = np.arctanh(np.clip(corr, -0.9999, 0.9999))
    se    = 1.0 / np.sqrt(n - 3)
    z_crit = 1.96
    return np.tanh(z - z_crit * se), np.tanh(z + z_crit * se)


def compute_row(x_vals, y_vals, label_x, label_y, metric):
    combined = pd.concat([x_vals, y_vals], axis=1, keys=["x", "y"]).dropna()
    n = len(combined)
    if n < 3:
        return None  # not enough data (e.g. IJBB/IJBC accuracy)

    r,   p_r   = pearsonr( combined["x"], combined["y"])
    rho, p_rho = spearmanr(combined["x"], combined["y"])

    r_lo,   r_hi   = fisher_ci(r,   n)
    rho_lo, rho_hi = fisher_ci(rho, n)

    return {
        "metric":    metric,
        "dataset_x": label_x,
        "dataset_y": label_y,
        "n_models":  n,
        "r":         round(r,      4),
        "r_ci_lo":   round(r_lo,   4),
        "r_ci_hi":   round(r_hi,   4),
        "p_r":       round(p_r,    4),
        "rho":       round(rho,    4),
        "rho_ci_lo": round(rho_lo, 4),
        "rho_ci_hi": round(rho_hi, 4),
        "p_rho":     round(p_rho,  4),
    }


def main(metrics_path: str, real_path: str, synth_path: str, output_path: str):
    # ── Load data ─────────────────────────────────────────────────────────────
    bm   = pd.read_csv(metrics_path)
    real = pd.read_csv(real_path)
    synth = pd.read_csv(synth_path)

    print(f"Loaded biometric metrics : {len(bm):,} rows")
    print(f"Loaded real accuracy     : {len(real):,} rows")
    print(f"Loaded synth accuracy    : {len(synth):,} rows")

    # Build score lookup: scores[dataset][metric] -> Series indexed by model
    scores = {}

    # Biometric metrics (from biometric_metrics.csv)
    for ds in bm["dataset"].unique():
        scores[ds] = bm[bm["dataset"] == ds].set_index("model")[BIOMETRIC_METRICS]

    # Accuracy (from summary CSVs) — add as extra column
    for ds in real["dataset"].unique():
        acc = real[real["dataset"] == ds].set_index("model")["accuracy"]
        if ds in scores:
            scores[ds]["accuracy"] = acc
        else:
            scores[ds] = acc.to_frame()

    for ds in synth["dataset"].unique():
        acc = synth[synth["dataset"] == ds].set_index("model")["accuracy"]
        if ds in scores:
            scores[ds]["accuracy"] = acc
        else:
            scores[ds] = acc.to_frame()

    # ── Compute correlations ──────────────────────────────────────────────────
    records = []

    for metric in ALL_METRICS:
        # Real vs Real (upper triangle, no self-pairs)
        for i, dx in enumerate(REAL_DATASETS):
            for dy in REAL_DATASETS[i+1:]:
                if metric not in scores[dx].columns or metric not in scores[dy].columns:
                    continue
                row = compute_row(scores[dx][metric], scores[dy][metric], dx, dy, metric)
                if row:
                    records.append(row)

        # MorphFace / Vec2Face vs each real dataset
        for synth_ds in FOCUS_SYNTH:
            for real_ds in REAL_DATASETS:
                if metric not in scores[synth_ds].columns or metric not in scores[real_ds].columns:
                    continue
                row = compute_row(scores[synth_ds][metric], scores[real_ds][metric],
                                  synth_ds, real_ds, metric)
                if row:
                    records.append(row)

    # ── Save & print ──────────────────────────────────────────────────────────
    out = pd.DataFrame(records)
    out.to_csv(output_path, index=False)
    print(f"\nSaved {len(out):,} rows to '{output_path}'")

    for metric in ALL_METRICS:
        sub = out[out["metric"] == metric].drop(columns="metric")
        if sub.empty:
            continue
        print()
        print("=" * 90)
        print(f"Metric: {metric}")
        print("=" * 90)
        print(sub.to_string(index=False))


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--metrics", default=os.path.join(script_dir, "biometric_metrics.csv"))
    parser.add_argument("--real",    default=os.path.join(script_dir, "real_summary.csv"))
    parser.add_argument("--synth",   default=os.path.join(script_dir, "synth_summary.csv"))
    parser.add_argument("--output",  default=os.path.join(script_dir, "comparison_table.csv"))
    args = parser.parse_args()
    main(args.metrics, args.real, args.synth, args.output)
