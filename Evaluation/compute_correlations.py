"""
compute_correlations.py
-----------------------
Computes Pearson r, Spearman rho, and their p-values for each
(metric, real_dataset, synthetic_dataset) pair directly from the raw
biometric_metrics.csv, then saves the full correlation table to
biometric_corr_all_metrics_recomputed.csv.

Usage
-----
    python compute_correlations.py [--input PATH] [--output PATH]

Defaults:
    --input   biometric_metrics.csv              (same directory as script)
    --output  biometric_corr_all_metrics_recomputed.csv
"""

import argparse
import os
import pandas as pd
from scipy.stats import pearsonr, spearmanr

REAL_DATASETS  = {"AgeDB-30", "CALFW", "CFP-FP", "CPLFW", "IJBB", "IJBC", "LFW"}
SYNTH_DATASETS = {"CemiFace", "Control10k", "DCFace", "Digi2Real", "HyperFace",
                  "IDiffFace", "Langevin-DisCo", "MorphFace", "SFace2",
                  "SynFace", "SynMulti-PIE", "Vec2Face"}
METRICS = ["EER", "FMR100", "FMR1000", "ZeroFMR", "TAR@FAR1", "TAR@FAR01"]


def main(input_path: str, output_path: str):
    df = pd.read_csv(input_path)
    print(f"Loaded {len(df):,} rows from '{input_path}'")

    # Build per-dataset score tables: {dataset -> DataFrame indexed by model}
    scores = {
        ds: df[df["dataset"] == ds].set_index("model")[METRICS]
        for ds in df["dataset"].unique()
    }

    records = []
    for metric in METRICS:
        for real_ds in sorted(REAL_DATASETS):
            for synth_ds in sorted(SYNTH_DATASETS):
                real_scores  = scores[real_ds][metric]
                synth_scores = scores[synth_ds][metric]

                # Align on shared models
                combined = pd.concat([real_scores, synth_scores], axis=1, keys=["real", "synth"]).dropna()
                n = len(combined)
                if n < 3:
                    print(f"  ⚠  Skipping {metric} | {real_ds} vs {synth_ds}: only {n} shared models")
                    continue

                r,   p_r   = pearsonr( combined["real"], combined["synth"])
                rho, p_rho = spearmanr(combined["real"], combined["synth"])

                records.append({
                    "metric":    metric,
                    "dataset_x": real_ds,
                    "dataset_y": synth_ds,
                    "n_models":  n,
                    "r":         r,
                    "p_r":       p_r,
                    "rho":       rho,
                    "p_rho":     p_rho,
                })

    out = pd.DataFrame(records)
    out.to_csv(output_path, index=False, float_format="%.6f")
    print(f"Saved {len(out):,} correlation rows to '{output_path}'")
    return out


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input",  default=os.path.join(script_dir, "biometric_metrics.csv"))
    parser.add_argument("--output", default=os.path.join(script_dir, "biometric_corr_all_metrics_recomputed.csv"))
    args = parser.parse_args()
    main(args.input, args.output)
