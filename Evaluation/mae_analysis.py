"""
MAE Analysis: MorphFace & Vec2Face vs Real Datasets
=====================================================
For each model and metric:
  - accuracy (from synth_summary / real_summary): MAE = mean(|synth - real|) across models
  - biometric metrics (EER, FMR100, FMR1000, ZeroFMR, TAR@FAR1, TAR@FAR01): same approach

All MAE values are expressed as ratios (0-1 scale): 1 percentage point = 0.01.

Real benchmarks used:
  - Accuracy: LFW, CALFW, CPLFW, AgeDB-30, CFP-FP  (5 datasets)
  - Biometric: LFW, CALFW, CFP-FP, CPLFW, AgeDB-30, IJBB, IJBC  (7 datasets)
"""

import pandas as pd
import numpy as np
from pathlib import Path

# -- Paths ------------------------------------------------------------------
SYNTH_ACC   = Path("Evaluation/synth_summary.csv")
REAL_ACC    = Path("Evaluation/real_summary.csv")
BIO_METRICS = Path("Evaluation/biometric_metrics.csv")
OUT_DIR     = Path("Evaluation/")
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGETS = ["MorphFace", "Vec2Face"]

# -- Real accuracy benchmarks (5 datasets) ---------------------------------
REAL_ACC_DATASETS = ["LFW", "CALFW", "CPLFW", "AgeDB-30", "CFP-FP"]

# -- Real biometric benchmarks (7 datasets) --------------------------------
REAL_BIO_DATASETS = ["LFW", "CALFW", "CFP-FP", "CPLFW", "AgeDB-30", "IJBB", "IJBC"]
BIO_METRIC_COLS   = ["EER", "FMR100", "FMR1000", "ZeroFMR", "TAR@FAR1", "TAR@FAR01"]

def to_ratio(df):
    """Divide all numeric values by 100 so 1 pp -> 0.01."""
    return df.apply(lambda col: col.map(
        lambda x: round(x / 100, 6) if pd.notna(x) else x
    ))

# ==========================================================================
# 1. ACCURACY MAE
# ==========================================================================
print("=" * 70)
print("PART 1 - ACCURACY MAE  (MorphFace & Vec2Face vs 5 real benchmarks)")
print("=" * 70)

synth_acc = pd.read_csv(SYNTH_ACC)
real_acc  = pd.read_csv(REAL_ACC)

real_acc_f = real_acc[real_acc["dataset"].isin(REAL_ACC_DATASETS)].copy()

acc_results = []

for target in TARGETS:
    synth_target = synth_acc[synth_acc["dataset"] == target][["model", "accuracy"]].rename(
        columns={"accuracy": "synth_acc"}
    )
    for real_ds in REAL_ACC_DATASETS:
        real_ds_df = real_acc_f[real_acc_f["dataset"] == real_ds][["model", "accuracy"]].rename(
            columns={"accuracy": "real_acc"}
        )
        merged = pd.merge(synth_target, real_ds_df, on="model")
        merged["abs_error"] = (merged["synth_acc"] - merged["real_acc"]).abs()
        mae = merged["abs_error"].mean()
        acc_results.append({
            "SynthDataset": target,
            "RealBenchmark": real_ds,
            "MAE (pp)": round(mae, 4),
        })

acc_df = pd.DataFrame(acc_results)

acc_pivot = acc_df.pivot(index="SynthDataset", columns="RealBenchmark", values="MAE (pp)")
acc_pivot.columns.name = None
acc_pivot.index.name = "SynthDataset \\ RealBenchmark"
acc_pivot["Mean MAE"] = acc_pivot[REAL_ACC_DATASETS].mean(axis=1).round(4)

acc_pivot_ratio = to_ratio(acc_pivot)
acc_pivot_ratio.index.name = "SynthDataset \\ RealBenchmark"

print("\nAccuracy MAE Table (ratio scale; 1 pp = 0.01)\n")
print(acc_pivot_ratio.to_string())
print()

# ==========================================================================
# 2. BIOMETRIC METRICS MAE
# ==========================================================================
print("=" * 70)
print("PART 2 - BIOMETRIC METRICS MAE  (MorphFace & Vec2Face vs 7 real benchmarks)")
print("=" * 70)

bio = pd.read_csv(BIO_METRICS)

real_bio  = bio[(bio["split"] == "real")  & (bio["dataset"].isin(REAL_BIO_DATASETS))].copy()
synth_bio = bio[(bio["split"] == "synth") & (bio["dataset"].isin(TARGETS))].copy()

bio_results = {metric: [] for metric in BIO_METRIC_COLS}

for metric in BIO_METRIC_COLS:
    for target in TARGETS:
        synth_t = synth_bio[synth_bio["dataset"] == target][["model", metric]].rename(
            columns={metric: "synth_val"}
        )
        for real_ds in REAL_BIO_DATASETS:
            real_ds_df = real_bio[real_bio["dataset"] == real_ds][["model", metric]].rename(
                columns={metric: "real_val"}
            )
            merged = pd.merge(synth_t, real_ds_df, on="model").dropna(subset=["synth_val", "real_val"])
            if len(merged) == 0:
                continue
            mae = (merged["synth_val"] - merged["real_val"]).abs().mean()
            bio_results[metric].append({
                "SynthDataset": target,
                "RealBenchmark": real_ds,
                "MAE": round(mae, 4),
            })

bio_pivots = {}
for metric in BIO_METRIC_COLS:
    df = pd.DataFrame(bio_results[metric])
    if df.empty:
        continue
    pivot = df.pivot(index="SynthDataset", columns="RealBenchmark", values="MAE")
    pivot.columns.name = None
    pivot.index.name = f"{metric} \\ RealBenchmark"
    ordered_cols = [c for c in REAL_BIO_DATASETS if c in pivot.columns]
    pivot = pivot[ordered_cols]
    pivot["Mean MAE"] = pivot.mean(axis=1).round(4)

    pivot_ratio = to_ratio(pivot)
    pivot_ratio.index.name = f"{metric} \\ RealBenchmark"
    bio_pivots[metric] = pivot_ratio

    print(f"\n{'-'*60}")
    print(f"  {metric} MAE Table (ratio scale; 1 pp = 0.01)\n")
    print(pivot_ratio.to_string())
    print()

# ==========================================================================
# 3. EXPORT — single CSV with all tables stacked
# ==========================================================================

out_path = OUT_DIR / "mae_results.csv"

with open(out_path, "w") as f:

    # -- Accuracy -----------------------------------------------------------
    f.write("# Accuracy MAE (ratio scale; 1 pp = 0.01)\n")
    acc_pivot_ratio.to_csv(f)
    f.write("\n")

    # -- Biometric metrics --------------------------------------------------
    for metric, pivot in bio_pivots.items():
        f.write(f"# {metric} MAE (ratio scale; 1 pp = 0.01)\n")
        pivot.to_csv(f)
        f.write("\n")

    # -- Summary ------------------------------------------------------------
    summary_rows = []
    for target in TARGETS:
        row = {"SynthDataset": target, "Accuracy": acc_pivot_ratio.loc[target, "Mean MAE"]}
        for metric, pivot in bio_pivots.items():
            row[metric] = pivot.loc[target, "Mean MAE"] if target in pivot.index else np.nan
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows).set_index("SynthDataset")
    f.write("# SUMMARY — Mean MAE across all real benchmarks\n")
    summary_df.to_csv(f)

print(f"\n* Saved -> {out_path}")

# -- Also print summary to stdout ------------------------------------------
print("\n" + "=" * 70)
print("SUMMARY - Mean MAE across all real benchmarks (ratio scale; 1 pp = 0.01)")
print("=" * 70)
print("\n", summary_df.to_string())
print("\nDone.")
