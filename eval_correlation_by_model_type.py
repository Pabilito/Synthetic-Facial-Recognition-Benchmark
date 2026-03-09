"""
eval_correlation_by_model_type.py
─────────────────────────────────────────────────────────────────────────────
Runs the correlation analysis twice:

1. Only REAL models
2. Only SYNTHETIC models

Synthetic models:
    IDiffFace_CF50
    SFace_IResNet50
    hsface10k
    hsface300k

Outputs (per type):
  correlation_all_datasets_<type>.png
  correlation_heatmap_<type>.png
  dataset_correlations_<type>.csv
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from scipy import stats


# ── Configuration ────────────────────────────────────────────────────────────

SYNTH_MODELS = {
    "IDiffFace_CF50",
    "SFace_IResNet50",
    "hsface10k",
    "hsface300k",
}


# ── Paths ────────────────────────────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).parent
EVAL_DIR = SCRIPT_DIR / "Evaluation"

REAL_CSV = EVAL_DIR / "real_summary.csv"
SYNTH_CSV = EVAL_DIR / "synth_summary.csv"

CMAP = plt.get_cmap("tab20")


# ── Load helper ──────────────────────────────────────────────────────────────

def load(path: Path, label: str):

    if not path.exists():
        print(f"[ERROR] {label} not found: {path}")
        sys.exit(1)

    df = pd.read_csv(path)

    df["accuracy"] = pd.to_numeric(df["accuracy"], errors="coerce")
    df["std"] = pd.to_numeric(df["std"], errors="coerce")

    return df


# ── Regression helper ────────────────────────────────────────────────────────

def regression_line(x, y):

    mask = np.isfinite(x) & np.isfinite(y)

    if mask.sum() < 3:
        return None

    slope, intercept, r, p, _ = stats.linregress(x[mask], y[mask])

    x_line = np.linspace(x[mask].min(), x[mask].max(), 100)
    y_line = slope * x_line + intercept

    return x_line, y_line, r, p


# ── Scatter plot panel ───────────────────────────────────────────────────────

def plot_panel(ax, merged, ds_x, ds_y, model_colors):

    x = merged["accuracy_x"].values
    y = merged["accuracy_y"].values

    for _, row in merged.iterrows():

        color = model_colors[row["model"]]

        ax.errorbar(
            row["accuracy_x"],
            row["accuracy_y"],
            xerr=row["std_x"],
            yerr=row["std_y"],
            fmt="o",
            color=color,
            ecolor=color,
            elinewidth=0.8,
            capsize=3,
            markersize=6,
        )

    res = regression_line(x, y)

    r_val = np.nan
    p_val = np.nan

    if res:

        x_line, y_line, r_val, p_val = res

        ax.plot(
            x_line,
            y_line,
            linestyle="--",
            linewidth=1.2,
            color="black",
            alpha=0.6,
        )

        p_str = f"p={p_val:.3f}" if p_val >= 0.001 else "p<0.001"

        ax.annotate(
            f"r={r_val:.3f}  {p_str}",
            xy=(0.05, 0.93),
            xycoords="axes fraction",
            fontsize=8,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7),
        )

    ax.set_xlabel(f"{ds_x} accuracy (%)", fontsize=8)
    ax.set_ylabel(f"{ds_y} accuracy (%)", fontsize=8)

    ax.set_title(f"{ds_x} ↔ {ds_y}", fontsize=9, fontweight="bold")

    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))

    ax.tick_params(labelsize=7)
    ax.grid(True, linestyle=":", alpha=0.4)

    return r_val, p_val


# ── Heatmap ──────────────────────────────────────────────────────────────────

def plot_heatmap(results_df, datasets, split_index, suffix):

    r_matrix = (
        results_df
        .pivot(index="dataset_x", columns="dataset_y", values="r")
        .reindex(index=datasets, columns=datasets)
    )

    fig, ax = plt.subplots(figsize=(8, 7))

    im = ax.imshow(r_matrix, vmin=-1, vmax=1)

    ax.set_xticks(range(len(datasets)))
    ax.set_yticks(range(len(datasets)))

    ax.set_xticklabels(datasets, rotation=45, ha="right")
    ax.set_yticklabels(datasets)

    ax.set_title(f"Dataset Correlation Heatmap ({suffix})")

    for i in range(len(datasets)):
        for j in range(len(datasets)):

            val = r_matrix.iloc[i, j]

            if np.isfinite(val):

                ax.text(
                    j,
                    i,
                    f"{val:.2f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                )

    ax.axhline(split_index - 0.5, color="black", linewidth=1.5)
    ax.axvline(split_index - 0.5, color="black", linewidth=1.5)

    fig.colorbar(im, ax=ax, label="Pearson r")

    fig.tight_layout()

    out = EVAL_DIR / f"correlation_heatmap_{suffix}.png"
    fig.savefig(out, dpi=500)

    print("Saved →", out)


# ── Core evaluation ──────────────────────────────────────────────────────────

def run_evaluation(all_df, real_datasets, synth_datasets, suffix):

    datasets = real_datasets + synth_datasets

    models = sorted(all_df["model"].unique())

    model_colors = {
        m: CMAP(i / max(len(models) - 1, 1))
        for i, m in enumerate(models)
    }

    n = len(datasets)

    fig, axes = plt.subplots(n, n, figsize=(4.5 * n, 4 * n), squeeze=False)

    fig.suptitle(f"Dataset Correlations ({suffix})", fontsize=13, fontweight="bold")

    results = []

    for r_idx, ds_x in enumerate(datasets):

        x_sub = all_df[all_df["dataset"] == ds_x][
            ["model", "accuracy", "std"]
        ].rename(columns={"accuracy": "accuracy_x", "std": "std_x"})

        for c_idx, ds_y in enumerate(datasets):

            y_sub = all_df[all_df["dataset"] == ds_y][
                ["model", "accuracy", "std"]
            ].rename(columns={"accuracy": "accuracy_y", "std": "std_y"})

            merged = x_sub.merge(y_sub, on="model", how="inner")

            ax = axes[r_idx][c_idx]

            if merged.empty:

                r_val = np.nan
                p_val = np.nan

                ax.text(0.5, 0.5, "No common models",
                        ha="center", va="center", transform=ax.transAxes)

            else:

                r_val, p_val = plot_panel(ax, merged, ds_x, ds_y, model_colors)

            results.append({
                "dataset_x": ds_x,
                "dataset_y": ds_y,
                "r": r_val,
                "p": p_val,
                "n_models": len(merged)
            })

    fig.tight_layout()

    scatter_out = EVAL_DIR / f"correlation_all_datasets_{suffix}.png"
    fig.savefig(scatter_out, dpi=500, bbox_inches="tight")

    print("Saved →", scatter_out)

    results_df = pd.DataFrame(results)

    csv_out = EVAL_DIR / f"dataset_correlations_{suffix}.csv"
    results_df.to_csv(csv_out, index=False)

    print("Saved →", csv_out)

    plot_heatmap(
        results_df,
        datasets,
        split_index=len(real_datasets),
        suffix=suffix
    )


# ── Main ─────────────────────────────────────────────────────────────────────

def main():

    real_df = load(REAL_CSV, "real_summary.csv")
    synth_df = load(SYNTH_CSV, "synth_summary.csv")

    real_datasets = sorted(real_df["dataset"].unique())
    synth_datasets = sorted(synth_df["dataset"].unique())

    all_df = pd.concat([real_df, synth_df], ignore_index=True)

    # real models
    real_models_df = all_df[~all_df["model"].isin(SYNTH_MODELS)]

    # synthetic models
    synth_models_df = all_df[all_df["model"].isin(SYNTH_MODELS)]

    print("Running REAL model evaluation")
    run_evaluation(real_models_df, real_datasets, synth_datasets, "real")

    print("\nRunning SYNTHETIC model evaluation")
    run_evaluation(synth_models_df, real_datasets, synth_datasets, "synth")


if __name__ == "__main__":
    main()