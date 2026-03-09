"""
eval_correlation.py
─────────────────────────────────────────────────────────────────────────────
Reads Evaluation/real_summary.csv and Evaluation/synth_summary.csv and:

1. Plots scatter correlations between all dataset pairs
2. Generates a correlation heatmap (real datasets first, synthetic second)
3. Saves correlation statistics (r, p, n_models)

Outputs:
  correlation_all_datasets.png
  correlation_heatmap.png
  dataset_correlations.csv
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from scipy import stats


# ── Paths ─────────────────────────────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).parent
EVAL_DIR = SCRIPT_DIR / "Evaluation"

REAL_CSV = EVAL_DIR / "real_summary.csv"
SYNTH_CSV = EVAL_DIR / "synth_summary.csv"

OUT_SCATTER = EVAL_DIR / "correlation_all_datasets.png"
OUT_HEATMAP = EVAL_DIR / "correlation_heatmap.png"
OUT_STATS = EVAL_DIR / "dataset_correlations.csv"

CMAP = plt.get_cmap("tab20")


# ── Load helper ───────────────────────────────────────────────────────────────

def load(path: Path, label: str):

    if not path.exists():
        print(f"[ERROR] {label} not found: {path}")
        sys.exit(1)

    df = pd.read_csv(path)

    df["accuracy"] = pd.to_numeric(df["accuracy"], errors="coerce")
    df["std"] = pd.to_numeric(df["std"], errors="coerce")

    return df


# ── Regression helper ─────────────────────────────────────────────────────────

def regression_line(x, y):

    mask = np.isfinite(x) & np.isfinite(y)

    if mask.sum() < 3:
        return None

    slope, intercept, r, p, _ = stats.linregress(x[mask], y[mask])

    x_line = np.linspace(x[mask].min(), x[mask].max(), 100)
    y_line = slope * x_line + intercept

    return x_line, y_line, r, p


# ── Scatter plot panel ─────────────────────────────────────────────────────────

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


# ── Heatmap ───────────────────────────────────────────────────────────────────

def plot_heatmap(results_df, datasets, split_index):

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

    ax.set_title("Dataset Correlation Heatmap (Pearson r)", fontsize=12)

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

    # separator between real and synthetic datasets
    ax.axhline(split_index - 0.5, color="black", linewidth=1.5)
    ax.axvline(split_index - 0.5, color="black", linewidth=1.5)

    fig.colorbar(im, ax=ax, label="Pearson r")

    fig.tight_layout()

    fig.savefig(OUT_HEATMAP, dpi=500)

    print(f"Saved heatmap → {OUT_HEATMAP}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():

    real_df = load(REAL_CSV, "real_summary.csv")
    synth_df = load(SYNTH_CSV, "synth_summary.csv")

    real_df["source"] = "real"
    synth_df["source"] = "synthetic"

    all_df = pd.concat([real_df, synth_df], ignore_index=True)

    real_datasets = sorted(real_df["dataset"].unique())
    synth_datasets = sorted(synth_df["dataset"].unique())

    datasets = real_datasets + synth_datasets

    models = sorted(all_df["model"].unique())

    print("Real datasets:", real_datasets)
    print("Synthetic datasets:", synth_datasets)
    print("Models:", models)

    model_colors = {
        m: CMAP(i / max(len(models) - 1, 1))
        for i, m in enumerate(models)
    }

    n = len(datasets)

    fig, axes = plt.subplots(
        n,
        n,
        figsize=(4.5 * n, 4 * n),
        squeeze=False,
    )

    fig.suptitle(
        "Model Accuracy Correlation Across All Datasets",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )

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

                ax.text(
                    0.5,
                    0.5,
                    "No common models",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    fontsize=9,
                    color="grey",
                )

                r_val = np.nan
                p_val = np.nan

            else:

                r_val, p_val = plot_panel(
                    ax,
                    merged,
                    ds_x,
                    ds_y,
                    model_colors
                )

            results.append({
                "dataset_x": ds_x,
                "dataset_y": ds_y,
                "r": r_val,
                "p": p_val,
                "n_models": len(merged)
            })

    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=model_colors[m],
            markersize=7,
            label=m,
        )
        for m in models
    ]

    fig.legend(
        handles=handles,
        title="Model",
        loc="lower center",
        ncol=min(len(models), 4),
        fontsize=8,
        title_fontsize=9,
        bbox_to_anchor=(0.5, -0.03),
    )

    fig.tight_layout()

    fig.savefig(OUT_SCATTER, dpi=500, bbox_inches="tight")

    print(f"Saved scatter plots → {OUT_SCATTER}")

    results_df = pd.DataFrame(results)

    results_df.to_csv(OUT_STATS, index=False)

    print(f"Saved statistics → {OUT_STATS}")

    plot_heatmap(
        results_df,
        datasets,
        split_index=len(real_datasets)
    )


if __name__ == "__main__":
    main()