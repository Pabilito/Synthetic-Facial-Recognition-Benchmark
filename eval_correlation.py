"""
eval_correlation.py
─────────────────────────────────────────────────────────────────────────────
Reads Evaluation/real_summary.csv and Evaluation/synth_summary.csv and plots
correlations between model accuracies for ALL dataset pairs.

Each panel shows:
  - One point per model
  - Pearson r and p-value
  - Error bars
  - Linear regression line

Output:
Evaluation/correlation_all_datasets.png
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

OUT_FIG = EVAL_DIR / "correlation_all_datasets.png"

CMAP = plt.get_cmap("tab20")

# ── Load helper ───────────────────────────────────────────────────────────────


def load(path: Path, label: str) -> pd.DataFrame:
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


# ── Plot helper ───────────────────────────────────────────────────────────────


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

    if res:

        x_line, y_line, r, p = res

        ax.plot(
            x_line,
            y_line,
            linestyle="--",
            linewidth=1.2,
            color="black",
            alpha=0.6,
        )

        p_str = f"p={p:.3f}" if p >= 0.001 else "p<0.001"

        ax.annotate(
            f"r={r:.3f}  {p_str}",
            xy=(0.05, 0.93),
            xycoords="axes fraction",
            fontsize=8,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7),
        )

    ax.set_xlabel(f"{ds_x} accuracy (%)", fontsize=8)
    ax.set_ylabel(f"{ds_y} accuracy (%)", fontsize=8)

    ax.set_title(f"{ds_x}  ↔  {ds_y}", fontsize=9, fontweight="bold")

    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))

    ax.tick_params(labelsize=7)
    ax.grid(True, linestyle=":", alpha=0.4)


# ── Main ──────────────────────────────────────────────────────────────────────


def main():

    real_df = load(REAL_CSV, "real_summary.csv")
    synth_df = load(SYNTH_CSV, "synth_summary.csv")

    real_df["source"] = "real"
    synth_df["source"] = "synth"

    all_df = pd.concat([real_df, synth_df], ignore_index=True)

    datasets = sorted(all_df["dataset"].unique())

    all_models = sorted(all_df["model"].unique())

    print(f"Datasets : {datasets}")
    print(f"Models   : {all_models}")

    model_colors = {
        m: CMAP(i / max(len(all_models) - 1, 1)) for i, m in enumerate(all_models)
    }

    n_rows = len(datasets)
    n_cols = len(datasets)

    if n_rows == 0:
        print("[ERROR] No datasets found.")
        sys.exit(1)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4.5 * n_cols, 4.0 * n_rows),
        squeeze=False,
    )

    fig.suptitle(
        "Model Accuracy Correlation Across All Datasets",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )

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

                ax.set_title(f"{ds_x} ↔ {ds_y}", fontsize=9)

                continue

            plot_panel(ax, merged, ds_x, ds_y, model_colors)

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
        for m in all_models
    ]

    fig.legend(
        handles=handles,
        title="Model",
        loc="lower center",
        ncol=min(len(all_models), 4),
        fontsize=8,
        title_fontsize=9,
        bbox_to_anchor=(0.5, -0.03),
    )

    fig.tight_layout()

    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(OUT_FIG, dpi=150, bbox_inches="tight")

    print(f"\nSaved → {OUT_FIG}")


if __name__ == "__main__":
    main()