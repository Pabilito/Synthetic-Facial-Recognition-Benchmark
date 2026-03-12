"""
plot_spearman_heatmap.py
------------------------
Reads comparison_table.csv and plots a heatmap of Spearman rho
with MorphFace and Vec2Face on one axis and the 7 real datasets
on the other, one subplot per metric.

Usage
-----
    python plot_spearman_heatmap.py [--input PATH] [--output PATH]

Defaults (same directory as script):
    --input   comparison_table.csv
    --output  spearman_heatmap.png
"""

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

SYNTH_ROWS   = ["MorphFace", "Vec2Face"]
REAL_COLS    = ["AgeDB-30", "CALFW", "CFP-FP", "CPLFW", "IJB-B", "IJB-C", "LFW"]
# Internal CSV names → display names
COL_RENAME   = {"IJBB": "IJB-B", "IJBC": "IJB-C"}
METRIC_ORDER = ["accuracy", "ZeroFMR", "EER", "FMR100", "FMR1000"]
METRIC_LABEL = {
    "accuracy": "Accuracy", "ZeroFMR": "ZeroFMR",
    "EER": "EER", "FMR100": "FMR100", "FMR1000": "FMR1000",
}
# Metrics that have no IJB-B / IJB-C data (accuracy)
SKIP_IJB = {"accuracy"}


def build_pivot(df, metric):
    """Return a DataFrame: rows=synth, cols=real datasets (display names)."""
    sub = df[df["metric"] == metric].copy()
    # Rename IJB codes to display names in a working copy
    sub["dataset_y"] = sub["dataset_y"].replace(COL_RENAME)

    pivot = pd.DataFrame(index=SYNTH_ROWS, columns=REAL_COLS, dtype=float)
    for _, row in sub.iterrows():
        if row["dataset_x"] in SYNTH_ROWS and row["dataset_y"] in REAL_COLS:
            pivot.loc[row["dataset_x"], row["dataset_y"]] = row["rho"]
    return pivot


def main(input_path: str, output_path: str):
    df = pd.read_csv(input_path)

    n_metrics = len(METRIC_ORDER)
    fig, axes = plt.subplots(
        n_metrics, 1,
        figsize=(11, 2.8 * n_metrics),
        constrained_layout=True,
    )
    if n_metrics == 1:
        axes = [axes]

    norm = plt.Normalize(vmin=0.5, vmax=1.0)
    cmap = "RdBu_r"   # red = high, blue = low

    for ax, metric in zip(axes, METRIC_ORDER):
        pivot = build_pivot(df, metric)

        # For accuracy: drop IJB-B and IJB-C columns entirely
        if metric in SKIP_IJB:
            display_cols = [c for c in REAL_COLS if c not in ("IJB-B", "IJB-C")]
        else:
            display_cols = REAL_COLS

        pivot = pivot[display_cols]
        data = pivot.values.astype(float)
        im   = ax.imshow(data, aspect="auto", cmap=cmap, norm=norm)

        ax.set_xticks(range(len(display_cols)))
        ax.set_xticklabels(display_cols, fontsize=10)
        ax.set_yticks(range(len(SYNTH_ROWS)))
        ax.set_yticklabels(SYNTH_ROWS, fontsize=10, fontstyle="italic")
        ax.set_title(METRIC_LABEL[metric], fontsize=11, fontweight="bold", pad=5)

        # Annotate cells with larger font
        for r in range(len(SYNTH_ROWS)):
            for c in range(len(display_cols)):
                val = data[r, c]
                if np.isnan(val):
                    txt, colour = "N/A", "#888888"
                else:
                    txt = f"{val:.2f}"
                    # white text on very dark cells, black otherwise
                    colour = "white" if abs(val) > 0.85 else "black"
                ax.text(c, r, txt, ha="center", va="center",
                        fontsize=11, fontweight="bold", color=colour)

        cbar = fig.colorbar(im, ax=ax, fraction=0.015, pad=0.01)
        cbar.set_label("Spearman ρ", fontsize=9)
        cbar.ax.tick_params(labelsize=8)

    fig.suptitle(
        "Spearman ρ: MorphFace & Vec2Face vs Real Datasets",
        fontsize=14, fontweight="bold",
    )

    fig.savefig(output_path, dpi=500, bbox_inches="tight")
    print(f"Saved heatmap to '{output_path}'")


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input",  default=os.path.join(script_dir, "comparison_table.csv"))
    parser.add_argument("--output", default=os.path.join(script_dir, "spearman_heatmap.png"))
    args = parser.parse_args()
    main(args.input, args.output)
