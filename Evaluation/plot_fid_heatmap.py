"""
plot_fid_heatmap.py
-------------------
Reads fid_results.csv (InceptionV3) and fid_results_r18.csv (IResNet-R18)
and produces:
  1. fid_comparison_table.csv  — long-format table with all FID values
  2. fid_heatmap.png           — heatmap: real datasets (y-axis) vs
                                 all datasets (x-axis), one subplot per extractor

Usage
-----
    python plot_fid_heatmap.py [--inception PATH] [--iresnet PATH] [--output-dir PATH]

Defaults: same directory as script.
"""

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.colors import LogNorm

# ---------------------------------------------------------------------------
# Name mapping: internal CSV names → display names
# ---------------------------------------------------------------------------
NAME_MAP = {
    # Real
    "real/agedb_30_112x112": "AgeDB-30",
    "real/calfw_112x112":    "CALFW",
    "real/cfp_fp_112x112":   "CFP-FP",
    "real/cplfw_112x112":    "CPLFW",
    "real/lfw_112x112":      "LFW",
    # Synth
    "synth/CemiFace":        "CemiFace",
    "synth/Control10k":      "ControlFace10k",
    "synth/DCFace":          "DCFace",
    "synth/Digi2Real":       "Digi2Real",
    "synth/HyperFace":       "HyperFace",
    "synth/IDiffFace":       "IDiff-Face",
    "synth/Langevin-DisCo":  "Langevin-DisCo",
    "synth/MorphFace":       "MorphFace",
    "synth/SFace2":          "SFace2",
    "synth/SynFace":         "SynFace",
    "synth/SynMulti-PIE":    "SynMulti-PIE",
    "synth/Vec2Face":        "Vec2Face",
}

REAL_ORDER  = ["AgeDB-30", "CALFW", "CFP-FP", "CPLFW", "LFW"]
SYNTH_ORDER = ["CemiFace", "ControlFace10k", "DCFace", "Digi2Real", "HyperFace",
               "IDiff-Face", "Langevin-DisCo", "MorphFace", "SFace2",
               "SynFace", "SynMulti-PIE", "Vec2Face"]

# Column order: real first, then synth
ALL_ORDER   = REAL_ORDER + SYNTH_ORDER


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_fid_csv(path: str) -> pd.DataFrame:
    """Load a FID CSV and return a symmetric (a,b) → fid lookup."""
    df = pd.read_csv(path)
    # Keep only numeric FID rows
    df = df[pd.to_numeric(df["fid"], errors="coerce").notna()].copy()
    df["fid"] = df["fid"].astype(float)
    return df


def build_matrix(df: pd.DataFrame, row_names: list, col_names: list) -> pd.DataFrame:
    """
    Build a pivot matrix: rows × cols of FID values.
    FID is symmetric so we check both (a,b) and (b,a).
    """
    # Build lookup in display-name space
    lookup = {}
    for _, row in df.iterrows():
        a = NAME_MAP.get(row["dataset_a"])
        b = NAME_MAP.get(row["dataset_b"])
        if a and b:
            lookup[(a, b)] = row["fid"]
            lookup[(b, a)] = row["fid"]

    mat = pd.DataFrame(index=row_names, columns=col_names, dtype=float)
    for r in row_names:
        for c in col_names:
            if r == c:
                mat.loc[r, c] = 0.0
            else:
                mat.loc[r, c] = lookup.get((r, c), np.nan)
    return mat


def build_long_table(inc_df, r18_df) -> pd.DataFrame:
    """Merge both CSVs into a long-format comparison table."""
    rows = []
    for _, row in inc_df.iterrows():
        a = NAME_MAP.get(row["dataset_a"])
        b = NAME_MAP.get(row["dataset_b"])
        if a and b:
            rows.append({"dataset_a": a, "dataset_b": b,
                         "fid_inception": row["fid"], "fid_iresnet": np.nan})

    # Build lookup for r18
    r18_lookup = {}
    for _, row in r18_df.iterrows():
        a = NAME_MAP.get(row["dataset_a"])
        b = NAME_MAP.get(row["dataset_b"])
        if a and b:
            r18_lookup[(a, b)] = row["fid"]
            r18_lookup[(b, a)] = row["fid"]

    for r in rows:
        key = (r["dataset_a"], r["dataset_b"])
        r["fid_iresnet"] = r18_lookup.get(key, np.nan)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_heatmap(mat: pd.DataFrame, ax: plt.Axes, title: str, cmap: str,
                 vmin: float, vmax: float, show_xlabels: bool):
    data = mat.values.astype(float)

    # Log scale for better contrast across wide FID range
    norm = LogNorm(vmin=max(vmin, 1e-1), vmax=vmax)
    im   = ax.imshow(data, aspect="auto", cmap=cmap, norm=norm,
                     interpolation="nearest")

    n_rows, n_cols = data.shape

    # Grid lines
    ax.set_xticks(np.arange(-0.5, n_cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_rows, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.6)
    ax.tick_params(which="minor", length=0)

    # Separator line between real and synth columns
    n_real = len(REAL_ORDER)
    ax.axvline(x=n_real - 0.5, color="white", linewidth=2.5, linestyle="--")

    # Separator line between real and synth rows
    ax.axhline(y=n_real - 0.5, color="white", linewidth=2.5, linestyle="--")

    # Axis labels
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(mat.index, fontsize=8.5)

    if show_xlabels:
        ax.set_xticks(range(n_cols))
        ax.set_xticklabels(mat.columns, fontsize=8.5,
                           rotation=40, ha="right", rotation_mode="anchor")
    else:
        ax.set_xticks([])

    # Annotate cells
    for r in range(n_rows):
        for c in range(n_cols):
            val = data[r, c]
            if np.isnan(val):
                continue
            if val == 0.0:
                txt = "–"
            elif val >= 1000:
                txt = f"{val/1000:.1f}k"
            elif val >= 100:
                txt = f"{val:.0f}"
            elif val >= 10:
                txt = f"{val:.1f}"
            else:
                txt = f"{val:.2f}"

            # Text color based on brightness of cell
            normed = (np.log10(max(val, 0.1)) - np.log10(max(vmin, 0.1))) / \
                     (np.log10(vmax) - np.log10(max(vmin, 0.1)))
            colour = "white" if normed > 0.6 else "black"
            ax.text(c, r, txt, ha="center", va="center",
                    fontsize=7, fontweight="bold", color=colour)

    ax.set_title(title, fontsize=11, fontweight="bold", pad=6)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.018, pad=0.01)
    cbar.set_label("FID (log scale)", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    # Column-group labels above top axis
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks([(n_real - 1) / 2, n_real + (n_cols - n_real - 1) / 2])
    ax2.set_xticklabels(["Real", "Synthetic"], fontsize=9,
                         fontweight="bold", color="#444444")
    ax2.tick_params(length=0)

    return im


def main(inception_path: str, iresnet_path: str, output_dir: str):
    # --- Load ---
    inc_df = load_fid_csv(inception_path)
    r18_df = load_fid_csv(iresnet_path)

    # --- Build matrices: rows = all datasets, cols = all datasets ---
    # We show real on Y axis, all on X axis
    inc_mat = build_matrix(inc_df, row_names=REAL_ORDER, col_names=ALL_ORDER)
    r18_mat = build_matrix(r18_df, row_names=REAL_ORDER, col_names=ALL_ORDER)

    # --- Save long table ---
    long_df = build_long_table(inc_df, r18_df)
    table_path = os.path.join(output_dir, "fid_comparison_table.csv")
    long_df.to_csv(table_path, index=False)
    print(f"Saved table → {table_path}")

    # --- Global vmin/vmax across both matrices for consistent colorscale ---
    all_vals = pd.concat([inc_mat, r18_mat]).values.astype(float)
    finite   = all_vals[np.isfinite(all_vals) & (all_vals > 0)]
    vmin     = max(float(np.nanpercentile(finite, 2)),  0.5)
    vmax     = float(np.nanpercentile(finite, 98))

    # --- Figure: 2 subplots (one per extractor) ---
    fig, axes = plt.subplots(
        2, 1,
        figsize=(16, 10),
        constrained_layout=True,
    )
    fig.patch.set_facecolor("#F7F7F5")

    cmap = "RdYlGn_r"   # green=low FID (similar), red=high FID (dissimilar)

    plot_heatmap(inc_mat, axes[0],
                 title="FID — InceptionV3 features (2048-d, ImageNet)",
                 cmap=cmap, vmin=vmin, vmax=vmax, show_xlabels=False)

    plot_heatmap(r18_mat, axes[1],
                 title="FID — IResNet-R18 features (512-d, MS1MV3 ArcFace)",
                 cmap=cmap, vmin=vmin, vmax=vmax, show_xlabels=True)

    fig.suptitle(
        "Fréchet Inception Distance: Real Datasets (rows) vs All Datasets (columns)",
        fontsize=13, fontweight="bold", y=1.01,
    )

    out_path = os.path.join(output_dir, "fid_heatmap.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"Saved heatmap → {out_path}")


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--inception",   default=os.path.join(script_dir, "fid_results.csv"))
    parser.add_argument("--iresnet",     default=os.path.join(script_dir, "fid_results_r18.csv"))
    parser.add_argument("--output-dir",  default=script_dir)
    args = parser.parse_args()
    main(args.inception, args.iresnet, args.output_dir)
