"""
eval_biometric_correlation.py
─────────────────────────────────────────────────────────────────────────────
Computes pairwise dataset correlations (Pearson r + 95% CIs) for each
biometric metric produced by eval_biometric.py:

    EER, FMR100, FMR1000, ZeroFMR, TAR@FAR1, TAR@FAR01

Input
─────
  Evaluation/biometric_metrics.csv   (produced by eval_biometric.py)
  Columns: split, model, dataset, n_genuine, n_impostor,
           EER, FMR100, FMR1000, ZeroFMR, TAR@FAR1, TAR@FAR01
  All metric values are percentages (float strings).

Method
──────
  Correlations are computed across models for every ordered (dataset_x,
  dataset_y) pair, separately per metric.  Two CI methods are used:

  Fisher z-transformation (parametric)
      z  = arctanh(r),  SE = 1 / sqrt(n − 3)
      95% CI: tanh(z ± 1.96·SE)   Valid for n ≥ 4.

  Bootstrap (non-parametric, B = 10 000 resamples)
      Resample model rows with replacement, recompute r.
      95% CI = [2.5th, 97.5th percentile] of the bootstrap distribution.

  Note on metric direction
      EER / FMR100 / FMR1000 / ZeroFMR  →  lower is better
      TAR@FAR1 / TAR@FAR01               →  higher is better
  A positive Pearson r always means "models that score well on dataset X
  also score well on dataset Y" regardless of direction.  The forest plot
  colours this consistently (green = positive correlation).

Outputs (all written to Evaluation/)
─────────────────────────────────────
  biometric_corr_<metric>.csv          – r, Fisher CI, bootstrap CI, n
  biometric_corr_heatmap_<metric>.png  – heatmap with r [lo, hi] per cell
  biometric_corr_forest_<metric>.png   – forest plot (unique pairs, sorted)

Usage
─────
  python eval_biometric_correlation.py
  python eval_biometric_correlation.py --metrics EER FMR100
  python eval_biometric_correlation.py --splits real synth
  python eval_biometric_correlation.py --in path/to/biometric_metrics.csv
  python eval_biometric_correlation.py --bootstrap-b 5000
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from scipy import stats


# ── Constants ─────────────────────────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).parent
EVAL_DIR   = SCRIPT_DIR / "Evaluation"

DEFAULT_INPUT = EVAL_DIR / "biometric_metrics.csv"

ALL_METRICS = ["EER", "FMR100", "FMR1000", "ZeroFMR", "TAR@FAR1", "TAR@FAR01"]

# Metrics where lower values indicate better performance.
# Used only for axis labels — does not affect r computation.
LOWER_IS_BETTER = {"EER", "FMR100", "FMR1000", "ZeroFMR"}

BOOTSTRAP_B = 10_000
RNG_SEED    = 42

CMAP = plt.get_cmap("tab20")


# ── Load ──────────────────────────────────────────────────────────────────────

def load_biometric_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        print(f"[ERROR] Input file not found: {path}")
        sys.exit(1)

    df = pd.read_csv(path)

    required = {"split", "model", "dataset"} | set(ALL_METRICS)
    missing  = required - set(df.columns)
    if missing:
        print(f"[ERROR] Missing columns in {path.name}: {missing}")
        sys.exit(1)

    for m in ALL_METRICS:
        df[m] = pd.to_numeric(df[m], errors="coerce")

    return df


# ── CI helpers (identical to eval_correlation_ci.py) ─────────────────────────

def fisher_ci(r: float, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """95% CI for Pearson r via Fisher z-transformation."""
    if n < 4 or not np.isfinite(r):
        return np.nan, np.nan
    z      = np.arctanh(np.clip(r, -0.9999, 0.9999))
    se     = 1.0 / np.sqrt(n - 3)
    z_crit = stats.norm.ppf(1 - alpha / 2)
    return (
        float(np.clip(np.tanh(z - z_crit * se), -1, 1)),
        float(np.clip(np.tanh(z + z_crit * se), -1, 1)),
    )


def bootstrap_ci(
    x: np.ndarray,
    y: np.ndarray,
    B: int   = BOOTSTRAP_B,
    seed: int = RNG_SEED,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Non-parametric bootstrap 95% CI for Pearson r."""
    mask = np.isfinite(x) & np.isfinite(y)
    xm, ym = x[mask], y[mask]
    n = len(xm)
    if n < 4:
        return np.nan, np.nan

    rng = np.random.default_rng(seed)
    rs  = np.empty(B)
    for i in range(B):
        idx    = rng.integers(0, n, size=n)
        xs, ys = xm[idx], ym[idx]
        if xs.std() == 0 or ys.std() == 0:
            rs[i] = np.nan
        else:
            rs[i] = np.corrcoef(xs, ys)[0, 1]

    rs = rs[np.isfinite(rs)]
    if len(rs) == 0:
        return np.nan, np.nan

    return (
        float(np.percentile(rs, 100 * alpha / 2)),
        float(np.percentile(rs, 100 * (1 - alpha / 2))),
    )


# ── Core computation ──────────────────────────────────────────────────────────

def compute_ci_table(
    df: pd.DataFrame,
    datasets: list[str],
    metric: str,
) -> pd.DataFrame:
    """
    For every ordered (dataset_x, dataset_y) pair, correlate the per-model
    values of `metric` and compute both CI flavours.

    The data frame is first pivoted so each row is a model and each column
    is a dataset — this guarantees that only models present in *both*
    datasets contribute to each pairwise r.
    """
    pivot = (
        df[df["dataset"].isin(datasets)]
        .groupby(["model", "dataset"])[metric]
        .mean()          # if a model appears multiple times (shouldn't), take mean
        .unstack("dataset")
        .reindex(columns=datasets)
    )

    rows  = []
    total = len(datasets) ** 2
    done  = 0

    for ds_x in datasets:
        for ds_y in datasets:
            done += 1

            pair_df = pivot[[ds_x, ds_y]].dropna()
            n       = len(pair_df)

            # Use iloc to avoid the duplicate-column issue when ds_x == ds_y
            # (selecting the same column twice by name gives a 2-col DataFrame
            # whose .values is 2-D; pearsonr then returns an array, not a scalar).
            x_arr = pair_df.iloc[:, 0].values
            y_arr = pair_df.iloc[:, 1].values

            if n >= 3:
                r, p = stats.pearsonr(x_arr, y_arr)
                r = float(r)
                p = float(p)
            else:
                r, p = np.nan, np.nan

            f_lo, f_hi = fisher_ci(r, n)
            b_lo, b_hi = bootstrap_ci(x_arr, y_arr)

            r_str    = f"{r:.3f}"    if np.isfinite(r)    else "nan"
            f_str    = f"[{f_lo:.3f},{f_hi:.3f}]"
            b_str    = f"[{b_lo:.3f},{b_hi:.3f}]"
            print(f"  [{done:>{len(str(total))}}/{total}]"
                  f"  {metric:<10}  {ds_x} ↔ {ds_y}"
                  f"  r={r_str}  Fisher={f_str}  Boot={b_str}  n={n}")

            rows.append({
                "metric":          metric,
                "dataset_x":       ds_x,
                "dataset_y":       ds_y,
                "n_models":        n,
                "r":               r,
                "p":               p,
                "fisher_ci_lo":    f_lo,
                "fisher_ci_hi":    f_hi,
                "bootstrap_ci_lo": b_lo,
                "bootstrap_ci_hi": b_hi,
            })

    return pd.DataFrame(rows)


# ── Heatmap ───────────────────────────────────────────────────────────────────

def plot_heatmap(
    ci_df:       pd.DataFrame,
    datasets:    list[str],
    split_index: int,
    metric:      str,
    out_path:    Path,
) -> None:
    """
    Heatmap of Pearson r, with bootstrap CI shown inside each cell.
    A black dividing line separates real from synthetic datasets.
    """
    def _pivot(col):
        return (
            ci_df.pivot(index="dataset_x", columns="dataset_y", values=col)
            .reindex(index=datasets, columns=datasets)
        )

    r_mat  = _pivot("r")
    lo_mat = _pivot("bootstrap_ci_lo")
    hi_mat = _pivot("bootstrap_ci_hi")

    n   = len(datasets)
    fig, ax = plt.subplots(figsize=(max(8, n * 1.5), max(7, n * 1.3)))

    im = ax.imshow(r_mat.values.astype(float), vmin=-1, vmax=1, cmap="coolwarm")

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(datasets, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(datasets, fontsize=8)

    direction = "↓ better" if metric in LOWER_IS_BETTER else "↑ better"
    ax.set_title(
        f"Dataset Correlation Heatmap — {metric} ({direction})\n"
        f"Pearson r  with 95 % Bootstrap CI",
        fontsize=11,
    )

    for i in range(n):
        for j in range(n):
            r_val  = r_mat.iloc[i, j]
            lo_val = lo_mat.iloc[i, j]
            hi_val = hi_mat.iloc[i, j]
            if np.isfinite(r_val):
                txt = f"{r_val:.2f}\n[{lo_val:.2f}, {hi_val:.2f}]"
                txt_color = "white" if abs(r_val) > 0.55 else "black"
                ax.text(j, i, txt, ha="center", va="center",
                        fontsize=6.5, color=txt_color, linespacing=1.4)

    if 0 < split_index < n:
        ax.axhline(split_index - 0.5, color="black", linewidth=1.8)
        ax.axvline(split_index - 0.5, color="black", linewidth=1.8)

    fig.colorbar(im, ax=ax, label="Pearson r", shrink=0.8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"  Saved heatmap → {out_path}")
    plt.close(fig)


# ── Forest plot ───────────────────────────────────────────────────────────────

def plot_forest(
    ci_df:    pd.DataFrame,
    metric:   str,
    out_path: Path,
) -> None:
    """
    Horizontal forest plot — one row per unique unordered dataset pair,
    sorted by r ascending so the strongest correlations appear at the top.
    Error bars show 95% bootstrap CI.
    """
    sub = ci_df[ci_df["dataset_x"] < ci_df["dataset_y"]].copy()
    sub = sub.dropna(subset=["r", "bootstrap_ci_lo", "bootstrap_ci_hi"])
    sub = sub.sort_values("r", ascending=True).reset_index(drop=True)

    if sub.empty:
        print(f"  [WARN] No valid pairs for {metric} forest plot — skipping.")
        return

    labels  = sub["dataset_x"] + " ↔ " + sub["dataset_y"]
    r_vals  = sub["r"].values
    err_lo  = r_vals - sub["bootstrap_ci_lo"].values
    err_hi  = sub["bootstrap_ci_hi"].values - r_vals
    n_vals  = sub["n_models"].values
    p_vals  = sub["p"].values

    fig_h = max(5, len(sub) * 0.5 + 2)
    fig, ax = plt.subplots(figsize=(9, fig_h))

    y_pos  = np.arange(len(sub))
    colors = ["#d73027" if r < 0 else "#1a9850" for r in r_vals]

    ax.errorbar(
        r_vals, y_pos,
        xerr=[err_lo, err_hi],
        fmt="o",
        color="black",
        elinewidth=1.2,
        capsize=4,
        markersize=0,
        zorder=2,
    )
    ax.scatter(r_vals, y_pos, color=colors, s=45, zorder=3)

    # Annotate n and significance
    for i, (r, n, p) in enumerate(zip(r_vals, n_vals, p_vals)):
        sig = ""
        if np.isfinite(p):
            if p < 0.001:
                sig = "***"
            elif p < 0.01:
                sig = "**"
            elif p < 0.05:
                sig = "*"
        ax.text(
            1.02, y_pos[i], f"n={n}{sig}",
            va="center", ha="left", fontsize=7.5,
            transform=ax.get_yaxis_transform(),
        )

    ax.axvline(0, color="grey", linewidth=0.8, linestyle="--")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=8)

    direction = "↓ better" if metric in LOWER_IS_BETTER else "↑ better"
    ax.set_xlabel("Pearson r  (95 % bootstrap CI)", fontsize=10)
    ax.set_title(
        f"Pairwise Dataset Correlations — {metric} ({direction})\n"
        f"* p<0.05  ** p<0.01  *** p<0.001",
        fontsize=11,
    )
    ax.set_xlim(-1.05, 1.2)
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
    ax.grid(axis="x", linestyle=":", alpha=0.4)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"  Saved forest plot → {out_path}")
    plt.close(fig)


# ── Per-metric scatter grid (optional but consistent with other eval scripts) ─

def plot_scatter_grid(
    df:          pd.DataFrame,
    datasets:    list[str],
    metric:      str,
    out_path:    Path,
) -> None:
    """
    N×N grid of scatter plots (one panel per dataset pair), one dot per model.
    Regression line + r annotation in each panel.
    """
    models       = sorted(df["model"].unique())
    n_models     = len(models)
    model_colors = {
        m: CMAP(i / max(n_models - 1, 1))
        for i, m in enumerate(models)
    }

    # Pivot: index=model, columns=dataset, values=metric
    pivot = (
        df[df["dataset"].isin(datasets)]
        .groupby(["model", "dataset"])[metric]
        .mean()
        .unstack("dataset")
        .reindex(columns=datasets)
    )

    n   = len(datasets)
    fig, axes = plt.subplots(n, n, figsize=(4.0 * n, 3.5 * n), squeeze=False)
    direction  = "↓ better" if metric in LOWER_IS_BETTER else "↑ better"
    fig.suptitle(
        f"{metric} ({direction}) — Dataset Correlations",
        fontsize=13, fontweight="bold",
    )

    for r_idx, ds_x in enumerate(datasets):
        for c_idx, ds_y in enumerate(datasets):
            ax  = axes[r_idx][c_idx]
            sub = pivot[[ds_x, ds_y]].dropna()

            if sub.empty:
                ax.text(0.5, 0.5, "No common models",
                        ha="center", va="center",
                        transform=ax.transAxes, fontsize=8, color="grey")
                ax.set_title(f"{ds_x} ↔ {ds_y}", fontsize=8, fontweight="bold")
                continue

            for model in sub.index:
                color = model_colors[model]
                ax.scatter(
                    sub.loc[model, ds_x],
                    sub.loc[model, ds_y],
                    color=color, s=36, zorder=3,
                )

            x_arr = sub[ds_x].values
            y_arr = sub[ds_y].values
            mask  = np.isfinite(x_arr) & np.isfinite(y_arr)

            if mask.sum() >= 3:
                slope, intercept, r_val, p_val, _ = stats.linregress(
                    x_arr[mask], y_arr[mask]
                )
                x_line = np.linspace(x_arr[mask].min(), x_arr[mask].max(), 100)
                ax.plot(x_line, slope * x_line + intercept,
                        linestyle="--", linewidth=1.1, color="black", alpha=0.55)

                p_str = f"p={p_val:.3f}" if p_val >= 0.001 else "p<0.001"
                ax.annotate(
                    f"r={r_val:.3f}  {p_str}",
                    xy=(0.05, 0.93), xycoords="axes fraction",
                    fontsize=7.5,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7),
                )

            ax.set_xlabel(f"{ds_x} — {metric} (%)", fontsize=7.5)
            ax.set_ylabel(f"{ds_y} — {metric} (%)", fontsize=7.5)
            ax.set_title(f"{ds_x} ↔ {ds_y}", fontsize=8, fontweight="bold")
            ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
            ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
            ax.tick_params(labelsize=6.5)
            ax.grid(True, linestyle=":", alpha=0.4)

    # Shared legend
    handles = [
        plt.Line2D([0], [0], marker="o", color="w",
                   markerfacecolor=model_colors[m], markersize=7, label=m)
        for m in models
    ]
    fig.legend(
        handles=handles, title="Model",
        loc="lower center", ncol=min(n_models, 4),
        fontsize=7.5, title_fontsize=8.5,
        bbox_to_anchor=(0.5, -0.02),
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"  Saved scatter grid → {out_path}")
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Biometric metric correlations with 95% CIs across datasets.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--in", dest="input_csv", type=str, default=str(DEFAULT_INPUT),
        metavar="PATH",
        help=f"biometric_metrics.csv path (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--metrics", nargs="+", default=ALL_METRICS, metavar="METRIC",
        choices=ALL_METRICS,
        help=f"Metrics to analyse (default: all). Choices: {ALL_METRICS}",
    )
    parser.add_argument(
        "--splits", nargs="+", default=None, metavar="SPLIT",
        help="Process only these split names (e.g. real synth). Default: all.",
    )
    parser.add_argument(
        "--bootstrap-b", type=int, default=BOOTSTRAP_B, metavar="B",
        help=f"Bootstrap resamples (default: {BOOTSTRAP_B})",
    )
    parser.add_argument(
        "--no-scatter", action="store_true",
        help="Skip the N×N scatter grid plots (faster).",
    )
    args = parser.parse_args()

    df = load_biometric_csv(Path(args.input_csv))

    # ── Filter splits ─────────────────────────────────────────────────────────
    available_splits = sorted(df["split"].unique())
    if args.splits:
        unknown = set(args.splits) - set(available_splits)
        if unknown:
            print(f"[WARN] Unknown splits: {unknown}. Available: {available_splits}")
        df = df[df["split"].isin(args.splits)]
        if df.empty:
            print("[ERROR] No data after split filter.")
            sys.exit(1)

    # Combine splits: when a model appears in both real and synth splits for
    # the same metric+dataset, keep both rows (they refer to different
    # evaluation datasets, so the dataset column already distinguishes them).
    # The pivot in compute_ci_table groups by (model, dataset) and takes the
    # mean — if a model/dataset combo truly appears twice within one split,
    # this averages them gracefully.

    real_datasets  = sorted(df.loc[df["split"] == "real",  "dataset"].unique())
    synth_datasets = sorted(df.loc[df["split"] == "synth", "dataset"].unique())
    # For custom splits fall back to all datasets
    if not real_datasets and not synth_datasets:
        all_datasets = sorted(df["dataset"].unique())
        real_datasets, synth_datasets = all_datasets, []
    datasets    = real_datasets + synth_datasets
    split_index = len(real_datasets)

    print(f"Real  datasets ({len(real_datasets)}):  {real_datasets}")
    print(f"Synth datasets ({len(synth_datasets)}): {synth_datasets}")
    print(f"Models: {sorted(df['model'].unique())}")
    print(f"Metrics: {args.metrics}")
    print(f"Bootstrap B = {args.bootstrap_b:,}\n")

    all_ci_rows = []   # accumulate across metrics for the combined CSV

    for metric in args.metrics:
        print(f"\n{'─' * 60}")
        print(f"  Metric: {metric}")
        print(f"{'─' * 60}")

        safe_metric = metric.replace("@", "_at_")  # safe for filenames

        ci_df = compute_ci_table(df, datasets, metric)
        all_ci_rows.append(ci_df)

        # Per-metric CSV
        csv_out = EVAL_DIR / f"biometric_corr_{safe_metric}.csv"
        ci_df.to_csv(csv_out, index=False, float_format="%.6f")
        print(f"  Saved CI table  → {csv_out}")

        # Heatmap
        plot_heatmap(
            ci_df, datasets, split_index, metric,
            out_path=EVAL_DIR / f"biometric_corr_heatmap_{safe_metric}.png",
        )

        # Forest plot
        plot_forest(
            ci_df, metric,
            out_path=EVAL_DIR / f"biometric_corr_forest_{safe_metric}.png",
        )

        # Scatter grid (optional)
        if not args.no_scatter:
            plot_scatter_grid(
                df, datasets, metric,
                out_path=EVAL_DIR / f"biometric_corr_scatter_{safe_metric}.png",
            )

    # ── Combined CSV with all metrics ─────────────────────────────────────────
    if len(args.metrics) > 1:
        combined = pd.concat(all_ci_rows, ignore_index=True)
        combined_out = EVAL_DIR / "biometric_corr_all_metrics.csv"
        combined.to_csv(combined_out, index=False, float_format="%.6f")
        print(f"\nSaved combined CI table → {combined_out}")

    print("\nDone.")


if __name__ == "__main__":
    main()
