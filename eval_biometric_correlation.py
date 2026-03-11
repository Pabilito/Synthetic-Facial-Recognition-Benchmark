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

  If biometric_corr_<metric>.csv already exists it is loaded directly,
  skipping the expensive biometric_metrics.csv read and bootstrap computation.
  Only missing metrics are recomputed.  Use --recompute to force fresh runs.

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
  also score well on dataset Y" regardless of direction.

Outputs (all written to Evaluation/)
─────────────────────────────────────
  biometric_corr_<metric>.csv              – r, Fisher CI, bootstrap CI, n
  biometric_corr_heatmap_<metric>.png      – heatmap with r [lo, hi] per cell
  biometric_corr_heatmap_<metric>_noci.png – same heatmap, r values only

  Axes: Y = real datasets only; X = real + synthetic datasets.
  Synth-vs-synth pairs are excluded.

Usage
─────
  python eval_biometric_correlation.py
  python eval_biometric_correlation.py --metrics EER FMR100
  python eval_biometric_correlation.py --splits real synth
  python eval_biometric_correlation.py --in path/to/biometric_metrics.csv
  python eval_biometric_correlation.py --bootstrap-b 5000
  python eval_biometric_correlation.py --recompute   # ignore cached CSVs
"""

import argparse
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from scipy import stats

# Force all text to black globally
matplotlib.rcParams["text.color"]      = "black"
matplotlib.rcParams["axes.labelcolor"] = "black"
matplotlib.rcParams["xtick.color"]     = "black"
matplotlib.rcParams["ytick.color"]     = "black"


# ── Constants ─────────────────────────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).parent
EVAL_DIR   = SCRIPT_DIR / "Evaluation"

DEFAULT_INPUT = EVAL_DIR / "biometric_metrics.csv"

ALL_METRICS = ["EER", "FMR100", "FMR1000", "ZeroFMR", "TAR@FAR1", "TAR@FAR01"]

LOWER_IS_BETTER = {"EER", "FMR100", "FMR1000", "ZeroFMR"}

BOOTSTRAP_B = 10_000
RNG_SEED    = 42

CMAP = plt.get_cmap("tab20")


# ── Load biometric_metrics.csv ────────────────────────────────────────────────

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


# ── Load cached per-metric corr CSV ──────────────────────────────────────────

def load_corr_csv(path: Path) -> pd.DataFrame | None:
    """
    Load a previously saved biometric_corr_<metric>.csv.
    Returns None if the file doesn't exist or is missing required columns.
    """
    if not path.exists():
        return None
    df = pd.read_csv(path)
    required = {"metric", "dataset_x", "dataset_y", "n_models",
                "r", "p", "fisher_ci_lo", "fisher_ci_hi",
                "bootstrap_ci_lo", "bootstrap_ci_hi"}
    if not required.issubset(df.columns):
        print(f"  [warn] Cache {path.name} is missing columns — recomputing.")
        return None
    print(f"  [cache] Loaded {path.name}")
    return df


# ── CI helpers ────────────────────────────────────────────────────────────────

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
    B: int    = BOOTSTRAP_B,
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
    df:            pd.DataFrame,
    real_datasets: list[str],
    all_datasets:  list[str],
    metric:        str,
) -> pd.DataFrame:
    """
    Compute Pearson r + CIs for every (real_dataset_x, any_dataset_y) pair.
    Synth-vs-synth pairs are skipped entirely.

    Rows  (dataset_x) = real datasets only.
    Cols  (dataset_y) = real + synth datasets.
    """
    pivot = (
        df[df["dataset"].isin(all_datasets)]
        .groupby(["model", "dataset"])[metric]
        .mean()
        .unstack("dataset")
        .reindex(columns=all_datasets)
    )

    rows  = []
    pairs = [(x, y) for x in real_datasets for y in all_datasets]
    total = len(pairs)

    for done, (ds_x, ds_y) in enumerate(pairs, 1):
        pair_df = pivot[[ds_x, ds_y]].dropna()
        n       = len(pair_df)

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

        r_str = f"{r:.3f}" if np.isfinite(r) else "nan"
        f_str = f"[{f_lo:.3f},{f_hi:.3f}]"
        b_str = f"[{b_lo:.3f},{b_hi:.3f}]"
        print(f"  [{done:>{len(str(total))}}/{total}]"
              f"  {metric:<10}  {ds_x} \u2194 {ds_y}"
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


# ── Heatmap helpers ───────────────────────────────────────────────────────────

def _build_heatmap_matrices(
    ci_df:         pd.DataFrame,
    real_datasets: list[str],
    all_datasets:  list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Pivot ci_df into (r, ci_lo, ci_hi) matrices shaped (real x all)."""
    def _pivot(col):
        return (
            ci_df.pivot(index="dataset_x", columns="dataset_y", values=col)
            .reindex(index=real_datasets, columns=all_datasets)
        )
    return _pivot("r"), _pivot("bootstrap_ci_lo"), _pivot("bootstrap_ci_hi")


def _draw_heatmap_base(
    r_mat:         pd.DataFrame,
    real_datasets: list[str],
    all_datasets:  list[str],
    split_index:   int,
    metric:        str,
    title_suffix:  str,
) -> tuple[plt.Figure, plt.Axes, object]:
    """
    Create the figure/axes/imshow shared by both heatmap variants.
    Returns (fig, ax, im) so callers can add cell annotations then save.
    """
    n_rows = len(real_datasets)
    n_cols = len(all_datasets)

    fig, ax = plt.subplots(
        figsize=(max(9, n_cols * 1.6), max(7, n_rows * 1.4))
    )

    im = ax.imshow(
        r_mat.values.astype(float),
        vmin=-1, vmax=1, cmap="coolwarm",
        aspect="auto",
    )

    ax.set_xticks(range(n_cols))
    ax.set_yticks(range(n_rows))
    ax.set_xticklabels(all_datasets,  rotation=45, ha="right", fontsize=9,  color="black")
    ax.set_yticklabels(real_datasets, fontsize=9, color="black")
    ax.set_xlabel("Dataset (X axis — real + synthetic)", fontsize=10, color="black")
    ax.set_ylabel("Dataset (Y axis — real only)",        fontsize=10, color="black")
    ax.tick_params(colors="black")

    direction = "\u2193 better" if metric in LOWER_IS_BETTER else "\u2191 better"
    ax.set_title(
        f"Dataset Correlation Heatmap \u2014 {metric} ({direction}){title_suffix}",
        fontsize=12, color="black",
    )

    # Vertical dividing line between real and synth on X axis
    if 0 < split_index < n_cols:
        ax.axvline(split_index - 0.5, color="black", linewidth=1.8)

    cbar = fig.colorbar(im, ax=ax, label="Pearson r", shrink=0.8)
    cbar.ax.yaxis.label.set_color("black")
    cbar.ax.tick_params(colors="black")

    return fig, ax, im


# ── Heatmap with CI ───────────────────────────────────────────────────────────

def plot_heatmap(
    ci_df:         pd.DataFrame,
    real_datasets: list[str],
    all_datasets:  list[str],
    split_index:   int,
    metric:        str,
    out_path:      Path,
) -> None:
    """Heatmap of Pearson r with bootstrap CI shown inside each cell."""
    r_mat, lo_mat, hi_mat = _build_heatmap_matrices(ci_df, real_datasets, all_datasets)

    fig, ax, _ = _draw_heatmap_base(
        r_mat, real_datasets, all_datasets, split_index, metric,
        title_suffix="\nPearson r  with 95 % Bootstrap CI",
    )

    n_rows, n_cols = len(real_datasets), len(all_datasets)
    for i in range(n_rows):
        for j in range(n_cols):
            r_val  = r_mat.iloc[i, j]
            lo_val = lo_mat.iloc[i, j]
            hi_val = hi_mat.iloc[i, j]
            if np.isfinite(r_val):
                txt = f"{r_val:.2f}\n[{lo_val:.2f}, {hi_val:.2f}]"
                txt_color = "white" if abs(r_val) > 0.55 else "black"
                ax.text(j, i, txt, ha="center", va="center",
                        fontsize=6.5, color=txt_color, linespacing=1.4)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"  Saved heatmap      -> {out_path}")
    plt.close(fig)


# ── Heatmap without CI ────────────────────────────────────────────────────────

def plot_heatmap_noci(
    ci_df:         pd.DataFrame,
    real_datasets: list[str],
    all_datasets:  list[str],
    split_index:   int,
    metric:        str,
    out_path:      Path,
) -> None:
    """Heatmap of Pearson r only — no CI annotations, larger font."""
    r_mat, _, _ = _build_heatmap_matrices(ci_df, real_datasets, all_datasets)

    fig, ax, _ = _draw_heatmap_base(
        r_mat, real_datasets, all_datasets, split_index, metric,
        title_suffix="",
    )

    n_rows, n_cols = len(real_datasets), len(all_datasets)
    for i in range(n_rows):
        for j in range(n_cols):
            r_val = r_mat.iloc[i, j]
            if np.isfinite(r_val):
                txt_color = "white" if abs(r_val) > 0.55 else "black"
                ax.text(j, i, f"{r_val:.2f}", ha="center", va="center",
                        fontsize=9, color=txt_color)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"  Saved heatmap noci -> {out_path}")
    plt.close(fig)


# ── Per-metric scatter grid ───────────────────────────────────────────────────

def plot_scatter_grid(
    df:            pd.DataFrame,
    real_datasets: list[str],
    all_datasets:  list[str],
    metric:        str,
    out_path:      Path,
) -> None:
    """
    Grid of scatter plots: rows = real datasets, cols = all datasets.
    One dot per model.  Regression line + r annotation in each panel.
    """
    models       = sorted(df["model"].unique())
    n_models     = len(models)
    model_colors = {m: CMAP(i / max(n_models - 1, 1)) for i, m in enumerate(models)}

    pivot = (
        df[df["dataset"].isin(all_datasets)]
        .groupby(["model", "dataset"])[metric]
        .mean()
        .unstack("dataset")
        .reindex(columns=all_datasets)
    )

    n_rows = len(real_datasets)
    n_cols = len(all_datasets)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(4.0 * n_cols, 3.5 * n_rows),
        squeeze=False,
    )
    direction = "\u2193 better" if metric in LOWER_IS_BETTER else "\u2191 better"
    fig.suptitle(
        f"{metric} ({direction}) \u2014 Dataset Correlations",
        fontsize=13, fontweight="bold", color="black",
    )

    for r_idx, ds_x in enumerate(real_datasets):
        for c_idx, ds_y in enumerate(all_datasets):
            ax  = axes[r_idx][c_idx]
            sub = pivot[[ds_x, ds_y]].dropna()

            if sub.empty:
                ax.text(0.5, 0.5, "No common models",
                        ha="center", va="center",
                        transform=ax.transAxes, fontsize=8, color="black")
                ax.set_title(f"{ds_x} \u2194 {ds_y}", fontsize=8,
                             fontweight="bold", color="black")
                continue

            for model in sub.index:
                ax.scatter(sub.loc[model, ds_x], sub.loc[model, ds_y],
                           color=model_colors[model], s=36, zorder=3)

            x_arr = sub[ds_x].values
            y_arr = sub[ds_y].values
            mask  = np.isfinite(x_arr) & np.isfinite(y_arr)

            if mask.sum() >= 3:
                slope, intercept, r_val, p_val, _ = stats.linregress(
                    x_arr[mask], y_arr[mask])
                x_line = np.linspace(x_arr[mask].min(), x_arr[mask].max(), 100)
                ax.plot(x_line, slope * x_line + intercept,
                        linestyle="--", linewidth=1.1, color="black", alpha=0.55)
                p_str = f"p={p_val:.3f}" if p_val >= 0.001 else "p<0.001"
                ax.annotate(
                    f"r={r_val:.3f}  {p_str}",
                    xy=(0.05, 0.93), xycoords="axes fraction",
                    fontsize=7.5, color="black",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7),
                )

            ax.set_xlabel(f"{ds_x} \u2014 {metric} (%)", fontsize=7.5, color="black")
            ax.set_ylabel(f"{ds_y} \u2014 {metric} (%)", fontsize=7.5, color="black")
            ax.set_title(f"{ds_x} \u2194 {ds_y}", fontsize=8,
                         fontweight="bold", color="black")
            ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
            ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
            ax.tick_params(labelsize=6.5, colors="black")
            ax.grid(True, linestyle=":", alpha=0.4)

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
        labelcolor="black",
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"  Saved scatter grid -> {out_path}")
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
        help="Skip the scatter grid plots (faster).",
    )
    parser.add_argument(
        "--recompute", action="store_true",
        help="Ignore cached biometric_corr_<metric>.csv files and recompute everything.",
    )
    args = parser.parse_args()

    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    # ── Try to load cached CSVs first ─────────────────────────────────────────
    # If all requested metrics have a cached CSV we can skip loading
    # biometric_metrics.csv entirely (and skip the expensive bootstrap).
    cached:   dict[str, pd.DataFrame] = {}
    need_raw: list[str] = []

    if not args.recompute:
        for metric in args.metrics:
            safe_metric = metric.replace("@", "_at_")
            csv_path    = EVAL_DIR / f"biometric_corr_{safe_metric}.csv"
            loaded      = load_corr_csv(csv_path)
            if loaded is not None:
                cached[metric] = loaded
            else:
                need_raw.append(metric)
    else:
        need_raw = list(args.metrics)

    # ── Load raw biometric_metrics.csv only for metrics that need it ──────────
    df:             pd.DataFrame | None = None
    real_datasets:  list[str] = []
    synth_datasets: list[str] = []

    if need_raw:
        df = load_biometric_csv(Path(args.input_csv))

        available_splits = sorted(df["split"].unique())
        if args.splits:
            unknown = set(args.splits) - set(available_splits)
            if unknown:
                print(f"[WARN] Unknown splits: {unknown}. Available: {available_splits}")
            df = df[df["split"].isin(args.splits)]
            if df.empty:
                print("[ERROR] No data after split filter.")
                sys.exit(1)

        real_datasets  = sorted(df.loc[df["split"] == "real",  "dataset"].unique())
        synth_datasets = sorted(df.loc[df["split"] == "synth", "dataset"].unique())
        if not real_datasets and not synth_datasets:
            all_ds = sorted(df["dataset"].unique())
            real_datasets, synth_datasets = all_ds, []

        print(f"Real  datasets ({len(real_datasets)}):  {real_datasets}")
        print(f"Synth datasets ({len(synth_datasets)}): {synth_datasets}")
        print(f"Models: {sorted(df['model'].unique())}")
        print(f"Metrics to recompute: {need_raw}")
        print(f"Bootstrap B = {args.bootstrap_b:,}\n")

    elif cached:
        # Infer dataset lists from cached data
        sample         = next(iter(cached.values()))
        real_datasets  = sorted(sample["dataset_x"].unique())
        all_ds_cached  = sorted(sample["dataset_y"].unique())
        synth_datasets = [d for d in all_ds_cached if d not in real_datasets]
        print(f"[cache] Real datasets:  {real_datasets}")
        print(f"[cache] Synth datasets: {synth_datasets}")

    all_datasets = real_datasets + synth_datasets
    split_index  = len(real_datasets)   # X-axis dividing line

    all_ci_rows: list[pd.DataFrame] = []

    for metric in args.metrics:
        print(f"\n{'─' * 60}")
        print(f"  Metric: {metric}")
        print(f"{'─' * 60}")

        safe_metric = metric.replace("@", "_at_")

        # ── Get (or compute) the ci_df ────────────────────────────────────
        if metric in cached:
            ci_df    = cached[metric]
            _real_ds = sorted(ci_df["dataset_x"].unique())
            _all_ds  = real_datasets + synth_datasets
        else:
            ci_df    = compute_ci_table(df, real_datasets, all_datasets, metric)
            _real_ds = real_datasets
            _all_ds  = all_datasets

            csv_out = EVAL_DIR / f"biometric_corr_{safe_metric}.csv"
            ci_df.to_csv(csv_out, index=False, float_format="%.6f")
            print(f"  Saved CI table  -> {csv_out}")

        all_ci_rows.append(ci_df)

        # ── Heatmap with CI ───────────────────────────────────────────────
        plot_heatmap(
            ci_df, _real_ds, _all_ds, split_index, metric,
            out_path=EVAL_DIR / f"biometric_corr_heatmap_{safe_metric}.png",
        )

        # ── Heatmap without CI ────────────────────────────────────────────
        plot_heatmap_noci(
            ci_df, _real_ds, _all_ds, split_index, metric,
            out_path=EVAL_DIR / f"biometric_corr_heatmap_{safe_metric}_noci.png",
        )

        # ── Scatter grid (optional, requires raw df) ──────────────────────
        if not args.no_scatter:
            if df is not None:
                plot_scatter_grid(
                    df, _real_ds, _all_ds, metric,
                    out_path=EVAL_DIR / f"biometric_corr_scatter_{safe_metric}.png",
                )
            else:
                print("  [skip] Scatter grid requires raw data; use --recompute"
                      " or provide --in to regenerate.")

    # ── Combined CSV with all metrics ─────────────────────────────────────────
    if len(args.metrics) > 1:
        combined     = pd.concat(all_ci_rows, ignore_index=True)
        combined_out = EVAL_DIR / "biometric_corr_all_metrics.csv"
        combined.to_csv(combined_out, index=False, float_format="%.6f")
        print(f"\nSaved combined CI table -> {combined_out}")

    print("\nDone.")


if __name__ == "__main__":
    main()
