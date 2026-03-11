"""
eval_rank_consistency.py
─────────────────────────────────────────────────────────────────────────────
Analyses how consistently models are ranked across datasets for each metric.

Three complementary views:
  1. Spearman rank correlation heatmap  — are the orderings aligned?
  2. Average rank displacement heatmap  — how many places does a model shift?
  3. Per-model rank stability table     — which models rank consistently?

Input
─────
  Evaluation/biometric_metrics.csv   (produced by eval_biometric.py)

Output  (all written to Evaluation/)
──────
  rank_spearman_<metric>.png      Spearman r heatmap per metric
  rank_displacement_<metric>.png  Mean |rank_i − rank_j| heatmap per metric
  rank_stability.csv              Per-model rank std + mean displacement
  rank_stability.png              Heatmap: model × dataset rank matrix

Usage
─────
  python eval_rank_consistency.py
  python eval_rank_consistency.py --metrics EER FMR100
  python eval_rank_consistency.py --split real
  python eval_rank_consistency.py --no-plots        # CSV output only
"""

import argparse
import csv
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

# ── Optional matplotlib ───────────────────────────────────────────────────────
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR  = Path(__file__).parent
EVAL_DIR    = SCRIPT_DIR / "Evaluation"
BIO_METRICS = EVAL_DIR / "biometric_metrics.csv"
OUT_DIR     = EVAL_DIR

ALL_METRICS = ["EER", "FMR100", "FMR1000", "ZeroFMR", "TAR@FAR1", "TAR@FAR01"]
# For TAR metrics higher is better; for the rest lower is better
HIGHER_IS_BETTER = {"TAR@FAR1", "TAR@FAR01"}


# ── Data loading ──────────────────────────────────────────────────────────────

def load_metrics(path: Path, split_filter: str | None, metric_filter: list[str]) -> pd.DataFrame:
    if not path.exists():
        print(f"[error] {path} not found.")
        sys.exit(1)

    df = pd.read_csv(path)
    if split_filter:
        df = df[df["split"] == split_filter]

    # Melt wide → long so each row is (model, dataset, metric, value)
    id_cols  = ["split", "model", "dataset"]
    val_cols = [c for c in ALL_METRICS if c in df.columns]
    df = df[id_cols + val_cols].copy()
    df = df.melt(id_vars=id_cols, value_vars=val_cols,
                 var_name="metric", value_name="value")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["value"])

    if metric_filter:
        df = df[df["metric"].isin(metric_filter)]

    return df


def build_pivot(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """
    Returns a DataFrame indexed by model, columns = datasets,
    values = metric score.  Only models present in ALL datasets are kept
    so that rankings are directly comparable.
    """
    sub = df[df["metric"] == metric].copy()
    pivot = sub.pivot_table(index="model", columns="dataset",
                            values="value", aggfunc="mean")
    # Drop models with any missing dataset
    pivot = pivot.dropna()
    return pivot


def rank_pivot(pivot: pd.DataFrame, metric: str) -> pd.DataFrame:
    """
    Convert scores to ranks (1 = best).
    For EER/FMR* lower score is better → rank ascending.
    For TAR*  higher score is better   → rank descending.
    """
    ascending = metric not in HIGHER_IS_BETTER
    ranked = pivot.rank(axis=0, ascending=ascending, method="average")
    return ranked


# ── Analysis functions ────────────────────────────────────────────────────────

def spearman_matrix(ranked: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pairwise Spearman r and p-value between all dataset columns."""
    datasets = ranked.columns.tolist()
    n = len(datasets)
    r_mat = np.ones((n, n))
    p_mat = np.zeros((n, n))
    for i, j in combinations(range(n), 2):
        r, p = stats.spearmanr(ranked.iloc[:, i], ranked.iloc[:, j])
        r_mat[i, j] = r_mat[j, i] = r
        p_mat[i, j] = p_mat[j, i] = p
    r_df = pd.DataFrame(r_mat, index=datasets, columns=datasets)
    p_df = pd.DataFrame(p_mat, index=datasets, columns=datasets)
    return r_df, p_df


def displacement_matrix(ranked: pd.DataFrame) -> pd.DataFrame:
    """Mean absolute rank difference between all dataset pairs."""
    datasets = ranked.columns.tolist()
    n = len(datasets)
    mat = np.zeros((n, n))
    for i, j in combinations(range(n), 2):
        d = np.abs(ranked.iloc[:, i].values - ranked.iloc[:, j].values).mean()
        mat[i, j] = mat[j, i] = d
    return pd.DataFrame(mat, index=datasets, columns=datasets)


def stability_table(ranked: pd.DataFrame) -> pd.DataFrame:
    """
    Per-model rank stability:
      rank_std   — std of ranks across all datasets (0 = perfectly stable)
      rank_range — max rank − min rank
      mean_rank  — average rank across datasets
      datasets   — number of datasets the model appears in
    """
    tbl = pd.DataFrame({
        "mean_rank":  ranked.mean(axis=1),
        "rank_std":   ranked.std(axis=1),
        "rank_range": ranked.max(axis=1) - ranked.min(axis=1),
        "n_datasets": ranked.notna().sum(axis=1),
    }).sort_values("rank_std")
    return tbl


# ── Plotting ──────────────────────────────────────────────────────────────────

def _fig_size(n: int) -> tuple[float, float]:
    base = max(6, n * 0.7)
    return base + 2, base


def plot_heatmap(matrix: pd.DataFrame,
                 title: str,
                 out_path: Path,
                 cmap: str = "RdBu_r",
                 vmin: float = -1.0,
                 vmax: float = 1.0,
                 fmt: str = ".2f",
                 annot_size: int = 7) -> None:
    n = len(matrix)
    fig, ax = plt.subplots(figsize=_fig_size(n))

    im = ax.imshow(matrix.values, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(matrix.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(matrix.index, fontsize=8)

    for i in range(n):
        for j in range(n):
            val = matrix.values[i, j]
            txt = format(val, fmt)
            color = "white" if abs(val - (vmin + vmax) / 2) > (vmax - vmin) * 0.3 else "black"
            ax.text(j, i, txt, ha="center", va="center",
                    fontsize=annot_size, color=color)

    ax.set_title(title, fontsize=11, pad=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path.name}")


def plot_rank_matrix(ranked: pd.DataFrame, metric: str, out_path: Path) -> None:
    """Heatmap of raw ranks: rows=models, columns=datasets."""
    n_models, n_ds = ranked.shape
    fig, ax = plt.subplots(figsize=(max(8, n_ds * 0.9), max(6, n_models * 0.4)))

    im = ax.imshow(ranked.values, cmap="RdYlGn_r", aspect="auto",
                   vmin=1, vmax=n_models)
    plt.colorbar(im, ax=ax, fraction=0.02, pad=0.02, label="Rank (1 = best)")

    ax.set_xticks(range(n_ds))
    ax.set_yticks(range(n_models))
    ax.set_xticklabels(ranked.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(ranked.index, fontsize=7)

    for i in range(n_models):
        for j in range(n_ds):
            val = ranked.values[i, j]
            ax.text(j, i, f"{val:.0f}", ha="center", va="center",
                    fontsize=6, color="black")

    ax.set_title(f"Model ranks per dataset — {metric}\n(sorted by rank stability)",
                 fontsize=11, pad=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path.name}")


def plot_stability_bar(stab: pd.DataFrame, metric: str, out_path: Path) -> None:
    """Horizontal bar chart of rank_std per model."""
    fig, ax = plt.subplots(figsize=(8, max(4, len(stab) * 0.35)))
    colors = plt.cm.RdYlGn_r(
        (stab["rank_std"] - stab["rank_std"].min()) /
        (stab["rank_std"].max() - stab["rank_std"].min() + 1e-9)
    )
    ax.barh(stab.index, stab["rank_std"], color=colors)
    ax.set_xlabel("Rank std across datasets (lower = more stable)")
    ax.set_title(f"Model rank stability — {metric}", fontsize=11)
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path.name}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyse model rank consistency across datasets.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--metrics", nargs="+", default=ALL_METRICS,
                        metavar="M", help="Metrics to analyse (default: all).")
    parser.add_argument("--split", default=None, metavar="S",
                        help="Filter to one split: real / synth (default: all).")
    parser.add_argument("--input", default=str(BIO_METRICS), metavar="PATH",
                        help=f"Input CSV (default: {BIO_METRICS}).")
    parser.add_argument("--out-dir", default=str(OUT_DIR), metavar="PATH",
                        help=f"Output directory (default: {OUT_DIR}).")
    parser.add_argument("--no-plots", action="store_true",
                        help="Skip plot generation, write CSVs only.")
    parser.add_argument("--min-datasets", type=int, default=2, metavar="N",
                        help="Min datasets a model must appear in (default: 2).")
    args = parser.parse_args()

    do_plots = not args.no_plots and HAS_MPL
    if not args.no_plots and not HAS_MPL:
        print("[warn] matplotlib not available — skipping plots.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.input} ...")
    df = load_metrics(Path(args.input), args.split, args.metrics)

    all_stability = []

    for metric in args.metrics:
        pivot = build_pivot(df, metric)
        if pivot.empty:
            print(f"  [skip] {metric}: no data.")
            continue

        # Filter models that appear in enough datasets
        pivot = pivot[pivot.notna().sum(axis=1) >= args.min_datasets]
        if len(pivot) < 3:
            print(f"  [skip] {metric}: fewer than 3 models with sufficient data.")
            continue

        ranked = rank_pivot(pivot, metric)
        # Sort models by rank stability for readability
        stab = stability_table(ranked)
        ranked = ranked.loc[stab.index]

        n_models  = len(ranked)
        n_datasets = len(ranked.columns)
        print(f"\n── {metric}  ({n_models} models × {n_datasets} datasets) ──")

        # 1. Spearman heatmap
        r_df, p_df = spearman_matrix(ranked)
        print("  Spearman r (dataset pairs):")
        for (ds_x, ds_y) in combinations(r_df.columns, 2):
            r = r_df.loc[ds_x, ds_y]
            p = p_df.loc[ds_x, ds_y]
            stars = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            print(f"    {ds_x:<20} vs {ds_y:<20}  r={r:+.3f}  p={p:.4f} {stars}")

        if do_plots:
            plot_heatmap(
                r_df,
                title=f"Rank correlation (Spearman r) — {metric}\n"
                      f"{n_models} models, lower rank = better model",
                out_path=out_dir / f"rank_spearman_{metric}.png",
                cmap="RdBu_r", vmin=-1, vmax=1, fmt=".2f",
            )

        # 2. Displacement heatmap
        disp_df = displacement_matrix(ranked)
        print("  Mean rank displacement (dataset pairs):")
        for (ds_x, ds_y) in combinations(disp_df.columns, 2):
            d = disp_df.loc[ds_x, ds_y]
            print(f"    {ds_x:<20} vs {ds_y:<20}  mean |Δrank| = {d:.2f}")

        if do_plots:
            vmax_disp = max(1.0, disp_df.values.max())
            plot_heatmap(
                disp_df,
                title=f"Mean absolute rank displacement — {metric}\n"
                      f"Average positions a model shifts between datasets",
                out_path=out_dir / f"rank_displacement_{metric}.png",
                cmap="YlOrRd", vmin=0, vmax=vmax_disp, fmt=".1f",
            )

        # 3. Raw rank matrix
        if do_plots:
            plot_rank_matrix(
                ranked, metric,
                out_path=out_dir / f"rank_matrix_{metric}.png",
            )
            plot_stability_bar(
                stab, metric,
                out_path=out_dir / f"rank_stability_bar_{metric}.png",
            )

        # Accumulate stability
        stab_metric = stab[["mean_rank", "rank_std", "rank_range"]].copy()
        stab_metric.columns = [f"{metric}_{c}" for c in stab_metric.columns]
        stab_metric.index.name = "model"
        all_stability.append(stab_metric)

    # 4. Combined stability CSV
    if all_stability:
        combined = all_stability[0]
        for s in all_stability[1:]:
            combined = combined.join(s, how="outer")
        out_csv = out_dir / "rank_stability.csv"
        combined.to_csv(out_csv)
        print(f"\nWrote rank stability table → {out_csv.name}")

        # Console summary: overall most/least stable models
        std_cols = [c for c in combined.columns if c.endswith("_rank_std")]
        if std_cols:
            combined["overall_rank_std"] = combined[std_cols].mean(axis=1)
            combined_sorted = combined.sort_values("overall_rank_std")
            print("\nMost stable models (lowest mean rank std across all metrics):")
            for model, row in combined_sorted.head(5).iterrows():
                print(f"  {model:<40}  rank_std = {row['overall_rank_std']:.2f}")
            print("\nLeast stable models:")
            for model, row in combined_sorted.tail(5).iterrows():
                print(f"  {model:<40}  rank_std = {row['overall_rank_std']:.2f}")


if __name__ == "__main__":
    main()
