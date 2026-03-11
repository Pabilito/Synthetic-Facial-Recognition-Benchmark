"""
eval_rank_consistency.py
─────────────────────────────────────────────────────────────────────────────
Analyses how consistently datasets rank models, i.e. whether two datasets
agree on which models are better.

Two core views (+ CI variants):
  1. rank_spearman_<metric>.png        Spearman r between every dataset pair.
                                       r=1: identical ranking. r=-1: reversed.
  2. rank_displacement_<metric>.png    Mean |rank_i − rank_j| between every
                                       dataset pair. 0 = identical ranking.
                                       Max possible = (n_models-1)/2.

  CI variants (bootstrap 95%, B=10,000):
  3. rank_spearman_ci_<metric>.png     Spearman r annotated with [lo, hi].
  4. rank_displacement_ci_<metric>.png Displacement annotated with [lo, hi].

Input
─────
  Evaluation/biometric_metrics.csv   (produced by eval_biometric.py)

Output  (all written to Evaluation/)
──────
  rank_spearman_<metric>.png
  rank_spearman_ci_<metric>.png
  rank_displacement_<metric>.png
  rank_displacement_ci_<metric>.png
  rank_consistency.csv               Tidy table of all pairwise values + CIs

Usage
─────
  python eval_rank_consistency.py
  python eval_rank_consistency.py --metrics EER FMR100
  python eval_rank_consistency.py --split real
  python eval_rank_consistency.py --no-ci        # skip bootstrap (faster)
  python eval_rank_consistency.py --no-plots     # CSV output only
"""

import argparse
import csv
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR  = Path(__file__).parent
EVAL_DIR    = SCRIPT_DIR / "Evaluation"
BIO_METRICS = EVAL_DIR / "biometric_metrics.csv"
OUT_DIR     = EVAL_DIR

ALL_METRICS      = ["EER", "FMR100", "FMR1000", "ZeroFMR", "TAR@FAR1", "TAR@FAR01"]
HIGHER_IS_BETTER = {"TAR@FAR1", "TAR@FAR01"}
BOOTSTRAP_B      = 10_000
BOOTSTRAP_SEED   = 42


# ── Data loading ──────────────────────────────────────────────────────────────

def load_metrics(path: Path, split_filter: str | None,
                 metric_filter: list[str]) -> pd.DataFrame:
    if not path.exists():
        print(f"[error] {path} not found.")
        sys.exit(1)
    df = pd.read_csv(path)
    if split_filter:
        df = df[df["split"] == split_filter]
    id_cols  = ["split", "model", "dataset"]
    val_cols = [c for c in ALL_METRICS if c in df.columns]
    df = df[id_cols + val_cols].melt(id_vars=id_cols, value_vars=val_cols,
                                      var_name="metric", value_name="value")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["value"])
    if metric_filter:
        df = df[df["metric"].isin(metric_filter)]
    return df


def build_ranked(df: pd.DataFrame, metric: str) -> tuple[pd.DataFrame, int]:
    """
    Returns (ranked, n_real):
      ranked  — models x datasets matrix of ranks (1 = best),
                columns ordered real-first then synth.
      n_real  — number of real dataset columns (used for separator line).
    Only models present in every dataset are kept.
    """
    sub      = df[df["metric"] == metric].copy()
    real_ds  = sorted(sub[sub["split"] == "real" ]["dataset"].unique())
    synth_ds = sorted(sub[sub["split"] == "synth"]["dataset"].unique())
    ordered  = real_ds + synth_ds

    pivot   = sub.pivot_table(index="model", columns="dataset",
                              values="value", aggfunc="mean")
    ordered = [c for c in ordered if c in pivot.columns]
    pivot   = pivot[ordered].dropna()

    ascending = metric not in HIGHER_IS_BETTER
    ranked    = pivot.rank(axis=0, ascending=ascending, method="average")
    n_real    = len([c for c in real_ds if c in pivot.columns])
    return ranked, n_real


# ── Pairwise metrics ──────────────────────────────────────────────────────────

def pairwise_spearman(ranked: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Spearman r and p-value for every dataset pair."""
    ds    = ranked.columns.tolist()
    n     = len(ds)
    r_mat = np.eye(n)
    p_mat = np.zeros((n, n))
    for i, j in combinations(range(n), 2):
        r, p = stats.spearmanr(ranked.iloc[:, i], ranked.iloc[:, j])
        r_mat[i, j] = r_mat[j, i] = r
        p_mat[i, j] = p_mat[j, i] = p
    return (pd.DataFrame(r_mat, index=ds, columns=ds),
            pd.DataFrame(p_mat, index=ds, columns=ds))


def pairwise_displacement(ranked: pd.DataFrame) -> pd.DataFrame:
    """Mean absolute rank difference for every dataset pair."""
    ds  = ranked.columns.tolist()
    n   = len(ds)
    mat = np.zeros((n, n))
    for i, j in combinations(range(n), 2):
        d = np.abs(ranked.iloc[:, i].values - ranked.iloc[:, j].values).mean()
        mat[i, j] = mat[j, i] = d
    return pd.DataFrame(mat, index=ds, columns=ds)


def bootstrap_ci_matrices(ranked: pd.DataFrame,
                           B: int = BOOTSTRAP_B,
                           seed: int = BOOTSTRAP_SEED,
                           alpha: float = 0.05,
                           ) -> dict[str, tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Bootstrap 95% CIs for Spearman r and mean displacement simultaneously.
    Resamples models (rows) with replacement B times.

    Vectorised implementation — no Python loop over dataset pairs:
      - Spearman r: computed via numpy corrcoef on the full (n_ds x n_ds)
        correlation matrix at once per resample.
      - Displacement: computed via broadcasting over all pairs at once.
    This is ~100x faster than calling scipy.spearmanr per pair per resample.
    """
    ds       = ranked.columns.tolist()
    n_ds     = len(ds)
    n_models = len(ranked)
    rng      = np.random.default_rng(seed)
    arr      = ranked.values.astype(np.float64)   # (n_models, n_ds)
    pairs    = list(combinations(range(n_ds), 2))
    n_pairs  = len(pairs)
    ix       = [p[0] for p in pairs]
    jx       = [p[1] for p in pairs]

    r_boot = np.empty((B, n_pairs), dtype=np.float64)
    d_boot = np.empty((B, n_pairs), dtype=np.float64)

    for b in range(B):
        sample = arr[rng.integers(0, n_models, size=n_models)]  # (n_models, n_ds)

        # Spearman r for all pairs at once via corrcoef on full matrix
        # corrcoef returns (n_ds, n_ds); extract upper triangle pairs
        corr         = np.corrcoef(sample.T)   # (n_ds, n_ds)
        r_boot[b]    = corr[ix, jx]

        # Mean absolute displacement for all pairs at once via broadcasting
        # sample[:, ix] shape (n_models, n_pairs), same for jx
        d_boot[b]    = np.abs(sample[:, ix] - sample[:, jx]).mean(axis=0)

    def _to_dfs(boot):
        lo_mat = np.eye(n_ds)
        hi_mat = np.ones((n_ds, n_ds))
        lo_vals = np.percentile(boot, 100 * alpha / 2,       axis=0)
        hi_vals = np.percentile(boot, 100 * (1 - alpha / 2), axis=0)
        for k, (i, j) in enumerate(pairs):
            lo_mat[i, j] = lo_mat[j, i] = lo_vals[k]
            hi_mat[i, j] = hi_mat[j, i] = hi_vals[k]
        return (pd.DataFrame(lo_mat, index=ds, columns=ds),
                pd.DataFrame(hi_mat, index=ds, columns=ds))

    return {
        "spearman":     _to_dfs(r_boot),
        "displacement": _to_dfs(d_boot),
    }


# ── Plotting ──────────────────────────────────────────────────────────────────

def _fig_size(n: int) -> tuple[float, float]:
    base = max(6, n * 0.75)
    return base + 2, base


def _draw_separator(ax, split_at: int, n: int) -> None:
    if 0 < split_at < n:
        kw = dict(color="black", linewidth=2.0)
        ax.axhline(split_at - 0.5, **kw)
        ax.axvline(split_at - 0.5, **kw)


def _base_heatmap(matrix, cmap, vmin, vmax, split_at):
    n   = len(matrix)
    fig, ax = plt.subplots(figsize=_fig_size(n))
    im  = ax.imshow(matrix.values, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(matrix.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(matrix.index, fontsize=8)
    _draw_separator(ax, split_at, n)
    return fig, ax, (vmin + vmax) / 2, (vmax - vmin)


def plot_heatmap(matrix: pd.DataFrame, title: str, out_path: Path,
                 cmap: str, vmin: float, vmax: float,
                 fmt: str, split_at: int) -> None:
    fig, ax, mid, span = _base_heatmap(matrix, cmap, vmin, vmax, split_at)
    n = len(matrix)
    annot_size = max(6, 9 - n // 4)
    for i in range(n):
        for j in range(n):
            v     = matrix.values[i, j]
            color = "white" if abs(v - mid) > span * 0.3 else "black"
            ax.text(j, i, format(v, fmt), ha="center", va="center",
                    fontsize=annot_size, color=color)
    ax.set_title(title, fontsize=11, pad=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=500, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path.name}")


def plot_heatmap_ci(matrix: pd.DataFrame, ci_lo: pd.DataFrame,
                    ci_hi: pd.DataFrame, title: str, out_path: Path,
                    cmap: str, vmin: float, vmax: float,
                    split_at: int) -> None:
    fig, ax, mid, span = _base_heatmap(matrix, cmap, vmin, vmax, split_at)
    n          = len(matrix)
    annot_size = max(5, 8 - n // 4)
    for i in range(n):
        for j in range(n):
            v     = matrix.values[i, j]
            lo    = ci_lo.values[i, j]
            hi    = ci_hi.values[i, j]
            color = "white" if abs(v - mid) > span * 0.3 else "black"
            ax.text(j, i, f"{v:.2f}\n[{lo:.2f}, {hi:.2f}]",
                    ha="center", va="center", fontsize=annot_size,
                    color=color, linespacing=1.3)
    ax.set_title(title, fontsize=11, pad=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=500, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path.name}")


# ── Output CSV ────────────────────────────────────────────────────────────────

def build_output_rows(metric, r_df, p_df, disp_df,
                      r_lo, r_hi, d_lo, d_hi, n_models) -> list[dict]:
    rows = []
    for ds_x, ds_y in combinations(r_df.columns, 2):
        row = {
            "metric":     metric,
            "dataset_x":  ds_x,
            "dataset_y":  ds_y,
            "n_models":   n_models,
            "spearman_r": round(r_df.loc[ds_x, ds_y], 6),
            "spearman_p": round(p_df.loc[ds_x, ds_y], 6),
            "mean_disp":  round(disp_df.loc[ds_x, ds_y], 4),
        }
        if r_lo is not None:
            row["spearman_ci_lo"] = round(r_lo.loc[ds_x, ds_y], 6)
            row["spearman_ci_hi"] = round(r_hi.loc[ds_x, ds_y], 6)
            row["disp_ci_lo"]     = round(d_lo.loc[ds_x, ds_y], 4)
            row["disp_ci_hi"]     = round(d_hi.loc[ds_x, ds_y], 4)
        rows.append(row)
    return rows


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyse how consistently datasets rank models.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--metrics",     nargs="+", default=ALL_METRICS, metavar="M")
    parser.add_argument("--split",       default=None, metavar="S",
                        help="Filter to one split: real / synth (default: all).")
    parser.add_argument("--input",       default=str(BIO_METRICS), metavar="PATH")
    parser.add_argument("--out-dir",     default=str(OUT_DIR),     metavar="PATH")
    parser.add_argument("--no-ci",       action="store_true",
                        help="Skip bootstrap CI computation (faster).")
    parser.add_argument("--no-plots",    action="store_true",
                        help="Skip plot generation, write CSV only.")
    parser.add_argument("--bootstrap-b", type=int, default=BOOTSTRAP_B, metavar="B",
                        help=f"Bootstrap resamples (default: {BOOTSTRAP_B}).")
    args = parser.parse_args()

    do_plots = not args.no_plots and HAS_MPL
    do_ci    = not args.no_ci
    if not args.no_plots and not HAS_MPL:
        print("[warn] matplotlib not available — skipping plots.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.input} ...")
    df = load_metrics(Path(args.input), args.split, args.metrics)

    all_rows = []

    for metric in args.metrics:
        ranked, n_real = build_ranked(df, metric)
        if ranked.empty or len(ranked) < 3 or len(ranked.columns) < 2:
            print(f"  [skip] {metric}: insufficient data.")
            continue

        n_models   = len(ranked)
        n_datasets = len(ranked.columns)
        print(f"\n── {metric}  ({n_models} models x {n_datasets} datasets) ──")

        r_df, p_df = pairwise_spearman(ranked)
        disp_df    = pairwise_displacement(ranked)

        r_lo = r_hi = d_lo = d_hi = None
        if do_ci:
            print(f"  Computing bootstrap CIs (B={args.bootstrap_b:,}) ...")
            cis       = bootstrap_ci_matrices(ranked, B=args.bootstrap_b)
            r_lo, r_hi = cis["spearman"]
            d_lo, d_hi = cis["displacement"]

        # Console summary
        print(f"  {'Dataset pair':<43} {'Spearman r':>10}  {'mean |Δrank|':>12}")
        print(f"  {'-'*43} {'-'*10}  {'-'*12}")
        for ds_x, ds_y in combinations(r_df.columns, 2):
            r = r_df.loc[ds_x, ds_y]
            p = p_df.loc[ds_x, ds_y]
            d = disp_df.loc[ds_x, ds_y]
            stars = "***" if p < 0.001 else "** " if p < 0.01 else "*  " if p < 0.05 else "   "
            print(f"  {ds_x:<20} vs {ds_y:<20}  {r:+.3f}{stars}  {d:>8.2f}")

        all_rows.extend(build_output_rows(
            metric, r_df, p_df, disp_df, r_lo, r_hi, d_lo, d_hi, n_models))

        if not do_plots:
            continue

        vmax_disp  = max(1.0, disp_df.values.max())
        max_rank   = (n_models - 1) / 2

        plot_heatmap(
            r_df,
            title=(f"Dataset rank correlation (Spearman r) — {metric}\n"
                   f"{n_models} models  |  r=+1: identical ranking   r=-1: reversed"),
            out_path=out_dir / f"rank_spearman_{metric}.png",
            cmap="RdBu_r", vmin=-1, vmax=1, fmt=".2f", split_at=n_real,
        )

        plot_heatmap(
            disp_df,
            title=(f"Mean absolute rank displacement — {metric}\n"
                   f"Average positions a model shifts between datasets"
                   f"  (0 = identical,  max \u2248 {max_rank:.0f})"),
            out_path=out_dir / f"rank_displacement_{metric}.png",
            cmap="YlOrRd", vmin=0, vmax=vmax_disp, fmt=".1f", split_at=n_real,
        )

        if not do_ci:
            continue

        plot_heatmap_ci(
            r_df, r_lo, r_hi,
            title=(f"Dataset rank correlation (Spearman r) with 95% bootstrap CI — {metric}\n"
                   f"{n_models} models  |  B={args.bootstrap_b:,} resamples"),
            out_path=out_dir / f"rank_spearman_ci_{metric}.png",
            cmap="RdBu_r", vmin=-1, vmax=1, split_at=n_real,
        )

        plot_heatmap_ci(
            disp_df, d_lo, d_hi,
            title=(f"Mean absolute rank displacement with 95% bootstrap CI — {metric}\n"
                   f"{n_models} models  |  B={args.bootstrap_b:,} resamples"),
            out_path=out_dir / f"rank_displacement_ci_{metric}.png",
            cmap="YlOrRd", vmin=0, vmax=vmax_disp, split_at=n_real,
        )

    # Write combined CSV
    if all_rows:
        out_csv    = out_dir / "rank_consistency.csv"
        fieldnames = list(all_rows[0].keys())
        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(all_rows)
        print(f"\nWrote {len(all_rows)} rows -> {out_csv.name}")


if __name__ == "__main__":
    main()
