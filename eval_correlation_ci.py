"""
eval_correlation_ci.py
─────────────────────────────────────────────────────────────────────────────
Extends eval_correlation.py with 95% confidence intervals for every
Pearson r value, using Fisher's z-transformation (exact) and bootstrap
(non-parametric).  Both are reported; bootstrap CIs are also shown as
error bars on a forest plot.

Method
──────
  Fisher z-transformation (parametric)
      z  = arctanh(r)
      SE = 1 / sqrt(n - 3)
      95% CI in z-space: z ± 1.96 * SE
      Back-transform via tanh to get [r_lo, r_hi]
      Valid when n ≥ 4 and the underlying relationship is bivariate normal.

  Bootstrap (non-parametric, B = 10 000 resamples)
      Resample (x_i, y_i) pairs with replacement, compute r each time.
      95% CI = [2.5th percentile, 97.5th percentile] of the bootstrap dist.
      No normality assumption; recommended when n is small or non-normal.

Outputs (written to Evaluation/)
──────────────────────────────────
  correlation_ci.csv          – r, Fisher CI, bootstrap CI for every pair
  correlation_ci_heatmap.png  – heatmap annotated with r [lo, hi]
  correlation_ci_forest.png   – forest plot of all r ± bootstrap CI
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
EVAL_DIR   = SCRIPT_DIR / "Evaluation"

REAL_CSV   = EVAL_DIR / "real_summary.csv"
SYNTH_CSV  = EVAL_DIR / "synth_summary.csv"

OUT_CSV     = EVAL_DIR / "correlation_ci.csv"
OUT_HEATMAP = EVAL_DIR / "correlation_ci_heatmap.png"
OUT_FOREST  = EVAL_DIR / "correlation_ci_forest.png"

BOOTSTRAP_B = 10_000
RNG_SEED    = 42


# ── Load ──────────────────────────────────────────────────────────────────────

def load(path: Path, label: str) -> pd.DataFrame:
    if not path.exists():
        print(f"[ERROR] {label} not found: {path}")
        sys.exit(1)
    df = pd.read_csv(path)
    df["accuracy"] = pd.to_numeric(df["accuracy"], errors="coerce")
    df["std"]      = pd.to_numeric(df["std"],      errors="coerce")
    return df


# ── CI methods ────────────────────────────────────────────────────────────────

def fisher_ci(r: float, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """
    95% CI for Pearson r via Fisher z-transformation.
    Returns (r_lo, r_hi) clipped to [-1, 1].
    Requires n >= 4; returns (nan, nan) otherwise.
    """
    if n < 4 or not np.isfinite(r):
        return np.nan, np.nan
    z    = np.arctanh(np.clip(r, -0.9999, 0.9999))
    se   = 1.0 / np.sqrt(n - 3)
    z_crit = stats.norm.ppf(1 - alpha / 2)          # 1.96 for 95%
    lo   = np.tanh(z - z_crit * se)
    hi   = np.tanh(z + z_crit * se)
    return float(np.clip(lo, -1, 1)), float(np.clip(hi, -1, 1))


def bootstrap_ci(
    x: np.ndarray,
    y: np.ndarray,
    B: int = BOOTSTRAP_B,
    seed: int = RNG_SEED,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """
    Non-parametric bootstrap 95% CI for Pearson r.
    Returns (r_lo, r_hi) or (nan, nan) when n < 4.
    """
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
        # Avoid degenerate resamples (zero variance)
        if xs.std() == 0 or ys.std() == 0:
            rs[i] = np.nan
        else:
            rs[i] = np.corrcoef(xs, ys)[0, 1]

    rs = rs[np.isfinite(rs)]
    if len(rs) == 0:
        return np.nan, np.nan

    lo = float(np.percentile(rs, 100 * alpha / 2))
    hi = float(np.percentile(rs, 100 * (1 - alpha / 2)))
    return lo, hi


# ── Compute all pairwise CIs ──────────────────────────────────────────────────

def compute_ci_table(
    all_df: pd.DataFrame,
    datasets: list[str],
) -> pd.DataFrame:
    """
    For every ordered dataset pair compute r, Fisher CI, and bootstrap CI.
    Returns a tidy DataFrame with one row per pair.
    """
    rows = []
    total = len(datasets) ** 2
    done  = 0

    for ds_x in datasets:
        x_sub = (
            all_df[all_df["dataset"] == ds_x][["model", "accuracy"]]
            .rename(columns={"accuracy": "acc_x"})
        )
        for ds_y in datasets:
            y_sub = (
                all_df[all_df["dataset"] == ds_y][["model", "accuracy"]]
                .rename(columns={"accuracy": "acc_y"})
            )
            merged = x_sub.merge(y_sub, on="model", how="inner").dropna()
            n = len(merged)

            if n >= 3:
                r, p = stats.pearsonr(merged["acc_x"], merged["acc_y"])
            else:
                r, p = np.nan, np.nan

            f_lo, f_hi = fisher_ci(r, n)
            b_lo, b_hi = bootstrap_ci(
                merged["acc_x"].values,
                merged["acc_y"].values,
            )

            done += 1
            print(f"  [{done}/{total}] {ds_x} ↔ {ds_y}  "
                  f"r={r:.3f}  Fisher=[{f_lo:.3f},{f_hi:.3f}]  "
                  f"Boot=[{b_lo:.3f},{b_hi:.3f}]  n={n}")

            rows.append({
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


# ── Heatmap annotated with CI ─────────────────────────────────────────────────

def plot_ci_heatmap(ci_df: pd.DataFrame, datasets: list[str], split_index: int) -> None:
    """
    Heatmap where each cell shows  r\n[lo, hi]  using the bootstrap CI.
    """
    r_mat = (
        ci_df.pivot(index="dataset_x", columns="dataset_y", values="r")
        .reindex(index=datasets, columns=datasets)
    )
    lo_mat = (
        ci_df.pivot(index="dataset_x", columns="dataset_y", values="bootstrap_ci_lo")
        .reindex(index=datasets, columns=datasets)
    )
    hi_mat = (
        ci_df.pivot(index="dataset_x", columns="dataset_y", values="bootstrap_ci_hi")
        .reindex(index=datasets, columns=datasets)
    )

    n = len(datasets)
    fig, ax = plt.subplots(figsize=(max(8, n * 1.4), max(7, n * 1.2)))

    im = ax.imshow(r_mat.values.astype(float), vmin=-1, vmax=1, cmap="coolwarm")

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(datasets, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(datasets, fontsize=8)
    ax.set_title(
        "Dataset Correlation Heatmap — Pearson r with 95% Bootstrap CI",
        fontsize=11,
    )

    for i in range(n):
        for j in range(n):
            r_val  = r_mat.iloc[i, j]
            lo_val = lo_mat.iloc[i, j]
            hi_val = hi_mat.iloc[i, j]
            if np.isfinite(r_val):
                cell_text = f"{r_val:.2f}\n[{lo_val:.2f}, {hi_val:.2f}]"
                # White text on dark cells, black on light
                brightness = abs(r_val)
                txt_color  = "white" if brightness > 0.55 else "black"
                ax.text(j, i, cell_text, ha="center", va="center",
                        fontsize=6.5, color=txt_color, linespacing=1.4)

    # Separator between real and synthetic
    ax.axhline(split_index - 0.5, color="black", linewidth=1.8)
    ax.axvline(split_index - 0.5, color="black", linewidth=1.8)

    fig.colorbar(im, ax=ax, label="Pearson r", shrink=0.8)
    fig.tight_layout()
    fig.savefig(OUT_HEATMAP, dpi=300, bbox_inches="tight")
    print(f"Saved CI heatmap → {OUT_HEATMAP}")
    plt.close(fig)


# ── Forest plot ───────────────────────────────────────────────────────────────

def plot_forest(ci_df: pd.DataFrame) -> None:
    """
    Horizontal forest plot: one row per unique unordered dataset pair,
    showing r with 95% bootstrap CI as error bars.
    Pairs are sorted by r descending.
    """
    # Keep only upper triangle (ds_x < ds_y alphabetically) to avoid duplicates
    sub = ci_df[ci_df["dataset_x"] < ci_df["dataset_y"]].copy()
    sub = sub.dropna(subset=["r", "bootstrap_ci_lo", "bootstrap_ci_hi"])
    sub = sub.sort_values("r", ascending=True).reset_index(drop=True)

    if sub.empty:
        print("[WARN] No valid pairs for forest plot.")
        return

    labels  = sub["dataset_x"] + " ↔ " + sub["dataset_y"]
    r_vals  = sub["r"].values
    err_lo  = r_vals - sub["bootstrap_ci_lo"].values
    err_hi  = sub["bootstrap_ci_hi"].values - r_vals
    n_vals  = sub["n_models"].values

    fig_h = max(5, len(sub) * 0.45 + 2)
    fig, ax = plt.subplots(figsize=(8, fig_h))

    y_pos = np.arange(len(sub))

    # Colour by sign of r
    colors = ["#d73027" if r < 0 else "#1a9850" for r in r_vals]

    ax.errorbar(
        r_vals, y_pos,
        xerr=[err_lo, err_hi],
        fmt="o",
        color="black",
        elinewidth=1.2,
        capsize=4,
        markersize=0,        # hide default marker; use scatter for colours
        zorder=2,
    )
    ax.scatter(r_vals, y_pos, color=colors, s=40, zorder=3)

    # Annotate with n
    for i, (r, n) in enumerate(zip(r_vals, n_vals)):
        ax.text(
            1.02, y_pos[i], f"n={n}",
            va="center", ha="left",
            fontsize=7.5, transform=ax.get_yaxis_transform(),
        )

    ax.axvline(0, color="grey", linewidth=0.8, linestyle="--")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Pearson r  (95% bootstrap CI)", fontsize=10)
    ax.set_title("Pairwise Dataset Correlations — Forest Plot", fontsize=11)
    ax.set_xlim(-1.05, 1.15)
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
    ax.grid(axis="x", linestyle=":", alpha=0.4)

    fig.tight_layout()
    fig.savefig(OUT_FOREST, dpi=300, bbox_inches="tight")
    print(f"Saved forest plot → {OUT_FOREST}")
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:

    real_df  = load(REAL_CSV,  "real_summary.csv")
    synth_df = load(SYNTH_CSV, "synth_summary.csv")

    real_df["source"]  = "real"
    synth_df["source"] = "synthetic"

    all_df = pd.concat([real_df, synth_df], ignore_index=True)

    real_datasets  = sorted(real_df["dataset"].unique())
    synth_datasets = sorted(synth_df["dataset"].unique())
    datasets       = real_datasets + synth_datasets

    print(f"Real datasets ({len(real_datasets)}):      {real_datasets}")
    print(f"Synthetic datasets ({len(synth_datasets)}): {synth_datasets}")
    print(f"Models: {sorted(all_df['model'].unique())}\n")
    print(f"Computing CIs for {len(datasets)**2} pairs "
          f"(bootstrap B={BOOTSTRAP_B:,}) …\n")

    ci_df = compute_ci_table(all_df, datasets)

    ci_df.to_csv(OUT_CSV, index=False, float_format="%.6f")
    print(f"\nSaved CI table → {OUT_CSV}")

    plot_ci_heatmap(ci_df, datasets, split_index=len(real_datasets))
    plot_forest(ci_df)


if __name__ == "__main__":
    main()
