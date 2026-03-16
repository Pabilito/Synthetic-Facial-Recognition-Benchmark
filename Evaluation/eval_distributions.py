"""
eval_distributions.py
─────────────────────────────────────────────────────────────────────────────
Reads Evaluation/real_detail.csv and Evaluation/synth_detail.csv and plots
mated (label=1) vs non-mated (label=0) cosine similarity distributions in a
grid:
  - Rows    : models
  - Columns : real datasets + synthetic datasets  (separated by thick line)
  - Each panel: distributions for that specific (model, dataset) pair
  - Skips   : IJBB and IJBC datasets (too large)

Strategy:
  Each CSV is read exactly ONCE in chunks. Per chunk, rows are grouped by
  (model, dataset) and cosine_sim values are appended directly into the cache
  dict. Peak RAM ~ one chunk + accumulated cache arrays.

Delete distribution_data.npz to force recomputation.

Outputs:
  distribution_data.npz           <- cache; delete to force recomputation
  distribution_grid.png           <- full grid (all models & datasets)
  distribution_grid_focused.png   <- focused: 3 models x 7 datasets
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

# ── Config ────────────────────────────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).parent
EVAL_DIR   = SCRIPT_DIR

REAL_CSV         = EVAL_DIR / "real_detail.csv"
SYNTH_CSV        = EVAL_DIR / "synth_detail.csv"
CACHE_NPZ        = EVAL_DIR / "distribution_data.npz"
OUT_GRID         = EVAL_DIR / "distribution_grid.png"
OUT_GRID_FOCUSED = EVAL_DIR / "distribution_grid_focused.png"

SKIP_DATASETS = {"IJBB", "IJBC"}

PLOT_XMIN, PLOT_XMAX = -0.5, 1.0
BINS       = 120
KDE_POINTS = 400

COLOR_MATED     = "#2196F3"
COLOR_NON_MATED = "#F44336"
ALPHA_FILL      = 0.20
ALPHA_LINE      = 0.85

# ── Rename maps (applied after cache load, no recomputation needed) ───────────

DATASET_RENAME = {
    "Control10k": "ControlFace10k",
}

MODEL_RENAME = {
    "InceptionResnetV1_CASIA": "FaceNet_CASIA",
}

# ── Focused figure config ─────────────────────────────────────────────────────

FOCUSED_MODELS = [
    "IResNet_R18_Glint360K",
    "FaceNet_CASIA",          # renamed from InceptionResnetV1_CASIA
    "SwinFace",
]

FOCUSED_SYNTH_DATASETS = [
    "Vec2Face",
    "MorphFace",
    "SynMulti-PIE",
    "ControlFace10k",         # stored as Control10k in cache, renamed on load
]

EXCLUDE_FROM_FOCUSED_REAL = {"CALFW", "CPLFW", "IJBB", "IJBC"}
# first 3 real datasets not in the above set are used


# ── Rename helpers ────────────────────────────────────────────────────────────

def apply_renames(cache: dict, models: list,
                  real_datasets: list,
                  synth_datasets: list) -> tuple[dict, list, list, list]:
    """Rename dataset and model labels in cache keys and metadata lists."""
    new_cache = {}
    for k, v in cache.items():
        parts = k.split("|")
        if len(parts) == 4:
            parts[1] = MODEL_RENAME.get(parts[1], parts[1])
            parts[2] = DATASET_RENAME.get(parts[2], parts[2])
        new_cache["|".join(parts)] = v
    models         = [MODEL_RENAME.get(m, m)   for m in models]
    real_datasets  = [DATASET_RENAME.get(d, d) for d in real_datasets]
    synth_datasets = [DATASET_RENAME.get(d, d) for d in synth_datasets]
    return new_cache, models, real_datasets, synth_datasets


# ── Single-pass streaming into cache ─────────────────────────────────────────

def stream_into_cache(path: Path, source: str, cache: dict,
                      chunk_size: int = 500_000):
    if not path.exists():
        print(f"[ERROR] {path} not found")
        sys.exit(1)

    print(f"Streaming {path.name} [{source}] ...", flush=True)
    rows_read = 0

    for chunk in pd.read_csv(
        path,
        usecols=["model", "dataset", "cosine_sim", "label"],
        dtype={"model": "category", "dataset": "category", "label": "int8"},
        chunksize=chunk_size,
    ):
        chunk = chunk[~chunk["dataset"].isin(SKIP_DATASETS)]
        chunk["cosine_sim"] = pd.to_numeric(chunk["cosine_sim"], errors="coerce")
        chunk = chunk.dropna(subset=["cosine_sim"])

        for (model, dataset), grp in chunk.groupby(
                ["model", "dataset"], observed=True):
            for lbl in (0, 1):
                key  = f"{source}|{model}|{dataset}|{lbl}"
                vals = grp.loc[grp["label"] == lbl,
                               "cosine_sim"].values.astype("float32")
                if key in cache:
                    cache[key] = np.concatenate([cache[key], vals])
                else:
                    cache[key] = vals

        rows_read += len(chunk)
        print(f"  {rows_read:>12,} rows processed ...", flush=True)

    print(f"  -> done  ({path.name})", flush=True)


# ── Build & save cache ────────────────────────────────────────────────────────

def build_and_save_cache() -> tuple[dict, list, list, list]:
    cache = {}
    stream_into_cache(REAL_CSV,  "real",  cache)
    stream_into_cache(SYNTH_CSV, "synth", cache)

    all_models     = sorted({k.split("|")[1] for k in cache})
    real_datasets  = sorted({k.split("|")[2] for k in cache
                              if k.startswith("real|")})
    synth_datasets = sorted({k.split("|")[2] for k in cache
                              if k.startswith("synth|")})

    print(f"\n  -> {len(all_models)} models, "
          f"{len(real_datasets)} real datasets, "
          f"{len(synth_datasets)} synth datasets")

    arrays = dict(cache)
    arrays["__models__"]         = np.array(all_models,     dtype=object)
    arrays["__real_datasets__"]  = np.array(real_datasets,  dtype=object)
    arrays["__synth_datasets__"] = np.array(synth_datasets, dtype=object)
    np.savez_compressed(CACHE_NPZ, **arrays)
    print(f"Saved cache -> {CACHE_NPZ}")

    return cache, all_models, real_datasets, synth_datasets


# ── Load cache ────────────────────────────────────────────────────────────────

def load_cache() -> tuple[dict, list, list, list]:
    print(f"Cache found: {CACHE_NPZ}  (delete to recompute)", flush=True)
    npz            = np.load(CACHE_NPZ, allow_pickle=True)
    models         = list(npz["__models__"])
    real_datasets  = list(npz["__real_datasets__"])
    synth_datasets = list(npz["__synth_datasets__"])
    cache          = {k: npz[k] for k in npz.files if not k.startswith("__")}
    print(f"  -> {len(models)} models, "
          f"{len(real_datasets)} real datasets, "
          f"{len(synth_datasets)} synth datasets")
    print(f"  -> models        : {models}")
    print(f"  -> real datasets : {real_datasets}")
    print(f"  -> synth datasets: {synth_datasets}")
    return cache, models, real_datasets, synth_datasets


# ── KDE ───────────────────────────────────────────────────────────────────────

def kde_curve(values: np.ndarray):
    if len(values) < 5:
        return None, None
    kde = gaussian_kde(values, bw_method="scott")
    xs  = np.linspace(PLOT_XMIN, PLOT_XMAX, KDE_POINTS)
    return xs, kde(xs)


# ── Panels ────────────────────────────────────────────────────────────────────

def draw_panel(ax, sim_mated: np.ndarray, sim_nonmated: np.ndarray,
               col_is_synth: bool):
    """Histogram + KDE line."""
    all_vals   = np.concatenate([sim_mated, sim_nonmated])
    hist_range = (float(all_vals.min()), float(all_vals.max()))

    for sims, color in [(sim_nonmated, COLOR_NON_MATED),
                        (sim_mated,    COLOR_MATED)]:
        if len(sims) == 0:
            continue
        ax.hist(sims, bins=BINS, range=hist_range,
                density=True, color=color, alpha=ALPHA_FILL, linewidth=0)
        xs, ys = kde_curve(sims)
        if xs is not None:
            ax.plot(xs, ys, color=color, linewidth=1.4, alpha=ALPHA_LINE)

    ax.set_xlim(PLOT_XMIN, PLOT_XMAX)
    ax.tick_params(labelsize=6)
    ax.grid(True, linestyle=":", alpha=0.35)
    if col_is_synth:
        ax.set_facecolor("#f5f0ff")


def draw_panel_kde_only(ax, sim_mated: np.ndarray, sim_nonmated: np.ndarray,
                        col_is_synth: bool):
    """KDE fill + line only — no histogram bars."""
    for sims, color in [(sim_nonmated, COLOR_NON_MATED),
                        (sim_mated,    COLOR_MATED)]:
        if len(sims) == 0:
            continue
        xs, ys = kde_curve(sims)
        if xs is None:
            continue
        ax.fill_between(xs, ys, alpha=ALPHA_FILL, color=color, linewidth=0)
        ax.plot(xs, ys, color=color, linewidth=1.6, alpha=ALPHA_LINE)

    ax.set_xlim(PLOT_XMIN, PLOT_XMAX)
    ax.tick_params(labelsize=7)
    ax.grid(True, linestyle=":", alpha=0.35)
    if col_is_synth:
        ax.set_facecolor("#f5f0ff")


def blank_panel(ax, reason="no data"):
    ax.set_facecolor("#f0f0f0")
    ax.text(0.5, 0.5, reason, ha="center", va="center",
            transform=ax.transAxes, fontsize=7, color="grey", style="italic")
    ax.set_xticks([])
    ax.set_yticks([])


# ── Y-axis standardisation ────────────────────────────────────────────────────

def standardise_yaxis(axes_row: list):
    """
    Set a common y-limit for all panels in a row and format all y-tick
    labels to the same number of decimal places.
    Blank panels (no yticks) are skipped.
    """
    # collect max y across the row (ignore blank panels)
    ymaxes = []
    for ax in axes_row:
        if ax.get_yticks().size > 0 and ax.get_ylim()[1] > 0:
            ymaxes.append(ax.get_ylim()[1])
    if not ymaxes:
        return
    ymax = max(ymaxes)

    # choose decimal places: enough to show at least 3 significant ticks
    for dp in (1, 2, 3):
        step = round(ymax / 4, dp)
        if step > 0:
            break

    formatter = mticker.FormatStrFormatter(f"%.{dp}f")

    for ax in axes_row:
        if ax.get_ylim()[1] == 0:
            continue
        ax.set_ylim(0, ymax * 1.05)
        ax.yaxis.set_major_formatter(formatter)


# ── Legend ────────────────────────────────────────────────────────────────────

def make_legend_handles():
    return [
        mpatches.Patch(color=COLOR_MATED,     alpha=0.7, label="Mated"),
        mpatches.Patch(color=COLOR_NON_MATED, alpha=0.7, label="Non-mated"),
    ]


# ── Shared grid drawing ───────────────────────────────────────────────────────

def _draw_grid(axes, cache, models, col_datasets, n_real_cols,
               panel_fn, n_rows, n_cols):

    # first pass: draw all panels
    for r_idx, model in enumerate(models):
        for c_idx, col_ds in enumerate(col_datasets):

            ax           = axes[r_idx][c_idx]
            col_is_synth = c_idx >= n_real_cols
            source       = "synth" if col_is_synth else "real"

            if r_idx == 0:
                ax.set_title(col_ds, fontsize=8, fontweight="bold", pad=3)
            if c_idx == 0:
                ax.set_ylabel(model, fontsize=8, fontweight="bold",
                              rotation=90, labelpad=4)
            if r_idx == n_rows - 1:
                ax.set_xlabel("Cosine Similarity", fontsize=7)
            else:
                ax.set_xticklabels([])

            sim_mated    = cache.get(f"{source}|{model}|{col_ds}|1", np.array([]))
            sim_nonmated = cache.get(f"{source}|{model}|{col_ds}|0", np.array([]))

            if len(sim_mated) + len(sim_nonmated) == 0:
                blank_panel(ax)
            else:
                panel_fn(ax, sim_mated, sim_nonmated, col_is_synth)

    # second pass: standardise y-axis per row, then hide non-leftmost tick labels
    for r_idx in range(n_rows):
        row_axes = [axes[r_idx][c] for c in range(n_cols)]
        standardise_yaxis(row_axes)
        for c_idx, ax in enumerate(row_axes):
            if c_idx != 0:
                ax.set_yticklabels([])

    # thick separator between real and synth columns
    if 0 < n_real_cols < n_cols:
        for r_idx in range(n_rows):
            axes[r_idx][n_real_cols - 1].spines["right"].set(
                linewidth=2.5, color="black")
            axes[r_idx][n_real_cols    ].spines["left" ].set(
                linewidth=2.5, color="black")


# ── Full grid ─────────────────────────────────────────────────────────────────

def plot_grid(cache: dict, models: list,
              real_datasets: list, synth_datasets: list):

    col_datasets = real_datasets + synth_datasets
    n_rows       = len(models)
    n_cols       = len(col_datasets)
    n_real_cols  = len(real_datasets)

    print(f"Plotting full grid: {n_rows} models x {n_cols} datasets ...",
          flush=True)

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(3.2 * n_cols, 2.8 * n_rows),
                             squeeze=False)
    fig.suptitle(
        "Mated vs Non-Mated Cosine Similarity Distributions\n"
        "(rows = models,  columns = evaluation datasets  |  real -- synth)",
        fontsize=12, fontweight="bold", y=1.01,
    )

    _draw_grid(axes, cache, models, col_datasets, n_real_cols,
               draw_panel, n_rows, n_cols)

    fig.legend(handles=make_legend_handles(), loc="lower center", ncol=2,
               fontsize=9, bbox_to_anchor=(0.5, -0.02), frameon=True)
    fig.tight_layout()
    fig.savefig(OUT_GRID, dpi=300, bbox_inches="tight")
    print(f"Saved full grid   -> {OUT_GRID}")


# ── Focused grid ──────────────────────────────────────────────────────────────

def plot_grid_focused(cache: dict, real_datasets: list, synth_datasets: list):

    focused_real   = [d for d in real_datasets
                      if d not in EXCLUDE_FROM_FOCUSED_REAL][:3]
    focused_synth  = [d for d in FOCUSED_SYNTH_DATASETS if d in synth_datasets]
    focused_models = list(FOCUSED_MODELS)

    col_datasets = focused_real + focused_synth
    n_rows       = len(focused_models)
    n_cols       = len(col_datasets)
    n_real_cols  = len(focused_real)

    print(f"Plotting focused grid: {n_rows} models x {n_cols} datasets ...",
          flush=True)
    print(f"  models : {focused_models}")
    print(f"  real   : {focused_real}")
    print(f"  synth  : {focused_synth}")

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(3.0 * n_cols, 2.6 * n_rows),
                             squeeze=False)
    fig.suptitle(
        "Mated vs Non-Mated Cosine Similarity Distributions (focused)\n"
        "(rows = models,  columns = evaluation datasets  |  real -- synth)",
        fontsize=11, fontweight="bold", y=1.01,
    )

    _draw_grid(axes, cache, focused_models, col_datasets, n_real_cols,
               draw_panel_kde_only, n_rows, n_cols)

    fig.legend(handles=make_legend_handles(), loc="lower center", ncol=2,
               fontsize=9, bbox_to_anchor=(0.5, -0.02), frameon=True)
    fig.tight_layout()
    fig.savefig(OUT_GRID_FOCUSED, dpi=300, bbox_inches="tight")
    print(f"Saved focused grid -> {OUT_GRID_FOCUSED}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():

    if CACHE_NPZ.exists():
        cache, models, real_datasets, synth_datasets = load_cache()
    else:
        print("=== Building cache (single pass per file) ===")
        cache, models, real_datasets, synth_datasets = build_and_save_cache()

    # apply renames (no cache rebuild needed)
    cache, models, real_datasets, synth_datasets = apply_renames(
        cache, models, real_datasets, synth_datasets)

    if OUT_GRID.exists():
        print(f"\n=== Skipping full grid (already exists: {OUT_GRID.name}) ===")
    else:
        print("\n=== Plotting full grid ===")
        plot_grid(cache, models, real_datasets, synth_datasets)

    if OUT_GRID_FOCUSED.exists():
        print(f"=== Skipping focused grid (already exists: {OUT_GRID_FOCUSED.name}) ===")
    else:
        print("\n=== Plotting focused grid ===")
        plot_grid_focused(cache, real_datasets, synth_datasets)


if __name__ == "__main__":
    main()
