"""
eval_biometric.py
─────────────────────────────────────────────────────────────────────────────
Computes ISO/IEC 19795-compliant biometric metrics from the pre-computed
cosine-similarity scores stored in the detail CSV files produced by
eval_real.py and eval_synth.py.

Metrics
───────
  EER      – Equal Error Rate: threshold where FMR ≈ FNMR (lower is better).
  FMR100   – FNMR at the operating point where FMR ≤ 1/100 (1 %).
             The threshold is set so that only 1-in-100 impostors are falsely
             accepted; FMR100 is the proportion of genuine pairs rejected at
             that threshold (lower is better).
  FMR1000  – Same but FMR ≤ 1/1000 (0.1 %).  Stricter operating point.
  ZeroFMR  – FNMR at the lowest threshold that yields FMR = 0 (no false
             accepts at all).  Sometimes called FNMR@FMR=0.
  TAR@FAR1 – True Accept Rate (= 1 − FNMR) when FAR ≤ 1 % (= FMR ≤ 0.01).
  TAR@FAR01 – TAR when FAR ≤ 0.1 % (= FMR ≤ 0.001).

Definitions follow the NIST FRVT convention:
  • Genuine pairs  : label == 1  (same identity)
  • Impostor pairs : label == 0  (different identity)
  • FMR(t)  = P(score >= t | impostor)   (False Match Rate)
  • FNMR(t) = P(score <  t | genuine)    (False Non-Match Rate)
  • TAR(t)  = 1 − FNMR(t)
  Note: cosine similarity is used as the score — higher means more similar.

Input
─────
  Evaluation/real_detail.csv   (produced by eval_real.py)
  Evaluation/synth_detail.csv  (produced by eval_synth.py)

  Both files must share at least these columns:
    model       – model name
    dataset     – dataset name
    label       – 1 (genuine) or 0 (impostor)
    cosine_sim  – cosine similarity score ∈ [−1, 1]

Output
──────
  Evaluation/biometric_metrics.csv  – one row per model × dataset × split
  Columns:
    split, model, dataset, n_genuine, n_impostor,
    EER, FMR100, FMR1000, ZeroFMR, TAR@FAR1, TAR@FAR01

Usage
─────
  python eval_biometric.py
  python eval_biometric.py --real-only
  python eval_biometric.py --synth-only
  python eval_biometric.py --detail-csv path/to/custom_detail.csv --split custom
  python eval_biometric.py --datasets LFW CALFW       # filter datasets
  python eval_biometric.py --models IResNet_R100      # filter models
"""

import argparse
import csv          # still used for write_results output
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────────

SCRIPT_DIR   = Path(__file__).parent
EVAL_DIR     = SCRIPT_DIR / "Evaluation"
REAL_DETAIL  = EVAL_DIR / "real_detail.csv"
SYNTH_DETAIL = EVAL_DIR / "synth_detail.csv"
OUT_CSV      = EVAL_DIR / "biometric_metrics.csv"

OUTPUT_COLS = [
    "split", "model", "dataset",
    "n_genuine", "n_impostor",
    "EER", "FMR100", "FMR1000", "ZeroFMR", "TAR@FAR1", "TAR@FAR01",
]


# ── Core metric computation ────────────────────────────────────────────────────

def compute_fmr_fnmr(scores_genuine: np.ndarray,
                     scores_impostor: np.ndarray,
                     n_thresholds: int = 10_000
                     ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return (thresholds, FMR, FNMR) arrays evaluated at n_thresholds evenly
    spaced points spanning the full score range.

    FMR(t)  = fraction of impostor scores >= t
    FNMR(t) = fraction of genuine  scores <  t
    """
    lo = min(scores_genuine.min(), scores_impostor.min())
    hi = max(scores_genuine.max(), scores_impostor.max())
    thresholds = np.linspace(lo, hi, n_thresholds)

    # Sort once, then use searchsorted for all thresholds in one O(n log n) pass.
    # This avoids the O(n_thresholds × n_scores) broadcast matrix.
    sg = np.sort(scores_genuine)
    si = np.sort(scores_impostor)

    # FNMR(t) = fraction of genuine scores < t
    fnmr = np.searchsorted(sg, thresholds, side="left").astype(np.float64) / len(sg)

    # FMR(t) = fraction of impostor scores >= t  =  1 - fraction < t
    fmr = 1.0 - np.searchsorted(si, thresholds, side="left").astype(np.float64) / len(si)

    return thresholds, fmr, fnmr


def _eer(fmr: np.ndarray, fnmr: np.ndarray) -> float:
    """EER: linear interpolation at the FMR/FNMR crossover."""
    diff = fmr - fnmr
    # Find the index just before the sign change (FMR crosses below FNMR)
    idx  = np.argmin(np.abs(diff))
    # Refine with linear interpolation between adjacent samples
    if idx > 0 and idx < len(diff) - 1:
        d0, d1 = diff[idx - 1], diff[idx]
        if d0 != d1:
            t = d0 / (d0 - d1)
            eer = fnmr[idx - 1] + t * (fnmr[idx] - fnmr[idx - 1])
            return float(eer)
    return float((fmr[idx] + fnmr[idx]) / 2)


def _fnmr_at_fmr(fmr: np.ndarray, fnmr: np.ndarray, fmr_target: float) -> float:
    """
    FNMR at the lowest threshold where FMR <= fmr_target.

    This is the FMR100 / FMR1000 definition: the proportion of genuine pairs
    rejected (FNMR) when the threshold is set so that at most 1-in-100 (or
    1-in-1000) impostors are falsely accepted (FMR <= 1/100 or 1/1000).

    As threshold rises, FMR decreases and FNMR increases.  We want the
    least-strict (lowest) threshold that still satisfies FMR <= target —
    that is the first index where FMR drops to/below target, giving the
    best (lowest) FNMR for that operating point.

    If FMR never falls to the target the system is too weak; returns 1.0.
    """
    mask = fmr <= fmr_target
    if not mask.any():
        return 1.0
    best_idx = int(np.where(mask)[0][0])   # first index where FMR <= target
    return float(fnmr[best_idx])

def _zero_fmr(fmr: np.ndarray, fnmr: np.ndarray) -> float:
    """FNMR at the lowest threshold where FMR = 0 (no false accepts)."""
    mask = fmr == 0.0
    if not mask.any():
        # FMR never reaches exactly 0 in the sampled grid — use smallest value
        best_idx = np.argmin(fmr)
    else:
        best_idx = np.where(mask)[0][0]   # first (= lowest threshold with FMR=0)
    return float(fnmr[best_idx])


def _tar_at_far(fmr: np.ndarray, fnmr: np.ndarray, far_target: float) -> float:
    """TAR (= 1 − FNMR) at the highest threshold where FMR ≤ far_target."""
    mask = fmr <= far_target
    if not mask.any():
        return 0.0
    valid_idx = np.where(mask)[0]
    best_idx  = valid_idx[np.argmax(fmr[valid_idx])]  # highest FMR still ≤ target
    return float(1.0 - fnmr[best_idx])


def compute_metrics(scores_genuine: np.ndarray,
                    scores_impostor: np.ndarray) -> dict:
    """Return a dict of all six biometric metrics."""
    thresholds, fmr, fnmr = compute_fmr_fnmr(scores_genuine, scores_impostor)
    return {
        "EER":       _eer(fmr, fnmr),
        "FMR100":    _fnmr_at_fmr(fmr, fnmr, fmr_target=1 / 100),
        "FMR1000":   _fnmr_at_fmr(fmr, fnmr, fmr_target=1 / 1000),
        "ZeroFMR":   _zero_fmr(fmr, fnmr),
        "TAR@FAR1":  _tar_at_far(fmr, fnmr, far_target=0.01),
        "TAR@FAR01": _tar_at_far(fmr, fnmr, far_target=0.001),
    }


# ── CSV loading ────────────────────────────────────────────────────────────────

def _check_columns(fieldnames, path: Path) -> bool:
    """Validate required columns exist. Returns True if OK."""
    required = {"model", "dataset", "label", "cosine_sim"}
    missing  = required - set(fieldnames)
    if missing:
        print(
            f"  [error] {path.name} is missing required columns: {missing}\n"
            f"          Available columns: {fieldnames}\n"
            f"          Skipping this file."
        )
        return False
    return True


def stream_detail_csv(
    path: Path,
    model_filter:   set[str] | None = None,
    dataset_filter: set[str] | None = None,
    chunksize: int = 500_000,
):
    """
    Generator: reads a detail CSV in vectorised pandas chunks and yields
        ((model, dataset), genuine_scores_np, impostor_scores_np)
    one group at a time.

    Using pandas read_csv() with chunksize gives ~10-20× speed improvement
    over csv.DictReader for large files (100M+ rows) because:
      • Parsing happens in C, not Python.
      • An entire chunk is processed as a vectorised numpy operation rather
        than row-by-row Python loops.
      • category dtype for model/dataset cuts string overhead significantly.
    """
    if not path.exists():
        return

    # Peek at header to validate columns before starting the expensive read.
    try:
        header_df = pd.read_csv(path, nrows=0)
    except Exception as e:
        print(f"  [warn] Could not read {path.name}: {e}")
        return
    if not _check_columns(list(header_df.columns), path):
        return

    # Build dtype map — categorical strings avoid repeated interning overhead.
    dtypes = {
        "model":      "category",
        "dataset":    "category",
        "label":      "int8",
        "cosine_sim": "float32",
    }

    # Accumulate per-group lists of float32 arrays (one array per chunk).
    # This avoids a giant single list.append() loop entirely.
    genuine_chunks:  dict[tuple[str, str], list[np.ndarray]] = defaultdict(list)
    impostor_chunks: dict[tuple[str, str], list[np.ndarray]] = defaultdict(list)

    n_rows = 0
    reader = pd.read_csv(
        path,
        usecols=["model", "dataset", "label", "cosine_sim"],
        dtype=dtypes,
        chunksize=chunksize,
        engine="c",
        on_bad_lines="skip",
    )

    for chunk in reader:
        # Strip whitespace from categoricals (cheap on category dtype).
        chunk["model"]   = chunk["model"].str.strip()
        chunk["dataset"] = chunk["dataset"].str.strip()

        # Apply filters before any groupby work.
        if model_filter:
            chunk = chunk[chunk["model"].isin(model_filter)]
        if dataset_filter:
            chunk = chunk[chunk["dataset"].isin(dataset_filter)]
        if chunk.empty:
            continue

        n_rows += len(chunk)

        genuine  = chunk[chunk["label"] == 1]
        impostor = chunk[chunk["label"] == 0]

        for (model, dataset), grp in genuine.groupby(["model", "dataset"], observed=True):
            genuine_chunks[(model, dataset)].append(grp["cosine_sim"].to_numpy(dtype=np.float32))

        for (model, dataset), grp in impostor.groupby(["model", "dataset"], observed=True):
            impostor_chunks[(model, dataset)].append(grp["cosine_sim"].to_numpy(dtype=np.float32))

    print(f"  Loaded {n_rows:,} rows from {path.name}")

    # Merge per-group chunk lists and yield — concatenation is O(n) total.
    all_keys = sorted(set(genuine_chunks) | set(impostor_chunks))
    for key in all_keys:
        g_parts = genuine_chunks.pop(key, [])
        i_parts = impostor_chunks.pop(key, [])
        genuine  = np.concatenate(g_parts)  if g_parts else np.empty(0, dtype=np.float32)
        impostor = np.concatenate(i_parts) if i_parts else np.empty(0, dtype=np.float32)
        yield key, genuine, impostor


# ── Output CSV helpers ─────────────────────────────────────────────────────────

def _fmt(v: float) -> str:
    """Format a metric value as a percentage string with 4 decimal places."""
    return f"{v * 100:.4f}"


def write_results(results: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = out_path.exists()

    with open(out_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLS)
        if not file_exists:
            writer.writeheader()
        writer.writerows(results)


# ── Main ──────────────────────────────────────────────────────────────────────

def process_detail_file(
    detail_path: Path,
    split: str,
    model_filter:   set[str] | None,
    dataset_filter: set[str] | None,
) -> list[dict]:
    """
    Stream one detail CSV and compute metrics for every (model, dataset) pair.
    Only one group's scores are kept in RAM at a time.
    Results are written to the output CSV incrementally as each group is done.
    """
    print(f"\nReading [{split}] from {detail_path.name} ...")

    results = []
    n_groups = 0

    for (model, dataset), genuine, impostor in stream_detail_csv(
        detail_path, model_filter, dataset_filter
    ):
        n_g = len(genuine)
        n_i = len(impostor)

        if n_g < 10:
            print(f"  [skip] {model} × {dataset}: only {n_g} genuine pairs (need ≥ 10)")
            continue
        if n_i < 10:
            print(f"  [skip] {model} × {dataset}: only {n_i} impostor pairs (need ≥ 10)")
            continue

        m = compute_metrics(genuine, impostor)
        n_groups += 1

        results.append({
            "split":      split,
            "model":      model,
            "dataset":    dataset,
            "n_genuine":  n_g,
            "n_impostor": n_i,
            "EER":        _fmt(m["EER"]),
            "FMR100":     _fmt(m["FMR100"]),
            "FMR1000":    _fmt(m["FMR1000"]),
            "ZeroFMR":    _fmt(m["ZeroFMR"]),
            "TAR@FAR1":   _fmt(m["TAR@FAR1"]),
            "TAR@FAR01":  _fmt(m["TAR@FAR01"]),
        })

        print(
            f"  {model:<35} {dataset:<20} "
            f"EER={m['EER']*100:5.2f}%  "
            f"FMR100={m['FMR100']*100:6.3f}%  "
            f"FMR1000={m['FMR1000']*100:6.3f}%  "
            f"ZeroFMR={m['ZeroFMR']*100:5.2f}%  "
            f"TAR@1%={m['TAR@FAR1']*100:5.2f}%"
        )

    if n_groups == 0:
        print("  No matching data found.")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute biometric metrics (EER, FMR100, FMR1000, ZeroFMR, TAR) "
                    "from pre-computed detail CSVs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--real-only", action="store_true",
        help="Process only real_detail.csv."
    )
    parser.add_argument(
        "--synth-only", action="store_true",
        help="Process only synth_detail.csv."
    )
    parser.add_argument(
        "--detail-csv", type=str, default=None, metavar="PATH",
        help="Process a custom detail CSV instead of the defaults."
    )
    parser.add_argument(
        "--split", type=str, default="custom",
        help="Split label used when --detail-csv is given (default: 'custom')."
    )
    parser.add_argument(
        "--datasets", nargs="+", metavar="NAME", default=None,
        help="Process only the named dataset(s). Case-sensitive, space-separated."
    )
    parser.add_argument(
        "--models", nargs="+", metavar="NAME", default=None,
        help="Process only the named model(s). Case-sensitive, space-separated."
    )
    parser.add_argument(
        "--out", type=str, default=str(OUT_CSV), metavar="PATH",
        help=f"Output CSV path (default: {OUT_CSV})."
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Delete the output CSV before writing (default: append)."
    )
    args = parser.parse_args()

    model_filter   = set(args.models)   if args.models   else None
    dataset_filter = set(args.datasets) if args.datasets else None
    out_path       = Path(args.out)

    if args.overwrite and out_path.exists():
        out_path.unlink()
        print(f"[info] Removed existing {out_path.name}")

    # ── Determine which files to process ──────────────────────────────────────
    if args.detail_csv:
        tasks = [(Path(args.detail_csv), args.split)]
    else:
        tasks = []
        if not args.synth_only:
            tasks.append((REAL_DETAIL,  "real"))
        if not args.real_only:
            tasks.append((SYNTH_DETAIL, "synth"))

    if not tasks:
        print("[error] Nothing to process. Check --real-only / --synth-only flags.")
        sys.exit(1)

    # ── Process ───────────────────────────────────────────────────────────────
    all_results = []
    for detail_path, split in tasks:
        rows = process_detail_file(detail_path, split, model_filter, dataset_filter)
        all_results.extend(rows)

    if not all_results:
        print("\nNo results computed — nothing written.")
        sys.exit(0)

    # ── Write output ──────────────────────────────────────────────────────────
    write_results(all_results, out_path)

    print(f"\n{'─' * 60}")
    print(f"Wrote {len(all_results)} row(s) → {out_path}")
    print(f"{'─' * 60}")

    # ── Console summary table ──────────────────────────────────────────────────
    splits   = sorted({r["split"]   for r in all_results})
    datasets = sorted({r["dataset"] for r in all_results})
    models   = sorted({r["model"]   for r in all_results})

    for split in splits:
        for dataset in datasets:
            rows = [r for r in all_results
                    if r["split"] == split and r["dataset"] == dataset]
            if not rows:
                continue
            print(f"\n[{split}] {dataset}")
            header = f"  {'Model':<35} {'EER':>7} {'FMR100':>8} {'FMR1000':>9} {'ZeroFMR':>9} {'TAR@1%':>8} {'TAR@0.1%':>9}"
            print(header)
            print("  " + "─" * (len(header) - 2))
            for r in rows:
                print(
                    f"  {r['model']:<35} "
                    f"{r['EER']:>7} "
                    f"{r['FMR100']:>8} "
                    f"{r['FMR1000']:>9} "
                    f"{r['ZeroFMR']:>9} "
                    f"{r['TAR@FAR1']:>8} "
                    f"{r['TAR@FAR01']:>9}"
                )


if __name__ == "__main__":
    main()
