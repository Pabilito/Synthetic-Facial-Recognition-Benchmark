"""
eval_biometric.py
─────────────────────────────────────────────────────────────────────────────
Computes ISO/IEC 19795-compliant biometric metrics from the pre-computed
cosine-similarity scores stored in the detail CSV files produced by
eval_real.py and eval_synth.py.

Metrics
───────
  EER      – Equal Error Rate: threshold where FMR ≈ FNMR (lower is better).
  FMR100   – False Match Rate at the operating point where FNMR ≤ 1/100 (1 %).
             Measures how many impostors sneak through when only 1 % of genuine
             pairs are rejected.
  FMR1000  – Same but FNMR ≤ 1/1000 (0.1 %).  Stricter operating point.
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
import csv
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np

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

    # Vectorised: shape (n_thresholds,)
    fmr  = (scores_impostor[None, :] >= thresholds[:, None]).mean(axis=1)
    fnmr = (scores_genuine [None, :] <  thresholds[:, None]).mean(axis=1)

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


def _fmr_at_fnmr(fmr: np.ndarray, fnmr: np.ndarray, fnmr_target: float) -> float:
    """
    FMR at the *highest threshold* where FNMR ≤ fnmr_target.

    As threshold rises, FNMR increases and FMR decreases.  The operating
    point of interest is the strictest threshold that still keeps the
    miss-rate within the allowed budget — i.e. the last index where
    FNMR ≤ target, which corresponds to the minimum FMR for that budget.

    If FNMR never falls to the target (even at the most permissive threshold)
    the system cannot meet the constraint; returns 1.0.
    """
    mask = fnmr <= fnmr_target
    if not mask.any():
        return 1.0
    # Last (highest-threshold) index that satisfies the constraint
    best_idx = int(np.where(mask)[0][-1])
    return float(fmr[best_idx])


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
        "FMR100":    _fmr_at_fnmr(fmr, fnmr, fnmr_target=1 / 100),
        "FMR1000":   _fmr_at_fnmr(fmr, fnmr, fnmr_target=1 / 1000),
        "ZeroFMR":   _zero_fmr(fmr, fnmr),
        "TAR@FAR1":  _tar_at_far(fmr, fnmr, far_target=0.01),
        "TAR@FAR01": _tar_at_far(fmr, fnmr, far_target=0.001),
    }


# ── CSV loading ────────────────────────────────────────────────────────────────

def load_detail_csv(path: Path,
                    model_filter:   set[str] | None = None,
                    dataset_filter: set[str] | None = None,
                    ) -> dict[tuple[str, str], tuple[list, list]]:
    """
    Read a detail CSV and return a dict:
        (model, dataset) -> (genuine_scores, impostor_scores)

    Expects columns: model, dataset, label, cos_sim
    (extra columns are silently ignored).
    """
    if not path.exists():
        return {}

    data: dict[tuple, tuple[list, list]] = defaultdict(lambda: ([], []))

    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            print(f"  [warn] Empty or unreadable CSV: {path.name}")
            return {}

        required = {"model", "dataset", "label", "cosine_sim"}
        missing  = required - set(reader.fieldnames)
        if missing:
            print(
                f"  [error] {path.name} is missing required columns: {missing}\n"
                f"          Available columns: {reader.fieldnames}\n"
                f"          Skipping this file."
            )
            return {}

        n_rows = 0
        for row in reader:
            model   = row["model"].strip()
            dataset = row["dataset"].strip()

            if model_filter   and model   not in model_filter:
                continue
            if dataset_filter and dataset not in dataset_filter:
                continue

            try:
                label   = int(row["label"])
                cos_sim = float(row["cosine_sim"])
            except (ValueError, KeyError):
                continue

            genuine_list, impostor_list = data[(model, dataset)]
            if label == 1:
                genuine_list.append(cos_sim)
            else:
                impostor_list.append(cos_sim)
            n_rows += 1

    print(f"  Loaded {n_rows:,} rows from {path.name}")
    return dict(data)


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
    """Load one detail CSV and compute metrics for every (model, dataset) pair."""
    print(f"\nReading [{split}] from {detail_path.name} ...")
    all_scores = load_detail_csv(detail_path, model_filter, dataset_filter)

    if not all_scores:
        print("  No matching data found.")
        return []

    results = []
    for (model, dataset), (genuine, impostor) in sorted(all_scores.items()):
        n_g = len(genuine)
        n_i = len(impostor)

        if n_g < 10:
            print(f"  [skip] {model} × {dataset}: only {n_g} genuine pairs (need ≥ 10)")
            continue
        if n_i < 10:
            print(f"  [skip] {model} × {dataset}: only {n_i} impostor pairs (need ≥ 10)")
            continue

        g = np.array(genuine,  dtype=np.float32)
        i = np.array(impostor, dtype=np.float32)

        m = compute_metrics(g, i)

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
