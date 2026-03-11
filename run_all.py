"""
run_all.py
─────────────────────────────────────────────────────────────────────────────
Master evaluation script with parallel model inference.

PIPELINE
────────
  Stage 1 — Parallel inference  (default: 4 models simultaneously on one GPU)
               All models are evaluated concurrently.  Each worker runs
               eval_real + eval_ijb + eval_synth for its assigned model and
               writes to an isolated temp dir (Evaluation/tmp/<model_name>/).
               After all workers finish, results are merged into the final
               Evaluation/ CSVs.

  Stage 2 — eval_biometric.py          EER / FMR100 / FMR1000 / ZeroFMR / TAR
  Stage 3 — eval_biometric_correlation.py   Pearson r + 95% CI
  Stage 4 — demographic_analysis.py         age / gender
  Stage 5 — eval_rank_consistency.py        Spearman r + rank displacement

PARALLELISM
───────────
  --parallel N   How many models to run simultaneously (default: 4).
                 All workers share the same GPU.  With 32 GB VRAM and 3 GB
                 models, N=4 fits comfortably.  Set N=1 for sequential mode.

  Each worker writes to Evaluation/tmp/<model_name>/ so there are no CSV
  race conditions.  The merge step concatenates all temp dirs after stage 1.

SKIP / RESUME
─────────────
  A worker is skipped if Evaluation/tmp/<model_name>/ already exists and
  is non-empty — safe to Ctrl-C and resume.

  Stage-level skips:
    Stage 1   skipped per-model as above; full re-run with --force-inference
    Stage 2   skipped if biometric_metrics.csv exists; --force-biometric
    Stage 3   skipped if biometric_corr_*.csv exists; --force-correlation
    Stage 5   skipped if rank_stability.csv exists

USAGE
─────
  python run_all.py                        # full pipeline, 4 models at a time
  python run_all.py --parallel 2           # 2 models at a time
  python run_all.py --parallel 1           # fully sequential
  python run_all.py --force                # re-run everything from scratch
  python run_all.py --force-inference      # redo stage 1 only
  python run_all.py --force-biometric      # redo stage 2 only
  python run_all.py --force-correlation         # redo stage 3 only
  python run_all.py --rank-consistency           # also run stage 5
  python run_all.py --no-ijb               # skip IJB-B/C
  python run_all.py --no-synth             # skip synthetic datasets
  python run_all.py --no-library-models    # skip edgeface / facenets
  python run_all.py --demographics         # also run stage 4
  python run_all.py --dry-run              # print commands without executing
"""

import argparse
import csv
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).parent
EVAL_DIR   = SCRIPT_DIR / "Evaluation"
TMP_DIR    = EVAL_DIR / "tmp"

REAL_DETAIL  = EVAL_DIR / "real_detail.csv"
REAL_SUMMARY = EVAL_DIR / "real_summary.csv"
SYNTH_DETAIL = EVAL_DIR / "synth_detail.csv"
SYNTH_SUMMARY= EVAL_DIR / "synth_summary.csv"
BIO_METRICS  = EVAL_DIR / "biometric_metrics.csv"
DEMO_CSV     = EVAL_DIR / "demographics.csv"
RANK_STAB    = EVAL_DIR / "rank_stability.csv"

# ── CSV schemas (must match eval_utils.py exactly) ────────────────────────────

DETAIL_HEADER  = ["model", "dataset", "img1", "img2", "label",
                  "cosine_sim", "fold", "correct"]
SUMMARY_HEADER = ["model", "dataset", "accuracy", "std"]


# ═══════════════════════════════════════════════════════════════════════════════
#  WORKER MODE  (called by the orchestrator as a subprocess)
#  python run_all.py --worker --model-name X --model-path Y --out-dir Z ...
# ═══════════════════════════════════════════════════════════════════════════════

def _worker_main(args) -> None:
    """
    Runs one model through all three inference stages and writes results to
    args.out_dir.  This is the entry point when --worker is passed.
    """
    import numpy as np
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    from torchvision import transforms

    sys.path.insert(0, str(SCRIPT_DIR))
    from load_model import ModelSpec, load_model_by_spec, release_model
    from eval_utils import (
        parse_ann, PairDataset, make_transform, extract_embeddings,
        eval_10fold, append_to_detail_csv, append_to_summary_csv,
        BATCH_SIZE, NUM_WORKERS,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.gpu if torch.cuda.is_available() else "cpu")
    spec   = ModelSpec(
        name=args.model_name,
        path=None if args.model_path == "library" else args.model_path,
    )

    print(f"\n[{args.model_name}] loading on {device} …", flush=True)
    bundle = load_model_by_spec(spec, device, skip_on_error=False)
    print(f"[{args.model_name}] loaded", flush=True)

    VAL_DIR   = SCRIPT_DIR / "data" / "real"
    SYNTH_DIR = SCRIPT_DIR / "data" / "synth"
    IJB_DIR   = SCRIPT_DIR / "data" / "real" / "ijb"

    real_detail  = out_dir / "real_detail.csv"
    real_summary = out_dir / "real_summary.csv"
    syth_detail  = out_dir / "synth_detail.csv"
    synth_summary= out_dir / "synth_summary.csv"

    # ── 1a: Real benchmarks ───────────────────────────────────────────────────
    REAL_DATASETS = {
        "LFW":      VAL_DIR / "lfw_ann.txt",
        "CALFW":    VAL_DIR / "calfw_ann.txt",
        "CPLFW":    VAL_DIR / "cplfw_ann.txt",
        "AgeDB-30": VAL_DIR / "agedb_30_ann.txt",
        "CFP-FP":   VAL_DIR / "cfp_fp_ann.txt",
    }
    for ds_name, ann_path in REAL_DATASETS.items():
        if not ann_path.exists():
            continue
        pairs     = parse_ann(ann_path)
        transform = make_transform(bundle.height, bundle.width, bundle.mean, bundle.std)
        loader    = DataLoader(PairDataset(pairs, VAL_DIR, transform),
                               batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=True)
        emb1, emb2, labels = extract_embeddings(bundle.model, loader, device)
        acc, std, cos_sims, fold_asgn, thresh_map = eval_10fold(emb1, emb2, labels)
        print(f"[{args.model_name}] {ds_name}: {acc:.2f} ± {std:.2f}%", flush=True)
        append_to_detail_csv(real_detail,  bundle.name, ds_name, pairs, cos_sims, fold_asgn, thresh_map)
        append_to_summary_csv(real_summary, bundle.name, ds_name, acc, std)

    # ── 1b: IJB-B / IJB-C ────────────────────────────────────────────────────
    if not args.no_ijb:
        from eval_ijb import (
            _read_template_media_list, _read_template_pair_list,
            _read_image_list, _IJBDataset, _extract_embeddings,
            _image2template_feature, _compute_scores, _tar_at_far,
        )
        for target in args.ijb_targets:
            ds_dir = IJB_DIR / target
            meta   = ds_dir / "meta"
            t      = target.lower()
            paths  = {
                "tid_mid":  meta / f"{t}_face_tid_mid.txt",
                "pairs":    meta / f"{t}_template_pair_label.txt",
                "img_list": meta / f"{t}_name_5pts_score.txt",
                "img_dir":  ds_dir / "loose_crop",
            }
            if not all(p.exists() for p in paths.values()):
                print(f"[{args.model_name}] {target}: missing files, skipping", flush=True)
                continue

            templates, medias              = _read_template_media_list(paths["tid_mid"])
            p1, p2, label                  = _read_template_pair_list(paths["pairs"])
            img_names, landmarks, faceness = _read_image_list(paths["img_list"])

            tf = transforms.Compose([
                transforms.Resize((bundle.height, bundle.width)),
                transforms.ToTensor(),
                transforms.Normalize(bundle.mean, bundle.std),
            ])
            dataset   = _IJBDataset(paths["img_dir"], img_names, landmarks, tf)
            img_feats = _extract_embeddings(bundle, dataset, not args.no_flip, device)

            if not args.no_flip:
                D = img_feats.shape[1] // 2
                img_input = img_feats[:, :D] + img_feats[:, D:]
            else:
                img_input = img_feats
            if not args.no_detector_score:
                img_input = img_input * faceness[:, np.newaxis]

            template_norm_feats, unique_templates = _image2template_feature(img_input, templates, medias)
            scores  = _compute_scores(template_norm_feats, unique_templates, p1, p2)
            tar_map = _tar_at_far(scores, label)

            pairs_csv = [(f"template_{p1[i]}", f"template_{p2[i]}", int(label[i])) for i in range(len(p1))]
            append_to_detail_csv(real_detail, bundle.name, target,
                                 pairs_csv, scores,
                                 np.full(len(p1), -1, dtype=int), {-1: 0.0})
            append_to_summary_csv(real_summary, bundle.name, target, float("nan"), 0.0)
            print(f"[{args.model_name}] {target}: TAR@FAR=1e-4 = {tar_map[1e-4]*100:.2f}%", flush=True)

    # ── 1c: Synthetic datasets ────────────────────────────────────────────────
    if not args.no_synth and SYNTH_DIR.exists():
        for ds_dir in sorted(SYNTH_DIR.iterdir()):
            if not ds_dir.is_dir():
                continue
            ann_file = SYNTH_DIR / f"{ds_dir.name}_ann.txt"
            if not ann_file.exists():
                continue
            pairs     = parse_ann(ann_file)
            transform = make_transform(bundle.height, bundle.width, bundle.mean, bundle.std)
            loader    = DataLoader(PairDataset(pairs, ds_dir, transform),
                                   batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=True)
            emb1, emb2, labels = extract_embeddings(bundle.model, loader, device)
            acc, std, cos_sims, fold_asgn, thresh_map = eval_10fold(emb1, emb2, labels)
            print(f"[{args.model_name}] {ds_dir.name}: {acc:.2f} ± {std:.2f}%", flush=True)
            append_to_detail_csv(syth_detail,   bundle.name, ds_dir.name, pairs, cos_sims, fold_asgn, thresh_map)
            append_to_summary_csv(synth_summary, bundle.name, ds_dir.name, acc, std)

    release_model(bundle)
    print(f"[{args.model_name}] done → {out_dir}", flush=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  ORCHESTRATOR MODE
# ═══════════════════════════════════════════════════════════════════════════════

def _banner(title: str) -> None:
    print(f"\n{'━' * 60}\n  {title}\n{'━' * 60}", flush=True)


def _run(cmd: list, dry_run: bool) -> bool:
    print("  $", " ".join(str(c) for c in cmd), flush=True)
    if dry_run:
        return True
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print(f"\n[ERROR] Command failed (exit {result.returncode}).")
        return False
    return True


def _worker_is_done(model_name: str) -> bool:
    d = TMP_DIR / model_name
    return d.exists() and any(d.iterdir())


def _launch_worker(spec_name: str, spec_path, gpu: str, args) -> tuple[str, bool]:
    """
    Spawn this script in --worker mode for one model.
    Streams output live to stdout (prefixed with model name).
    Returns (model_name, success).
    """
    cmd = [
        sys.executable, __file__,
        "--worker",
        "--model-name",  spec_name,
        "--model-path",  spec_path if spec_path is not None else "library",
        "--gpu",         gpu,
        "--out-dir",     str(TMP_DIR / spec_name),
    ]
    if args.no_ijb:           cmd.append("--no-ijb")
    if args.no_synth:         cmd.append("--no-synth")
    if args.no_flip:          cmd.append("--no-flip")
    if args.no_detector_score:cmd.append("--no-detector-score")
    if not args.no_ijb:       cmd += ["--ijb-targets"] + args.target

    if args.dry_run:
        print("  $", " ".join(str(c) for c in cmd), flush=True)
        return spec_name, True

    log_path = TMP_DIR / spec_name / "worker.log"
    (TMP_DIR / spec_name).mkdir(parents=True, exist_ok=True)

    with open(log_path, "w") as log_f:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        for line in proc.stdout:
            print(line, end="", flush=True)   # live output
            log_f.write(line)
        proc.wait()

    success = proc.returncode == 0
    if not success:
        print(f"\n  [{spec_name}] FAILED — see {log_path}")
    return spec_name, success


def _merge_csvs(filename: str, out_path: Path, header: list[str]) -> int:
    """Concatenate all tmp/<model>/<filename> into out_path. Returns row count."""
    sources = sorted(TMP_DIR.glob(f"*/{filename}"))
    if not sources:
        return 0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with open(out_path, "w", newline="") as out_f:
        writer = csv.DictWriter(out_f, fieldnames=header)
        writer.writeheader()
        for src in sources:
            with open(src, newline="") as in_f:
                for row in csv.DictReader(in_f):
                    writer.writerow({k: row[k] for k in header if k in row})
                    total += 1
    return total


def stage_inference(args, force: bool) -> bool:
    _banner("Stage 1 — Parallel inference")

    sys.path.insert(0, str(SCRIPT_DIR))
    from load_model import list_model_specs
    specs = list_model_specs(
        model_dir=str(SCRIPT_DIR / "Pretrained_Models"),
        include_library_models=not args.no_library_models,
    )
    if not specs:
        print("[ERROR] No models found.")
        return False

    if force:
        if TMP_DIR.exists():
            shutil.rmtree(TMP_DIR)
            print(f"  [force] Deleted {TMP_DIR}")
        for f in [REAL_DETAIL, REAL_SUMMARY, SYNTH_DETAIL, SYNTH_SUMMARY]:
            if f.exists():
                f.unlink()
                print(f"  [force] Deleted {f.name}")

    pending = [(s.name, s.path) for s in specs if not _worker_is_done(s.name)]
    skipped = [s.name for s in specs if _worker_is_done(s.name)]

    if skipped:
        print(f"  [skip] Already done ({len(skipped)}): {skipped}")
    if not pending:
        print("  All models already done.  Use --force-inference to re-run.")
        return True

    n = args.parallel

    # Determine visible CUDA devices (respects CUDA_VISIBLE_DEVICES).
    # Workers round-robin across available devices, so e.g. 4 workers on
    # 2 GPUs get slots cuda:0, cuda:1, cuda:0, cuda:1.
    import torch as _torch
    n_gpus = _torch.cuda.device_count() if _torch.cuda.is_available() else 0
    if n_gpus == 0:
        def gpu_for(i): return "cpu"
        device_str = "cpu"
    else:
        def gpu_for(i): return f"cuda:{i % n_gpus}"
        device_str = (f"cuda:0–cuda:{n_gpus-1} round-robin" if n_gpus > 1 else "cuda:0")

    print(f"  {len(pending)} model(s) pending, {n} at a time on {device_str}\n")

    all_ok = True
    with ThreadPoolExecutor(max_workers=n) as pool:
        futures = {
            pool.submit(_launch_worker, name, path, gpu_for(i), args): name
            for i, (name, path) in enumerate(pending)
        }
        for future in as_completed(futures):
            name, ok = future.result()
            if not ok:
                all_ok = False

    # Merge
    _banner("Merging worker outputs")
    for filename, out_path, header in [
        ("real_detail.csv",   REAL_DETAIL,   DETAIL_HEADER),
        ("real_summary.csv",  REAL_SUMMARY,  SUMMARY_HEADER),
        ("synth_detail.csv",  SYNTH_DETAIL,  DETAIL_HEADER),
        ("synth_summary.csv", SYNTH_SUMMARY, SUMMARY_HEADER),
    ]:
        n_rows = _merge_csvs(filename, out_path, header)
        if n_rows:
            print(f"  {out_path.name:<25} ← {n_rows:,} rows")

    return all_ok


def stage_biometric(args, force: bool) -> bool:
    _banner("Stage 2 — Biometric metrics  (EER / FMR100 / FMR1000 / ZeroFMR / TAR)")

    if not force and BIO_METRICS.exists():
        print(f"  [skip] {BIO_METRICS.name} exists.")
        return True
    if force and BIO_METRICS.exists():
        BIO_METRICS.unlink()

    ok = True
    if REAL_DETAIL.exists() or args.dry_run:
        ok = _run([sys.executable, "eval_biometric.py", "--real-only", "--overwrite"], args.dry_run)
    if ok and not args.no_synth and (SYNTH_DETAIL.exists() or args.dry_run):
        ok = _run([sys.executable, "eval_biometric.py", "--synth-only"], args.dry_run)
    return ok


def stage_correlation(args, force: bool) -> bool:
    _banner("Stage 3 — Dataset correlations  (Pearson r + 95% CI)")

    corr_exists = any(EVAL_DIR.glob("biometric_corr_*.csv"))
    if not force and corr_exists:
        print(f"  [skip] biometric_corr_*.csv exists.")
        return True
    if not BIO_METRICS.exists() and not args.dry_run:
        print(f"  [skip] {BIO_METRICS.name} not found — run stage 2 first.")
        return True
    return _run([sys.executable, "eval_biometric_correlation.py"], args.dry_run)


def stage_demographics(args) -> bool:
    _banner("Stage 4 — Demographic analysis  (age / gender)")
    if DEMO_CSV.exists():
        print(f"  [skip] {DEMO_CSV.name} exists.")
        return True
    cmd = [sys.executable, "demographic_analysis.py", "--resume"]
    if args.no_synth:
        cmd.append("--real-only")
    return _run(cmd, args.dry_run)



def stage_rank_consistency(args) -> bool:
    _banner("Stage 5 — Rank consistency  (Spearman r + rank displacement)")

    if RANK_STAB.exists():
        print(f"  [skip] {RANK_STAB.name} exists.")
        return True
    if not BIO_METRICS.exists() and not args.dry_run:
        print(f"  [skip] {BIO_METRICS.name} not found — run stage 2 first.")
        return True

    cmd = [sys.executable, "eval_rank_consistency.py"]
    if args.no_synth:
        cmd += ["--split", "real"]
    return _run(cmd, args.dry_run)


# ── Argument parser (shared by orchestrator and worker) ───────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run the full face verification pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    # Orchestrator flags
    p.add_argument("--parallel",           type=int, default=4, metavar="N")
    p.add_argument("--force",              action="store_true")
    p.add_argument("--force-inference",    action="store_true")
    p.add_argument("--force-biometric",    action="store_true")
    p.add_argument("--force-correlation",  action="store_true")
    p.add_argument("--no-ijb",             action="store_true")
    p.add_argument("--no-synth",           action="store_true")
    p.add_argument("--no-library-models",  action="store_true")
    p.add_argument("--no-flip",            action="store_true")
    p.add_argument("--no-detector-score",  action="store_true")
    p.add_argument("--target", nargs="+",  default=["IJBB", "IJBC"],
                   choices=["IJBB", "IJBC"])
    p.add_argument("--dry-run",            action="store_true")
    # Worker-only flags (not exposed in help)
    p.add_argument("--worker",             action="store_true",  help=argparse.SUPPRESS)
    p.add_argument("--model-name",         default=None,         help=argparse.SUPPRESS)
    p.add_argument("--model-path",         default=None,         help=argparse.SUPPRESS)
    p.add_argument("--gpu",                default="cpu",        help=argparse.SUPPRESS)
    p.add_argument("--out-dir",            default=None,         help=argparse.SUPPRESS)
    p.add_argument("--ijb-targets", nargs="+", default=["IJBB", "IJBC"],
                                                                  help=argparse.SUPPRESS)
    return p


def main() -> None:
    args = _build_parser().parse_args()

    # ── Worker entry point ────────────────────────────────────────────────────
    if args.worker:
        _worker_main(args)
        return

    # ── Orchestrator entry point ──────────────────────────────────────────────
    t0 = time.time()

    if not stage_inference(args, force=args.force or args.force_inference):
        sys.exit(1)

    if not stage_biometric(args, force=args.force or args.force_biometric):
        sys.exit(1)

    if not stage_correlation(args, force=args.force or args.force_correlation):
        sys.exit(1)

    if not stage_demographics(args):
        sys.exit(1)

    if not stage_rank_consistency(args):
        sys.exit(1)

    _banner(f"Pipeline complete  ({(time.time()-t0)/60:.1f} min)")
    print(f"  Outputs → {EVAL_DIR}/\n")


if __name__ == "__main__":
    main()
