"""
eval_synth.py
─────────────────────────────────────────────────────────────────────────────
Evaluates every model from load_all_models() on every synthetic dataset that
has a matching annotation file in data/synth/<dataset_name>_ann.txt.

Run generate_ann.py first to create those files.

Outputs
───────
  Evaluation/synth_detail.csv   – one row per model × pair
  Evaluation/synth_summary.csv  – one row per model × dataset

Usage
─────
  python eval_synth.py
  python eval_synth.py --dataset SynFace       # one dataset only
  python eval_synth.py --model edgeface_base   # one model only
  python eval_synth.py --no-library-models     # skip edgeface / facenets
"""

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from data.synth.synth_utils import SYNTH_ROOT, parse_ann
from load_model import list_model_specs, load_model_by_spec, release_model, ModelBundle
from eval_utils import (
    PairDataset, make_transform, extract_embeddings, eval_10fold,
    append_to_detail_csv, append_to_summary_csv, BATCH_SIZE, NUM_WORKERS,
)

# ── Paths ─────────────────────────────────────────────────────────────────────

SCRIPT_DIR  = Path(__file__).parent
MODEL_DIR   = SCRIPT_DIR / "Pretrained_Models"
EVAL_DIR    = SCRIPT_DIR / "Evaluation"
DETAIL_CSV  = EVAL_DIR / "synth_detail.csv"
SUMMARY_CSV = EVAL_DIR / "synth_summary.csv"


# ── Single model × dataset evaluation ────────────────────────────────────────

def evaluate(
    bundle: ModelBundle,
    ds_name: str,
    device: torch.device,
) -> tuple[float, float] | None:

    ann_file = SYNTH_ROOT / f"{ds_name}_ann.txt"
    ds_dir   = SYNTH_ROOT / ds_name

    if not ann_file.exists():
        print(f"  [skip] {ann_file.name} not found — run generate_ann.py first.")
        return None
    if not ds_dir.exists():
        print(f"  [skip] Dataset folder not found: {ds_dir}")
        return None

    pairs = parse_ann(ann_file)
    if len(pairs) < 10:
        print(f"  [skip] Only {len(pairs)} pairs in {ann_file.name}")
        return None

    transform = make_transform(bundle.height, bundle.width, bundle.mean, bundle.std)
    dataset   = PairDataset(pairs, ds_dir, transform)
    loader    = DataLoader(dataset, batch_size=BATCH_SIZE,
                           num_workers=NUM_WORKERS, pin_memory=True)

    emb1, emb2, labels = extract_embeddings(bundle.model, loader, device)
    acc, std, cos_sims, fold_asgn, thresh_map = eval_10fold(emb1, emb2, labels)

    print(f"    {ds_name:<30}: {acc:.2f} ± {std:.2f}%")

    append_to_detail_csv(DETAIL_CSV, bundle.name, ds_name,
                         pairs, cos_sims, fold_asgn, thresh_map)
    append_to_summary_csv(SUMMARY_CSV, bundle.name, ds_name, acc, std)

    return acc, std


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate all models on synthetic FR datasets."
    )
    parser.add_argument(
        "--model-dir", type=str, default=str(MODEL_DIR),
        help=f"Folder with .pt/.pth weight files (default: {MODEL_DIR})"
    )
    parser.add_argument(
        "--dataset", type=str, default=None,
        help="Evaluate only this dataset (folder name)."
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Evaluate only this model (by name)."
    )
    parser.add_argument(
        "--no-library-models", action="store_true",
        help="Skip edgeface_base and InceptionResnetV1 (library models)."
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # List models without loading weights yet
    specs = list_model_specs(
        model_dir=args.model_dir,
        include_library_models=not args.no_library_models,
    )
    if args.model:
        specs = [s for s in specs if s.name == args.model]
        if not specs:
            print(f"[ERROR] Model '{args.model}' not found.")
            return

    if not specs:
        print("[ERROR] No models found.")
        return

    print(f"Found {len(specs)} model(s): {[s.name for s in specs]}\n")

    # Resolve datasets
    if args.dataset:
        ds_names = [args.dataset]
    else:
        if not SYNTH_ROOT.exists():
            print(f"[ERROR] {SYNTH_ROOT} not found")
            return
        ds_names = sorted(d.name for d in SYNTH_ROOT.iterdir() if d.is_dir())

    if not ds_names:
        print("[ERROR] No datasets found.")
        return

    # Evaluate one model at a time — load, evaluate all datasets, release
    summary: dict[str, dict[str, tuple[float, float]]] = {}
    for spec in specs:
        bundle = load_model_by_spec(spec, device)
        if bundle is None:
            continue

        print(f"━━━ {bundle.name} ━━━")
        summary[bundle.name] = {}
        for ds_name in ds_names:
            result = evaluate(bundle, ds_name, device)
            if result:
                summary[bundle.name][ds_name] = result

        release_model(bundle)
        del bundle
        print()

    # Summary table
    col_w    = 35
    header   = f"{'Model':<35}" + "".join(f"{d:>{col_w}}" for d in ds_names)
    print("\n══ Summary ══")
    print(header)
    print("─" * len(header))
    for model_name, results in summary.items():
        row = f"{model_name:<35}"
        for ds in ds_names:
            if ds in results:
                acc, std = results[ds]
                row += f"{acc:>{col_w-4}.2f}±{std:.2f}"
            else:
                row += f"{'—':>{col_w}}"
        print(row)

    print(f"\nDetail  → {DETAIL_CSV}")
    print(f"Summary → {SUMMARY_CSV}")


if __name__ == "__main__":
    main()
