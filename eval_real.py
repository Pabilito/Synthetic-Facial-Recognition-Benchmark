"""
eval_real.py
─────────────────────────────────────────────────────────────────────────────
Evaluates every model returned by load_all_models() on the real face
verification benchmarks: LFW, CALFW, CPLFW, AgeDB-30.

Expected structure (relative to this script):
  data/val/
    lfw_112x112/        0.bmp, 1.bmp, ...
    calfw_112x112/
    cplfw_112x112/
    agedb_30_112x112/   (or however your folder is named)
    lfw_ann.txt         format: label img1_path img2_path
    calfw_ann.txt
    cplfw_ann.txt
    agedb_30_ann.txt

  Pretrained_Models/
    IResNet_R18_Glint360K.pth
    IResNet_R50_Glint360K.pth
    IResNet_R100_Glint360K.pth
    IDiffFace_CF50.pth
    SFace_IResNet50.pth
    TransFace_S_Glint360K.pt
    SwinFace.pt
    (+ edgeface_base and FaceNets loaded from libraries)

Outputs
───────
  Evaluation/real_detail.csv   – one row per model × pair
  Evaluation/real_summary.csv  – one row per model × dataset

Usage
─────
  python eval_real.py
  python eval_real.py --model-dir path/to/weights   # override model folder
  python eval_real.py --no-library-models           # skip edgeface / facenets
"""

import argparse
from pathlib import Path

from torch.utils.data import DataLoader
import torch

from load_model import list_model_specs, load_model_by_spec, release_model, ModelBundle
from eval_utils import (
    parse_ann, PairDataset, make_transform, extract_embeddings, eval_10fold,
    append_to_detail_csv, append_to_summary_csv, BATCH_SIZE, NUM_WORKERS,
)

# ── Paths ─────────────────────────────────────────────────────────────────────

SCRIPT_DIR  = Path(__file__).parent
VAL_DIR     = SCRIPT_DIR / "data" / "real"
MODEL_DIR   = SCRIPT_DIR / "Pretrained_Models"
EVAL_DIR    = SCRIPT_DIR / "Evaluation"
DETAIL_CSV  = EVAL_DIR / "real_detail.csv"
SUMMARY_CSV = EVAL_DIR / "real_summary.csv"

# ── Dataset registry ──────────────────────────────────────────────────────────

DATASETS: dict[str, Path] = {
    "LFW":      VAL_DIR / "lfw_ann.txt",
    "CALFW":    VAL_DIR / "calfw_ann.txt",
    "CPLFW":    VAL_DIR / "cplfw_ann.txt",
    "AgeDB-30": VAL_DIR / "agedb_30_ann.txt",
}


# ── Single model × dataset evaluation ────────────────────────────────────────

def evaluate(
    bundle: ModelBundle,
    ds_name: str,
    ann_path: Path,
    device: torch.device,
) -> tuple[float, float] | None:

    if not ann_path.exists():
        print(f"  [skip] Annotation file not found: {ann_path.name}")
        return None

    pairs = parse_ann(ann_path)
    transform = make_transform(bundle.height, bundle.width, bundle.mean, bundle.std)
    dataset   = PairDataset(pairs, VAL_DIR, transform)
    loader    = DataLoader(dataset, batch_size=BATCH_SIZE,
                           num_workers=NUM_WORKERS, pin_memory=True)

    emb1, emb2, labels = extract_embeddings(bundle.model, loader, device)
    acc, std, cos_sims, fold_asgn, thresh_map = eval_10fold(emb1, emb2, labels)

    print(f"    {ds_name:<12}: {acc:.2f} ± {std:.2f}%")

    append_to_detail_csv(DETAIL_CSV, bundle.name, ds_name,
                         pairs, cos_sims, fold_asgn, thresh_map)
    append_to_summary_csv(SUMMARY_CSV, bundle.name, ds_name, acc, std)

    return acc, std


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate all models on real face verification benchmarks."
    )
    parser.add_argument(
        "--model-dir", type=str, default=str(MODEL_DIR),
        help=f"Folder containing .pt/.pth weight files (default: {MODEL_DIR})"
    )
    parser.add_argument(
        "--no-library-models", action="store_true",
        help="Skip edgeface_base and InceptionResnetV1 (library models)."
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # List all models without loading weights yet
    specs = list_model_specs(
        model_dir=args.model_dir,
        include_library_models=not args.no_library_models,
    )
    if not specs:
        print("[ERROR] No models found.")
        return

    print(f"Found {len(specs)} model(s): {[s.name for s in specs]}\n")

    # Check datasets
    available = {name: path for name, path in DATASETS.items() if path.exists()}
    missing   = [name for name, path in DATASETS.items() if not path.exists()]
    if missing:
        print(f"[warn] Annotation files not found, skipping: {missing}")
    if not available:
        print("[ERROR] No annotation files found under data/real/")
        return

    # Evaluate one model at a time — load, evaluate all datasets, release
    summary: dict[str, dict[str, tuple[float, float]]] = {}
    for spec in specs:
        bundle = load_model_by_spec(spec, device)
        if bundle is None:
            continue

        print(f"━━━ {bundle.name} ━━━")
        summary[bundle.name] = {}
        for ds_name, ann_path in available.items():
            result = evaluate(bundle, ds_name, ann_path, device)
            if result:
                summary[bundle.name][ds_name] = result

        release_model(bundle)
        del bundle
        print()

    # Summary table
    ds_names = list(available.keys())
    col_w    = 13
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
