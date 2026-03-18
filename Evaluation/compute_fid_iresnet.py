"""
compute_fid_iresnet.py
-----------------
Same as compute_fid.py but uses a pretrained IResNet-R18 face recognition
model (Pretrained_Models/IResNet_R18_MS1MV3.pt) as the feature extractor
instead of InceptionV3.

Why: InceptionV3 was trained on ImageNet (no face class). IResNet-R18 trained
on MS1MV3 with ArcFace loss produces 512-d embeddings that encode
identity-discriminative face features, giving a more meaningful FID for
face dataset comparison.

Differences from compute_fid.py (v1):
  - Feature extractor : IResNet-R18 (512-d)  vs InceptionV3 (2048-d)
  - Input resolution  : 112×112               vs 299×299
  - Cache filename    : _fid_stats_r18.npz    vs _fid_stats.npz
  - Model path arg    : --model-path
  - No --dims flag    (always 512)

Everything else — ann-file parsing, incremental CSV, caching, crash
recovery, pair comparisons — is identical to v1.

Usage:
    python compute_fid_iresnet.py
    python compute_fid_iresnet.py --model-path ../Pretrained_Models/IResNet_R18_MS1MV3.pt
    python compute_fid_iresnet.py --batch-size 256 --device cuda
    python compute_fid_iresnet.py --recompute-stats
    python compute_fid_iresnet.py --recompute-pairs
"""

import re
import csv
import sys
import time
import argparse
import itertools
import numpy as np
import torch
import torchvision.transforms as TF

from pathlib import Path
from datetime import datetime, timedelta
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets.folder import default_loader
from scipy import linalg

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

# Separate cache file so v1 and v2 stats never collide
STATS_FILENAME = "_fid_stats_r18.npz"
FEATURE_DIM    = 512   # IResNet output dimension


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_iresnet_r18(model_path: Path, device: str) -> torch.nn.Module:
    """
    Load IResNet-R18 from a .pt file.

    IResNet_R18_MS1MV3.pt is saved as a full serialised model object
    (not a state-dict), so weights_only=False is required — same logic
    as _load_generic_pt() in load_model.py.
    """
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    print(f"Loading IResNet-R18 from {model_path} ...")
    model = torch.load(str(model_path), map_location=torch.device(device), weights_only=False)
    model = model.to(device).eval()
    print("Done.\n")
    return model


# ---------------------------------------------------------------------------
# Dataset — identical to v1
# ---------------------------------------------------------------------------

def parse_ann_file(ann_file: Path, split_dir: Path, dataset_dir: Path) -> list[Path]:
    """
    Parse annotation file with format: <label> <path1> <path2>

    Two formats:
      Real:  first path component == dataset folder name
             e.g. "agedb_30_112x112/0.bmp"  -> split_dir / p
      Synth: first path component is an identity ID (not the dataset folder)
             e.g. "27982/28.jpg"             -> dataset_dir / p

    Detection: check whether the first component matches the dataset folder name.
    One check on the first line — no per-image exists() calls.
    """
    base_dir    = None
    dataset_name = dataset_dir.name
    first_p      = None

    with open(ann_file) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 3:
                continue
            first_p = parts[1]
            first_component = Path(first_p).parts[0]
            base_dir = split_dir if first_component == dataset_name else dataset_dir
            break

    if base_dir is None:
        raise ValueError(f"Could not determine path format from: {ann_file}")

    print(f"            Path format: {'split_dir' if base_dir == split_dir else 'dataset_dir'} "
          f"(first component: {Path(first_p).parts[0]!r}, dataset: {dataset_name!r})")

    seen  = set()
    files = []
    with open(ann_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 3:
                continue
            for p in parts[1:]:
                resolved = base_dir / p
                if resolved not in seen:
                    seen.add(resolved)
                    files.append(resolved)

    if not files:
        raise ValueError(f"No image paths found in: {ann_file}")
    return sorted(files)


class AnnDataset(Dataset):
    """Loads images listed in an annotation file."""
    def __init__(self, files: list[Path], transform):
        self.files     = files
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = default_loader(str(self.files[idx]))
        return self.transform(img)


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def compute_stats(
    files: list[Path],
    model: torch.nn.Module,
    batch_size: int,
    device: str,
    desc: str = "",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Pass all images through IResNet-R18 and return (mu, sigma) of the
    512-d embedding distribution.

    Preprocessing matches load_model.py / the MS1MV3 training pipeline:
      - Resize to 112×112
      - ToTensor  (→ [0, 1])
      - Normalize with mean=std=[0.5, 0.5, 0.5]  (→ [-1, 1])
    """
    transform = TF.Compose([
        TF.Resize((112, 112)),
        TF.ToTensor(),
        TF.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    dataset = AnnDataset(files, transform)
    loader  = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=min(8, batch_size // 8),
        pin_memory=(device != "cpu"),
        drop_last=False,
    )

    features = []
    iterator = tqdm(loader, desc=desc, unit="batch", leave=False) if HAS_TQDM else loader

    with torch.no_grad():
        for batch in iterator:
            batch = batch.to(device)
            emb   = model(batch)          # (B, 512) — plain tensor from IResNet
            # IResNet forward() returns a BN-normalised embedding directly
            features.append(emb.cpu().float().numpy())

    features = np.concatenate(features, axis=0)   # (N, 512)
    mu    = np.mean(features, axis=0)
    sigma = np.cov(features, rowvar=False)
    return mu, sigma


def load_or_compute_stats(
    dataset: dict,
    model: torch.nn.Module,
    batch_size: int,
    device: str,
    recompute: bool,
    data_root: Path,
) -> tuple[np.ndarray, np.ndarray]:
    stats_path = dataset["path"] / STATS_FILENAME

    if stats_path.exists() and not recompute:
        data = np.load(stats_path)
        print(f"  [CACHE] {dataset['split']}/{dataset['name']}")
        return data["mu"], data["sigma"]

    label = f"{dataset['split']}/{dataset['name']}"
    print(f"  [COMPUTE] {label}")
    t0 = time.time()

    split_dir   = data_root / dataset["split"]
    dataset_dir = dataset["path"]
    files = parse_ann_file(dataset["ann_file"], split_dir, dataset_dir)
    print(f"            {len(files)} unique images from {dataset['ann_file'].name}")

    mu, sigma = compute_stats(files, model, batch_size, device, desc=f"  {label}")
    elapsed   = timedelta(seconds=int(time.time() - t0))
    np.savez(stats_path, mu=mu, sigma=sigma)
    print(f"            Done in {elapsed} → saved {stats_path.name}")
    return mu, sigma


# ---------------------------------------------------------------------------
# Fréchet distance — identical to v1
# ---------------------------------------------------------------------------

def frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6) -> float:
    mu1    = np.atleast_1d(mu1)
    mu2    = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)
    assert mu1.shape    == mu2.shape,    f"Mean shape mismatch: {mu1.shape} vs {mu2.shape}"
    assert sigma1.shape == sigma2.shape, f"Cov shape mismatch:  {sigma1.shape} vs {sigma2.shape}"

    diff    = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1 @ sigma2, disp=False)
    if not np.isfinite(covmean).all():
        offset  = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset) @ (sigma2 + offset))
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            raise ValueError("Imaginary component in sqrtm result")
        covmean = covmean.real
    return float(diff @ diff + np.trace(sigma1 + sigma2 - 2 * covmean))


# ---------------------------------------------------------------------------
# Dataset discovery — identical to v1
# ---------------------------------------------------------------------------

def strip_resolution(name: str) -> str:
    return re.sub(r"_\d+x\d+$", "", name)


def find_ann_file(name: str, dataset_dir: Path, split_dir: Path):
    base = strip_resolution(name)
    candidates = [
        dataset_dir / f"{name}_ann.txt",
        split_dir   / f"{name}_ann.txt",
        split_dir   / f"{base}_ann.txt",
        dataset_dir / f"{base}_ann.txt",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def discover_datasets(data_root: Path) -> list[dict]:
    datasets = []
    for split in ("real", "synth"):
        split_dir = data_root / split
        if not split_dir.exists():
            print(f"[WARN] Not found, skipping: {split_dir}")
            continue
        for dataset_dir in sorted(split_dir.iterdir()):
            if not dataset_dir.is_dir():
                continue
            name     = dataset_dir.name
            ann_file = find_ann_file(name, dataset_dir, split_dir)
            if ann_file is None:
                print(f"[SKIP] No annotation file for: {split}/{name}")
                continue
            print(f"  [OK]  {split}/{name:35s} ann: {ann_file.relative_to(data_root)}")
            datasets.append({"name": name, "split": split,
                             "path": dataset_dir, "ann_file": ann_file})
    return datasets


# ---------------------------------------------------------------------------
# Incremental CSV — identical to v1
# ---------------------------------------------------------------------------

def load_existing_results(output_path: Path) -> set[tuple[str, str]]:
    done = set()
    if not output_path.exists():
        return done
    with open(output_path, newline="") as f:
        for row in csv.DictReader(f):
            done.add((row["dataset_a"], row["dataset_b"]))
    return done


def append_result(output_path: Path, row: dict, write_header: bool):
    fieldnames = ["dataset_a", "dataset_b", "fid", "split_a", "split_b", "timestamp"]
    with open(output_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    script_dir = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(
        description="Compute FID between all dataset pairs using IResNet-R18 (v2)."
    )
    parser.add_argument(
        "--model-path", type=str,
        default=str(script_dir.parent / "Pretrained_Models" / "IResNet_R18_MS1MV3.pt"),
        help="Path to IResNet_R18_MS1MV3.pt",
    )
    parser.add_argument("--batch-size", type=int, default=256,
                        help="Larger than v1 — 112×112 images are much cheaper than 299×299")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=str, default="fid_results_r18.csv")
    parser.add_argument("--recompute-stats", action="store_true",
                        help="Force recompute cached R18 stats for all datasets")
    parser.add_argument("--recompute-pairs", action="store_true",
                        help="Force recompute all pair comparisons")
    args = parser.parse_args()

    model_path  = Path(args.model_path)
    data_root   = script_dir.parent / "data"
    output_path = script_dir / args.output

    print("=" * 60)
    print(f"  compute_fid_iresnet.py  —  IResNet-R18 / MS1MV3")
    print(f"  Model     : {model_path}")
    print(f"  Data root : {data_root}")
    print(f"  Output    : {output_path}")
    print(f"  Device    : {args.device}  |  Batch: {args.batch_size}")
    print(f"  Features  : {FEATURE_DIM}-d  |  Cache: {STATS_FILENAME}")
    print("=" * 60 + "\n")

    # --- Discover datasets ---
    datasets = discover_datasets(data_root)
    if len(datasets) < 2:
        print("Need at least 2 valid datasets. Exiting.")
        sys.exit(1)

    print(f"\nFound {len(datasets)} datasets.\n")

    # --- Load model once ---
    model = load_iresnet_r18(model_path, args.device)

    # --- Phase 1: compute / load stats ---
    print("=" * 60)
    print("  Phase 1 — Feature extraction (IResNet-R18, one pass per dataset)")
    print("=" * 60)
    t0 = time.time()

    for d in datasets:
        mu, sigma = load_or_compute_stats(
            d, model, args.batch_size, args.device,
            args.recompute_stats, data_root,
        )
        d["mu"]    = mu
        d["sigma"] = sigma

    t_stats = timedelta(seconds=int(time.time() - t0))
    print(f"\nPhase 1 done in {t_stats}.\n")

    # --- Phase 2: all-pairs FID ---
    all_pairs  = list(itertools.combinations(datasets, 2))
    done_pairs = set() if args.recompute_pairs else load_existing_results(output_path)
    pending    = [
        (a, b) for a, b in all_pairs
        if (f"{a['split']}/{a['name']}", f"{b['split']}/{b['name']}") not in done_pairs
    ]

    n_skip = len(all_pairs) - len(pending)
    print("=" * 60)
    print(f"  Phase 2 — FID computation")
    print(f"  {len(all_pairs)} total pairs | {n_skip} skipped (cached) | {len(pending)} to compute")
    print("=" * 60)

    if args.recompute_pairs and output_path.exists():
        output_path.unlink()

    write_header = not output_path.exists()
    t1           = time.time()
    pair_times   = []

    pair_iter = pending
    if HAS_TQDM:
        pair_iter = tqdm(pending, desc="  Pairs", unit="pair")

    for i, (a, b) in enumerate(pair_iter, 1):
        label_a = f"{a['split']}/{a['name']}"
        label_b = f"{b['split']}/{b['name']}"

        tp = time.time()
        try:
            fid = round(frechet_distance(a["mu"], a["sigma"], b["mu"], b["sigma"]), 4)
        except Exception as e:
            fid = f"ERROR: {e}"
        pair_times.append(time.time() - tp)

        if not HAS_TQDM:
            avg = sum(pair_times) / len(pair_times)
            eta = timedelta(seconds=int(avg * (len(pending) - i)))
            print(f"  [{i:>3}/{len(pending)}] {label_a}  vs  {label_b}  →  {fid}  (ETA {eta})")

        append_result(output_path, {
            "dataset_a": label_a,
            "dataset_b": label_b,
            "fid":       fid,
            "split_a":   a["split"],
            "split_b":   b["split"],
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }, write_header=(write_header and i == 1))

    t_pairs = timedelta(seconds=int(time.time() - t1))
    t_total = timedelta(seconds=int(time.time() - t0))

    print(f"\n{'=' * 60}")
    print(f"  All done!")
    print(f"  Feature extractor            : IResNet-R18 / MS1MV3 ({FEATURE_DIM}-d)")
    print(f"  Phase 1 (feature extraction) : {t_stats}")
    print(f"  Phase 2 (pair comparisons)   : {t_pairs}")
    print(f"  Total                        : {t_total}")
    print(f"  Results saved to             : {output_path}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
