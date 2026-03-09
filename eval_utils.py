"""
eval_utils.py
─────────────────────────────────────────────────────────────────────────────
Shared building blocks used by eval_real.py, eval_synth.py and generate_ann.py:
  - parse_ann       (annotation file parser)
  - PairDataset
  - make_transform
  - extract_embeddings
  - eval_10fold
  - append_to_detail_csv   (per-pair detail log)
  - append_to_summary_csv  (one row per model × dataset)
"""

import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

BATCH_SIZE  = 64
NUM_WORKERS = 4

# ── Annotation parser ─────────────────────────────────────────────────────────

def parse_ann(ann_txt: Path) -> list[tuple[str, str, int]]:
    """
    Load an annotation file (format: label img1_path img2_path).
    Returns list of (img1_path, img2_path, label).
    """
    pairs = []
    with open(ann_txt) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 3:
                label, p1, p2 = parts
                pairs.append((p1, p2, int(label)))
    return pairs

# ── Dataset ───────────────────────────────────────────────────────────────────

class PairDataset(Dataset):
    """Loads image pairs from an annotation list."""

    def __init__(self, pairs: list[tuple[str, str, int]],
                 base_dir: Path, transform):
        self.pairs     = pairs
        self.base_dir  = Path(base_dir)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx):
        p1, p2, label = self.pairs[idx]
        img1 = Image.open(self.base_dir / p1)
        img1.load()
        img1 = img1.convert("RGB")
        img2 = Image.open(self.base_dir / p2)
        img2.load()
        img2 = img2.convert("RGB")
        return self.transform(img1), self.transform(img2), int(label)


# ── Transform factory ─────────────────────────────────────────────────────────

def make_transform(height: int, width: int,
                   mean: list[float], std: list[float]) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((height, width)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


# ── Embedding extraction ──────────────────────────────────────────────────────

@torch.no_grad()
def extract_embeddings(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    e1_list, e2_list, labels = [], [], []
    for img1, img2, label in loader:
        out1 = model(img1.to(device))
        out2 = model(img2.to(device))
        # SwinFace returns a dict  → extract "Recognition" embedding
        # TransFace and some ViTs return a tuple → embedding is index 0
        if isinstance(out1, dict):
            out1 = out1["Recognition"]
        elif isinstance(out1, tuple):
            out1 = out1[0]
        if isinstance(out2, dict):
            out2 = out2["Recognition"]
        elif isinstance(out2, tuple):
            out2 = out2[0]
        e1_list.append(F.normalize(out1, dim=1).cpu())
        e2_list.append(F.normalize(out2, dim=1).cpu())
        labels.extend(label.tolist())
    return torch.cat(e1_list), torch.cat(e2_list), np.array(labels)


# ── 10-fold evaluation ────────────────────────────────────────────────────────

def eval_10fold(
    emb1: torch.Tensor,
    emb2: torch.Tensor,
    labels: np.ndarray,
    n_folds: int = 10,
) -> tuple[float, float, np.ndarray, np.ndarray, dict[int, float]]:
    """
    Returns
    ───────
    mean_acc        : float   – mean accuracy across folds (%)
    std_acc         : float   – std across folds (%)
    cos_sim         : ndarray shape (N,)
    fold_assignments: ndarray shape (N,)  – test fold index per pair
    best_thresh_map : dict[fold_idx → best_threshold]
    """
    cos_sim          = (emb1 * emb2).sum(dim=1).numpy()
    fold_assignments = np.empty(len(labels), dtype=int)
    best_thresh_map: dict[int, float] = {}
    accs: list[float] = []

    for fold_idx, (train_idx, test_idx) in enumerate(
            KFold(n_splits=n_folds, shuffle=False).split(cos_sim)):
        fold_assignments[test_idx] = fold_idx

        best_thresh, best_acc = 0.0, 0.0
        for thresh in np.arange(-1.0, 1.0, 0.005):
            acc = ((cos_sim[train_idx] >= thresh) == labels[train_idx]).mean()
            if acc > best_acc:
                best_acc, best_thresh = acc, thresh
        best_thresh_map[fold_idx] = best_thresh

        preds = (cos_sim[test_idx] >= best_thresh).astype(int)
        accs.append((preds == labels[test_idx]).mean())

    return (
        float(np.mean(accs)) * 100,
        float(np.std(accs))  * 100,
        cos_sim,
        fold_assignments,
        best_thresh_map,
    )


# ── CSV writers ───────────────────────────────────────────────────────────────

DETAIL_HEADER  = ["model", "dataset", "img1", "img2", "label",
                  "cosine_sim", "fold", "correct"]
SUMMARY_HEADER = ["model", "dataset", "accuracy", "std"]


def append_to_detail_csv(
    csv_path: Path,
    model_name: str,
    dataset_name: str,
    pairs: list[tuple[str, str, int]],
    cos_sims: np.ndarray,
    fold_assignments: np.ndarray,
    best_thresh_map: dict[int, float],
) -> None:
    """Append one row per image pair."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists()

    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=DETAIL_HEADER)
        if write_header:
            writer.writeheader()
        for i, (p1, p2, label) in enumerate(pairs):
            fold   = int(fold_assignments[i])
            thresh = best_thresh_map.get(fold, 0.0)
            pred   = int(cos_sims[i] >= thresh)
            writer.writerow({
                "model":      model_name,
                "dataset":    dataset_name,
                "img1":       p1,
                "img2":       p2,
                "label":      label,
                "cosine_sim": f"{cos_sims[i]:.6f}",
                "fold":       fold,
                "correct":    int(pred == label),
            })


def append_to_summary_csv(
    csv_path: Path,
    model_name: str,
    dataset_name: str,
    acc: float,
    std: float,
) -> None:
    """Append one row per model × dataset result."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists()

    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_HEADER)
        if write_header:
            writer.writeheader()
        writer.writerow({
            "model":    model_name,
            "dataset":  dataset_name,
            "accuracy": f"{acc:.4f}",
            "std":      f"{std:.4f}",
        })
