"""
eval_ijb.py
─────────────────────────────────────────────────────────────────────────────
Evaluates all models from load_model.py on IJB-B and/or IJB-C using the
standard template-based 1:1 verification protocol.

Results are written into the SAME CSV files as eval_real.py using the same
eval_utils helpers, so eval_biometric.py picks them up automatically and
reports EER / FMR100 / FMR1000 / ZeroFMR / TAR@FAR1 / TAR@FAR01 alongside
LFW / CALFW / CFP-FP / etc. with no extra steps.

PROTOCOL DIFFERENCES vs. LFW / CFP-FP
───────────────────────────────────────
  • Images are aligned at eval time via similarity transform on 5-pt landmarks
    (loose crops from the dataset, not pre-aligned 112×112 crops).
  • Multiple images per identity are pooled into a single template embedding
    via media-aware mean pooling (video frames averaged per clip, then summed
    across clips) followed by L2 normalisation.
  • Scores = cosine similarity between L2-normalised template embeddings.
  • No 10-fold CV: IJB uses all pairs for a single ROC evaluation.
    → fold = -1, correct = -1 in detail CSV (eval_biometric.py only reads
      model / dataset / label / cosine_sim, so this does not affect it).
    → accuracy in summary CSV = TAR @ FAR=1e-4 × 100, std = 0.0.

OUTPUTS  (shared with eval_real.py — appended, never overwritten)
──────────────────────────────────────────────────────────────────
  Evaluation/real_detail.csv     one row per template pair
  Evaluation/real_summary.csv    one row per model × dataset

  After running this script, execute eval_biometric.py as usual:
    python eval_biometric.py --real-only --datasets IJBB IJBC

EXPECTED FOLDER STRUCTURE
──────────────────────────
  data/real/ijb/
    IJBB/
      loose_crop/
      meta/
        ijbb_face_tid_mid.txt          "<img_name> <tid> <mid>"
        ijbb_name_5pts_score.txt       "<img_name> <lmk×10> <det_score>"
        ijbb_template_pair_label.txt   "<tid1> <tid2> <label>"
    IJBC/
      loose_crop/
      meta/
        ijbc_face_tid_mid.txt
        ijbc_name_5pts_score.txt
        ijbc_template_pair_label.txt

USAGE
─────
  python eval_ijb.py                    # evaluate on both IJBB and IJBC
  python eval_ijb.py --target IJBC
  python eval_ijb.py --no-flip          # disable horizontal-flip augmentation
  python eval_ijb.py --no-detector-score
  python eval_ijb.py --no-library-models
  python eval_ijb.py --ijb-dir /path/to/ijb
"""

import argparse
import warnings
from pathlib import Path

import cv2
import numpy as np
import sklearn.preprocessing
import torch
import torch.nn.functional as F
from PIL import Image as PILImage
from prettytable import PrettyTable
from sklearn.metrics import roc_curve
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

warnings.filterwarnings("ignore")

from load_model import list_model_specs, load_model_by_spec, release_model, ModelBundle
from eval_utils import (
    BATCH_SIZE, NUM_WORKERS,
    append_to_detail_csv, append_to_summary_csv,
)

# ── Paths ─────────────────────────────────────────────────────────────────────

SCRIPT_DIR  = Path(__file__).parent
IJB_DIR     = SCRIPT_DIR / "data" / "real" / "ijb"
MODEL_DIR   = SCRIPT_DIR / "Pretrained_Models"
EVAL_DIR    = SCRIPT_DIR / "Evaluation"
DETAIL_CSV  = EVAL_DIR / "real_detail.csv"
SUMMARY_CSV = EVAL_DIR / "real_summary.csv"

FAR_TARGETS = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]

# ── 5-point landmark alignment ────────────────────────────────────────────────

_ARCFACE_DST = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041],
], dtype=np.float32)


def align_face(img: np.ndarray, lmk: np.ndarray, size: int = 112) -> np.ndarray:
    """Warp a BGR uint8 image to a canonical 112×112 face crop."""
    from skimage.transform import SimilarityTransform
    tform = SimilarityTransform.from_estimate(lmk, _ARCFACE_DST)
    return cv2.warpAffine(img, tform.params[:2], (size, size), borderValue=0.0)


# ── Protocol file readers ─────────────────────────────────────────────────────

def _read_template_media_list(path: Path):
    """Returns (templates, medias) int arrays of shape (N,)."""
    data = np.genfromtxt(path, dtype=str)
    return data[:, 1].astype(int), data[:, 2].astype(int)


def _read_template_pair_list(path: Path):
    """Returns (p1, p2, label) int arrays of shape (M,)."""
    data = np.genfromtxt(path, dtype=str)
    return data[:, 0].astype(int), data[:, 1].astype(int), data[:, 2].astype(int)


def _read_image_list(path: Path):
    """
    Returns:
      img_names      : list[str]            length N
      landmarks      : float32 array (N,5,2)
      faceness_scores: float32 array (N,)
    """
    img_names, landmarks, scores = [], [], []
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 12:
                continue
            img_names.append(parts[0])
            landmarks.append(np.array(parts[1:11], dtype=np.float32).reshape(5, 2))
            scores.append(float(parts[11]))
    return img_names, np.array(landmarks, dtype=np.float32), np.array(scores, dtype=np.float32)


# ── Dataset ───────────────────────────────────────────────────────────────────

class _IJBDataset(Dataset):
    def __init__(self, img_dir: Path, img_names: list, landmarks: np.ndarray, transform):
        self.img_dir   = img_dir
        self.img_names = img_names
        self.landmarks = landmarks
        self.transform = transform

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img = cv2.imread(str(self.img_dir / self.img_names[idx]))
        if img is None:
            img = np.zeros((112, 112, 3), dtype=np.uint8)
        else:
            img = align_face(img, self.landmarks[idx])
        # BGR → RGB, then hand off to transform (same pipeline as eval_real.py)
        pil = PILImage.fromarray(img[:, :, ::-1].copy())
        return self.transform(pil), idx


# ── Embedding extraction ──────────────────────────────────────────────────────

@torch.no_grad()
def _extract_embeddings(
    bundle:   ModelBundle,
    dataset:  _IJBDataset,
    use_flip: bool,
    device:   torch.device,
) -> np.ndarray:
    """
    Returns float32 array (N, D) — one L2-normalised embedding per image.

    When use_flip=True each image is also passed through its horizontal mirror.
    Following IJB_11.py the two halves are concatenated (orig ∥ flip) so the
    caller can sum them; final shape (N, 2D).
    """
    norm_tf = transforms.Compose([
        transforms.Resize((bundle.height, bundle.width)),
        transforms.ToTensor(),
        transforms.Normalize(bundle.mean, bundle.std),
    ])
    flip_tf = transforms.Compose([
        transforms.RandomHorizontalFlip(p=1.0),
        norm_tf,
    ])

    loader = DataLoader(dataset, batch_size=BATCH_SIZE,
                        num_workers=NUM_WORKERS, shuffle=False, pin_memory=True)

    N = len(dataset)
    D = None
    feats  = None
    fflips = None

    def _fwd(batch: torch.Tensor) -> np.ndarray:
        out = bundle.model(batch.to(device))
        if isinstance(out, dict):    out = out["Recognition"]
        elif isinstance(out, tuple): out = out[0]
        return F.normalize(out, dim=1).cpu().numpy()

    for batch_imgs, batch_idx in loader:
        emb = _fwd(batch_imgs)
        if D is None:
            D = emb.shape[1]
            feats = np.zeros((N, D), dtype=np.float32)
            if use_flip:
                fflips = np.zeros((N, D), dtype=np.float32)
        idx = batch_idx.numpy()
        feats[idx] = emb

        if use_flip:
            pils = []
            for i in idx:
                img = cv2.imread(str(dataset.img_dir / dataset.img_names[i]))
                if img is None:
                    img = np.zeros((112, 112, 3), dtype=np.uint8)
                else:
                    img = align_face(img, dataset.landmarks[i])
                pils.append(flip_tf(PILImage.fromarray(img[:, :, ::-1].copy())))
            fflips[idx] = _fwd(torch.stack(pils))

    if use_flip:
        return np.concatenate([feats, fflips], axis=1)   # (N, 2D)
    return feats                                          # (N, D)


# ── Template aggregation (media-aware mean pool — identical to IJB_11.py) ─────

def _image2template_feature(img_feats: np.ndarray,
                             templates: np.ndarray,
                             medias:    np.ndarray):
    """Returns (template_norm_feats, unique_templates)."""
    unique_templates = np.unique(templates)
    template_feats   = np.zeros((len(unique_templates), img_feats.shape[1]), dtype=np.float32)

    for count, uqt in enumerate(unique_templates):
        ind_t       = np.where(templates == uqt)[0]
        face_feats  = img_feats[ind_t]
        face_medias = medias[ind_t]
        u_medias, u_counts = np.unique(face_medias, return_counts=True)

        media_feats = []
        for u, ct in zip(u_medias, u_counts):
            ind_m = np.where(face_medias == u)[0]
            media_feats.append(
                np.mean(face_feats[ind_m], axis=0, keepdims=True) if ct > 1
                else face_feats[ind_m]
            )
        template_feats[count] = np.sum(np.vstack(media_feats), axis=0)
        if count % 2000 == 0:
            print(f"    template aggregation: {count}/{len(unique_templates)}")

    return sklearn.preprocessing.normalize(template_feats), unique_templates


# ── Template-pair cosine similarity ──────────────────────────────────────────

def _compute_scores(template_norm_feats: np.ndarray,
                    unique_templates:    np.ndarray,
                    p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    t2id = np.zeros(max(unique_templates) + 1, dtype=int)
    for i, uqt in enumerate(unique_templates):
        t2id[uqt] = i
    scores = np.zeros(len(p1), dtype=np.float32)
    for start in range(0, len(p1), 100_000):
        s = slice(start, start + 100_000)
        scores[s] = (template_norm_feats[t2id[p1[s]]] *
                     template_norm_feats[t2id[p2[s]]]).sum(axis=1)
    return scores


# ── TAR @ FAR ─────────────────────────────────────────────────────────────────

def _tar_at_far(scores: np.ndarray, labels: np.ndarray,
                far_targets=FAR_TARGETS) -> dict[float, float]:
    """Largest TAR at each FAR level (flipud trick from IJB_11.py)."""
    fpr, tpr, _ = roc_curve(labels, scores)
    fpr = np.flipud(fpr)
    tpr = np.flipud(tpr)
    return {
        far: float(tpr[min(range(len(fpr)), key=lambda i: abs(fpr[i] - far))])
        for far in far_targets
    }


# ── Per-model × dataset evaluation ───────────────────────────────────────────

def evaluate_one(
    bundle:             ModelBundle,
    target:             str,
    ijb_dir:            Path,
    use_flip:           bool,
    use_detector_score: bool,
    device:             torch.device,
) -> dict[float, float] | None:

    ds_dir = ijb_dir / target
    meta   = ds_dir / "meta"
    t      = target.lower()

    paths = {
        "tid_mid":  meta / f"{t}_face_tid_mid.txt",
        "pairs":    meta / f"{t}_template_pair_label.txt",
        "img_list": meta / f"{t}_name_5pts_score.txt",
        "img_dir":  ds_dir / "loose_crop",
    }
    for key, p in paths.items():
        if not p.exists():
            print(f"    [skip] missing {key}: {p}")
            return None

    # ── Load protocol ─────────────────────────────────────────────────────────
    templates, medias         = _read_template_media_list(paths["tid_mid"])
    p1, p2, label             = _read_template_pair_list(paths["pairs"])
    img_names, landmarks, faceness = _read_image_list(paths["img_list"])

    print(f"    {target}: {len(img_names):,} images | "
          f"{len(np.unique(templates)):,} templates | "
          f"{len(p1):,} pairs ({label.sum():,} genuine)")

    # ── Extract per-image embeddings ─────────────────────────────────────────
    transform = transforms.Compose([
        transforms.Resize((bundle.height, bundle.width)),
        transforms.ToTensor(),
        transforms.Normalize(bundle.mean, bundle.std),
    ])
    dataset   = _IJBDataset(paths["img_dir"], img_names, landmarks, transform)

    print(f"    extracting embeddings …")
    img_feats = _extract_embeddings(bundle, dataset, use_flip, device)

    # Flip sum: orig + flip halves (identical to IJB_11.py)
    if use_flip:
        D = img_feats.shape[1] // 2
        img_input = img_feats[:, :D] + img_feats[:, D:]
    else:
        img_input = img_feats

    if use_detector_score:
        img_input = img_input * faceness[:, np.newaxis]

    # ── Template aggregation ─────────────────────────────────────────────────
    print(f"    aggregating templates …")
    template_norm_feats, unique_templates = _image2template_feature(
        img_input, templates, medias)

    # ── Score all pairs ───────────────────────────────────────────────────────
    print(f"    scoring {len(p1):,} pairs …")
    scores = _compute_scores(template_norm_feats, unique_templates, p1, p2)

    # ── TAR @ FAR ─────────────────────────────────────────────────────────────
    tar_map = _tar_at_far(scores, label)

    # ── Write to real_detail.csv (same helper as eval_real.py) ───────────────
    # img1/img2 use "template_<tid>" — eval_biometric.py only reads
    # model/dataset/label/cosine_sim so these placeholder strings are fine.
    # fold = -1, correct = -1: IJB has no 10-fold CV or per-pair threshold.
    pairs_for_csv = [
        (f"template_{p1[i]}", f"template_{p2[i]}", int(label[i]))
        for i in range(len(p1))
    ]
    fold_assignments = np.full(len(p1), -1, dtype=int)
    thresh_map       = {-1: 0.0}   # dummy — correct field will be written as 0
    append_to_detail_csv(
        DETAIL_CSV, bundle.name, target,
        pairs_for_csv, scores, fold_assignments, thresh_map,
    )

    # ── Write to real_summary.csv (same helper as eval_real.py) ──────────────
    # accuracy = TAR @ FAR=1e-4 × 100  (standard IJB headline number)
    # std      = 0.0  (no fold variance)
    acc = tar_map[1e-4] * 100
    append_to_summary_csv(SUMMARY_CSV, bundle.name, target, acc, 0.0)

    print(f"    {target}: TAR@FAR=1e-4 = {acc:.2f}%")
    return tar_map


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate all models on IJB-B/C. "
                    "Appends to real_detail.csv and real_summary.csv.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--target", nargs="+", default=["IJBB", "IJBC"],
                        choices=["IJBB", "IJBC"],
                        help="Dataset(s) to evaluate (default: both).")
    parser.add_argument("--ijb-dir", default=str(IJB_DIR), metavar="DIR",
                        help=f"Root containing IJBB/ and IJBC/ (default: {IJB_DIR})")
    parser.add_argument("--model-dir", default=str(MODEL_DIR), metavar="DIR",
                        help=f"Model weights folder (default: {MODEL_DIR})")
    parser.add_argument("--no-library-models", action="store_true",
                        help="Skip edgeface_base and InceptionResnetV1.")
    parser.add_argument("--no-flip", action="store_true",
                        help="Disable horizontal-flip augmentation.")
    parser.add_argument("--no-detector-score", action="store_true",
                        help="Disable faceness score weighting.")
    args = parser.parse_args()

    ijb_dir  = Path(args.ijb_dir)
    use_flip = not args.no_flip
    use_det  = not args.no_detector_score
    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device  : {device}")
    print(f"Targets : {args.target}")
    print(f"Flip    : {use_flip}  |  Detector score: {use_det}")
    print(f"Detail  → {DETAIL_CSV}")
    print(f"Summary → {SUMMARY_CSV}\n")

    specs = list_model_specs(model_dir=args.model_dir,
                             include_library_models=not args.no_library_models)
    if not specs:
        print("[ERROR] No models found.")
        return
    print(f"Found {len(specs)} model(s): {[s.name for s in specs]}\n")

    all_results: dict[str, dict[str, dict]] = {}

    for spec in specs:
        bundle = load_model_by_spec(spec, device)
        if bundle is None:
            continue

        print(f"━━━ {bundle.name} ━━━")
        all_results[bundle.name] = {}

        for target in args.target:
            tar_map = evaluate_one(bundle, target, ijb_dir, use_flip, use_det, device)
            if tar_map is not None:
                all_results[bundle.name][target] = tar_map

        release_model(bundle)
        del bundle
        print()

    # ── Summary table ─────────────────────────────────────────────────────────
    print("\n══ IJB-B / IJB-C  TAR (%) at FAR levels ══")
    for target in args.target:
        has_rows = any(target in r for r in all_results.values())
        if not has_rows:
            continue
        print(f"\n  {target}")
        tbl = PrettyTable(["Model"] + [f"FAR={x}" for x in FAR_TARGETS])
        for model_name, ds_results in all_results.items():
            if target not in ds_results:
                continue
            tbl.add_row([model_name] +
                        [f"{ds_results[target][x] * 100:.2f}" for x in FAR_TARGETS])
        print(tbl)

    print(f"\nDetail  → {DETAIL_CSV}")
    print(f"Summary → {SUMMARY_CSV}")
    print("\nTo compute EER / FMR100 / FMR1000 / ZeroFMR / TAR@FAR metrics, run:")
    print("  python eval_biometric.py --real-only --datasets IJBB IJBC")


if __name__ == "__main__":
    main()
