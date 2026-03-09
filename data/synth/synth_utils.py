"""
synth_utils.py
─────────────────────────────────────────────────────────────────────────────
Shared constants, path helpers, and the annotation-file parser used by both
generate_ann.py  and  eval_synth.py.
"""

from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).parent.parent.parent 
SYNTH_ROOT = SCRIPT_DIR / "data" / "synth"
EVAL_DIR   = SCRIPT_DIR / "Evaluation"
CSV_OUT    = EVAL_DIR / "synth.csv"

# ── Pair-generation settings ──────────────────────────────────────────────────

# Total pairs per dataset (must be divisible by 10 for clean 10-fold split).
N_PAIRS = 6000   # 3 000 same-identity + 3 000 different-identity
SEED    = 42

# ── Image extensions ──────────────────────────────────────────────────────────

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}

# ── Annotation helpers ────────────────────────────────────────────────────────

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
