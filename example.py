"""
Minimum example: Load EdgeFace-base and evaluate on LFW, CALFW, CPLFW.

Expected structure (relative to script location):
  data/val/
    lfw_112x112/       0.bmp, 1.bmp, ...
    calfw_112x112/
    cplfw_112x112/
    lfw_ann.txt        format: label img1_path img2_path
    calfw_ann.txt
    cplfw_ann.txt
"""

import os
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from sklearn.model_selection import KFold

# -- Paths ------------------------------------------------------------
BASE = os.path.join(os.path.dirname(__file__), "data", "val")

# -- Model ------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = torch.hub.load(
    "otroshi/edgeface",
    "edgeface_base",
    source="github",
    pretrained=True,
)
model = model.to(device).eval()

# -- Transform --------------------------------------------------------
transform = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])


# -- Dataset ----------------------------------------------------------
class PairDataset(Dataset):
    def __init__(self, pairs, img_dir, transform):
        self.pairs = pairs      # list of (img1_path, img2_path, label)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        p1, p2, label = self.pairs[idx]
        img1 = self.transform(Image.open(os.path.join(self.img_dir, p1)).convert("RGB"))
        img2 = self.transform(Image.open(os.path.join(self.img_dir, p2)).convert("RGB"))
        return img1, img2, int(label)


# -- Annotation parser (format: label img1_path img2_path) ------------
def parse_simple_ann(ann_txt):
    pairs = []
    with open(ann_txt) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 3:
                label, p1, p2 = parts
                pairs.append((p1, p2, int(label)))
    return pairs


# -- Embedding extraction ---------------------------------------------
@torch.no_grad()
def extract_embeddings(loader):
    e1_list, e2_list, labels = [], [], []
    for img1, img2, label in loader:
        e1_list.append(F.normalize(model(img1.to(device)), dim=1).cpu())
        e2_list.append(F.normalize(model(img2.to(device)), dim=1).cpu())
        labels.extend(label.tolist())
    return torch.cat(e1_list), torch.cat(e2_list), np.array(labels)


# -- 10-fold evaluation -----------------------------------------------
def eval_10fold(emb1, emb2, labels, n_folds=10):
    cos_sim = (emb1 * emb2).sum(dim=1).numpy()
    accs = []
    for train_idx, test_idx in KFold(n_splits=n_folds, shuffle=False).split(cos_sim):
        best_thresh, best_acc = 0.0, 0.0
        for thresh in np.arange(-1.0, 1.0, 0.005):
            acc = ((cos_sim[train_idx] >= thresh) == labels[train_idx]).mean()
            if acc > best_acc:
                best_acc, best_thresh = acc, thresh
        preds = (cos_sim[test_idx] >= best_thresh).astype(int)
        accs.append((preds == labels[test_idx]).mean())
    return np.mean(accs) * 100, np.std(accs) * 100


# -- Dataset config ---------------------------------------------------
DATASETS = {
    "LFW": {
        "ann": os.path.join(BASE, "lfw_ann.txt"),
        "img_dir": BASE,
    },
    "CALFW": {
        "ann": os.path.join(BASE, "calfw_ann.txt"),
        "img_dir": BASE,
    },
    "CPLFW": {
        "ann": os.path.join(BASE, "cplfw_ann.txt"),
        "img_dir": BASE,
    },
}

# -- Run evaluation ---------------------------------------------------
for name, cfg in DATASETS.items():
    pairs = parse_simple_ann(cfg["ann"])
    dataset = PairDataset(pairs, cfg["img_dir"], transform)
    loader = DataLoader(dataset, batch_size=64, num_workers=4, pin_memory=True)

    emb1, emb2, labels = extract_embeddings(loader)
    acc, std = eval_10fold(emb1, emb2, labels)

    print(f"{name:6s}: {acc:.2f} +/- {std:.2f}%")
