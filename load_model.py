"""
load_model.py
─────────────────────────────────────────────────────────────────────────────
Loads any supported face recognition model and returns a uniform named tuple:

    ModelBundle(model, name, height, width, mean, std)

The decision threshold is intentionally excluded — both eval_real.py and
eval_synth.py determine their own optimal threshold via 10-fold cross-
validation, so a fixed threshold is not needed here.

Supported models
────────────────
  From Pretrained_Models/ folder:
    IResNet_R18_*        (.pth)
    IResNet_R50_* / IDiffFace_CF50 / SFace_IResNet50  (.pth)
    IResNet_R100_*       (.pth)
    VIT_*                (.pth)
    TransFace_*          (.pt)
    SwinFace             (.pt)
    Any other            (.pt  → torch.load)

  From libraries (no local file needed):
    edgeface_base        (torch.hub)
    InceptionResnetV1_VGG    (facenet_pytorch, pretrained on VGGFace2)
    InceptionResnetV1_CASIA  (facenet_pytorch, pretrained on CASIA-WebFace)

  From HuggingFace (auto-downloaded on first use):
    hsface10k            (IResNet-50 + channel-attn, Vec2Face/HSFace, BooBooWu/Vec2Face)
    hsface300k           (IResNet-50 + channel-attn, Vec2Face/HSFace, BooBooWu/Vec2Face)
    hyperface_50k_stylegan   (IResNet-50, HyperFace,           Idiap/HyperFace-50k-StyleGAN)
    langevin_disco_30k       (IResNet-50, Langevin-DisCo,      Idiap/Langevin-DisCo-30k)
    langevin_dispersion_10k  (IResNet-50, Langevin-Dispersion, Idiap/Langevin-Dispersion-10k)

Notes on Idiap Lightning checkpoints (HyperFace / Langevin family)
──────────────────────────────────────────────────────────────────
  All three Idiap models share the same checkpoint format and inference
  convention, so a single loader (_load_hyperface) handles all of them:

  • Architecture: AdaFace Backbone (IR_50) from Pretrained_Models/net.py.
    The Idiap repo is based on AdaFace (mk-minchul/AdaFace) and uses its
    Backbone class (input_layer / body / output_layer).  We import net.py
    directly so weights load without any key remapping.

  • Checkpoint format: PyTorch Lightning .ckpt (weights_only=False required).
    Weights live under ckpt["state_dict"] with keys prefixed by "model." —
    this loader strips that prefix before calling backbone.load_state_dict().

  • Output: Backbone.forward() returns (embedding, norm). _HyperFaceWrapper
    unpacks the tuple and flips channels RGB→BGR transparently, so callers
    use the same RGB pipeline as every other model in this file.

Usage
─────
  from load_model import load_all_models, LIBRARY_MODELS

  # All file-based models in a folder + all library models:
  models = load_all_models(model_dir="Pretrained_Models", device=device)

  # Each entry: ModelBundle with .model .name .height .width .mean .std
  for bundle in models:
      transform = make_transform(bundle.height, bundle.width,
                                 bundle.mean, bundle.std)
"""

import os
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn

import Pretrained_Models.iresnet as iresnet
import Pretrained_Models.iresnet_attn as iresnet_attn
import Pretrained_Models.vit_face as vit
import Pretrained_Models.vit_trans as trans
from Pretrained_Models import swin_subnets
from Pretrained_Models import net as adaface_net

# ── Return type ───────────────────────────────────────────────────────────────

@dataclass
class ModelBundle:
    model:  nn.Module
    name:   str
    height: int
    width:  int
    mean:   list[float]
    std:    list[float]


# ── Defaults shared by most models ───────────────────────────────────────────

_MEAN = [0.5, 0.5, 0.5]
_STD  = [0.5, 0.5, 0.5]
_HW   = 112   # default height = width

# ── Library model names (no local file required) ──────────────────────────────

EDGEFACE_VARIANTS = [
    "edgeface_base",
    "edgeface_s_gamma_05",
    "edgeface_xs_gamma_06",
    "edgeface_xxs",
]

# HSFace models available on HuggingFace (BooBooWu/Vec2Face).
# All are IResNet-50 weights trained with ArcFace on synthetic HSFace datasets.
HSFACE_VARIANTS = [
    "hsface10k",
    "hsface300k",
]

# HyperFace / Langevin models available on HuggingFace (Idiap/*).
# IResNet-50 weights stored as PyTorch Lightning .ckpt checkpoints.
# Filenames confirmed from the repo file listings; see _HYPERFACE_REPO.
HYPERFACE_VARIANTS = [
    "hyperface_50k_stylegan",   # Idiap/HyperFace-50k-StyleGAN
    "langevin_disco_30k",       # Idiap/Langevin-DisCo-30k
    "langevin_dispersion_10k",  # Idiap/Langevin-Dispersion-10k
]

# Mapping from our canonical name → (HuggingFace repo id, filename on hub).
# Filenames confirmed directly from the repo file listings:
#   https://huggingface.co/Idiap/HyperFace-50k-StyleGAN/blob/main/HyperFace_50k_StyleGAN.ckpt
#   https://huggingface.co/Idiap/Langevin-DisCo-30k/blob/main/Langevin-DisCo-30k.ckpt
#   https://huggingface.co/Idiap/Langevin-Dispersion-10k/blob/main/Langevin-Dispersion-10k.ckpt
_HYPERFACE_REPO = {
    "hyperface_50k_stylegan":  ("Idiap/HyperFace-50k-StyleGAN",  "HyperFace_50k_StyleGAN.ckpt"),
    "langevin_disco_30k":      ("Idiap/Langevin-DisCo-30k",      "Langevin-DisCo-30k.ckpt"),
    "langevin_dispersion_10k": ("Idiap/Langevin-Dispersion-10k", "Langevin-Dispersion-10k.ckpt"),
}


LIBRARY_MODELS = (
    EDGEFACE_VARIANTS
    + HSFACE_VARIANTS
    + HYPERFACE_VARIANTS
    + ["InceptionResnetV1_VGG", "InceptionResnetV1_CASIA"]
)

# ── Individual loaders ────────────────────────────────────────────────────────

def _load_edgeface(variant: str, device: torch.device) -> ModelBundle:
    model = torch.hub.load(
        "otroshi/edgeface", variant, source="github", pretrained=True
    ).to(device).eval()
    print(f"[LOADED] {variant}")
    return ModelBundle(model, variant, _HW, _HW, _MEAN, _STD)


def _load_facenet(
    dataset: Literal["vggface2", "casia-webface"],
    device: torch.device,
) -> ModelBundle:
    from facenet_pytorch import InceptionResnetV1
    model = InceptionResnetV1(pretrained=dataset).to(device).eval()
    name  = "InceptionResnetV1_VGG" if dataset == "vggface2" else "InceptionResnetV1_CASIA"
    print(f"[LOADED] {name}")
    # FaceNet expects 160×160
    return ModelBundle(model, name, 160, 160, _MEAN, _STD)


def _load_hsface(variant: str, device: torch.device) -> ModelBundle:
    """
    Load an HSFace FR model (IResNet-50 + ArcFace) from HuggingFace.

    Weights are hosted at:
        https://huggingface.co/BooBooWu/Vec2Face  (fr_weights/<variant>.pth)

    huggingface_hub caches the file locally on first download, so subsequent
    calls are instant.  The file name on the hub is e.g. "fr_weights/hsface10k.pth".
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise ImportError(
            "huggingface_hub is required to load HSFace models. "
            "Install it with:  pip install huggingface_hub"
        ) from exc

    hub_filename = f"fr_weights/{variant}.pth"
    local_path = hf_hub_download(
        repo_id="BooBooWu/Vec2Face",
        filename=hub_filename,
    )
    # HSFace models use an attention-augmented IResNet-50 (see Pretrained_Models/iresnet_attn.py)
    model = iresnet_attn.iresnet50_attn()
    state = torch.load(local_path, map_location=device, weights_only=True)
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    print(f"[LOADED] {variant}")
    return ModelBundle(model, variant, _HW, _HW, _MEAN, _STD)


class _HyperFaceWrapper(nn.Module):
    """
    Thin wrapper around the AdaFace Backbone that:
      1. Flips channels RGB→BGR before the forward pass (the Idiap/AdaFace
         models expect BGR input).
      2. Unpacks the (embedding, norm) tuple that Backbone.forward() always
         returns, so every ModelBundle yields a plain embedding tensor.
    """
    def __init__(self, backbone: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x arrives in RGB order (our standard pipeline); flip to BGR
        x_bgr = x[:, [2, 1, 0], :, :]
        out = self.backbone(x_bgr)
        # Backbone.forward() returns (embedding, norm) — extract embedding
        if isinstance(out, (tuple, list)):
            return out[0]
        return out


def _load_hyperface(variant: str, device: torch.device) -> ModelBundle:
    """
    Load a HyperFace / Langevin FR model from HuggingFace.

    The Idiap codebase is derived from AdaFace (mk-minchul/AdaFace) and uses
    the Backbone class from Pretrained_Models/net.py (input_layer / body /
    output_layer).  We instantiate it via adaface_net.IR_50() so weights load
    without any key remapping.

    Backbone.forward() returns (embedding, norm); _HyperFaceWrapper unpacks
    this to a plain tensor consistent with every other ModelBundle.

    Checkpoint format: PyTorch Lightning .ckpt.  Weights live under
    ckpt["state_dict"] with all keys prefixed by "model." — stripped here.
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise ImportError(
            "huggingface_hub is required to load HyperFace/Langevin models. "
            "Install it with:  pip install huggingface_hub"
        ) from exc

    repo_id, hub_filename = _HYPERFACE_REPO[variant]
    local_path = hf_hub_download(repo_id=repo_id, filename=hub_filename)

    # Lightning checkpoint: weights_only=False because the file is a pickled
    # dict (not a plain state-dict tensor file).
    ckpt = torch.load(local_path, map_location=device, weights_only=False)

    # Strip the "model." prefix added by the LightningModule wrapper.
    state = {
        k[len("model."):]: v
        for k, v in ckpt["state_dict"].items()
        if k.startswith("model.")
    }

    # Use the real AdaFace Backbone from Pretrained_Models/net.py — keys
    # match the checkpoint exactly, no remapping needed.
    backbone = adaface_net.IR_50(input_size=(112, 112))
    backbone.load_state_dict(state, strict=True)
    model = _HyperFaceWrapper(backbone).to(device).eval()
    print(f"[LOADED] {variant}")
    return ModelBundle(model, variant, _HW, _HW, _MEAN, _STD)



def _load_iresnet(path: str, depth: int, device: torch.device) -> ModelBundle:
    name = os.path.splitext(os.path.basename(path))[0]
    # Try loading as a plain state dict first (standard .pth convention).
    # Some .pt files are torch.fx compiled full-model objects — if state-dict
    # loading fails, fall back to loading the full object directly.
    try:
        builder = {18: iresnet.iresnet18, 50: iresnet.iresnet50, 100: iresnet.iresnet100}[depth]
        model   = builder()
        state   = torch.load(path, map_location=device, weights_only=False)
        # If the loaded object is already a module (compiled model), use it directly
        if isinstance(state, torch.nn.Module):
            model = state
        else:
            model.load_state_dict(state, strict=True)
        model.to(device).eval()
        print(f"[LOADED] {name}")
        return ModelBundle(model, name, _HW, _HW, _MEAN, _STD)
    except Exception:
        # Last resort: treat as a fully serialised model object
        model = torch.load(path, map_location=device, weights_only=False)
        if isinstance(model, torch.nn.Module):
            model = model.eval()
            print(f"[LOADED] {name}")
            return ModelBundle(model, name, _HW, _HW, _MEAN, _STD)
        raise


def _load_vit(path: str, device: torch.device) -> ModelBundle:
    model = vit.ViT_face(
        image_size=112, patch_size=8, loss_type="CosFace", GPU_ID=device,
        num_class=93431, dim=512, depth=20, heads=8, mlp_dim=2048,
        dropout=0.1, emb_dropout=0.1,
    )
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True), strict=True)
    model.to(device).eval()
    name = os.path.splitext(os.path.basename(path))[0]
    print(f"[LOADED] {name}")
    return ModelBundle(model, name, _HW, _HW, _MEAN, _STD)


def _load_transface(path: str, device: torch.device) -> ModelBundle:
    name = os.path.splitext(os.path.basename(path))[0]
    # TransFace comes in two sizes (from the original paper / official repo):
    #   S (Small): embed_dim=512, depth=12, num_heads=8
    #   L (Large): embed_dim=768, depth=24, num_heads=12
    # Detect variant from filename; default to S if ambiguous.
    if "_L" in name or name.endswith("_L"):
        embed_dim, depth, num_heads = 768, 24, 12
    else:
        embed_dim, depth, num_heads = 512, 12, 8
    model = trans.VisionTransformer(
        img_size=112, patch_size=9, num_classes=512, embed_dim=embed_dim,
        depth=depth, num_heads=num_heads, drop_path_rate=0.05,
        norm_layer="ln", mask_ratio=0.0,
    )
    model.load_state_dict(torch.load(path, map_location=device, weights_only=False), strict=True)
    model.to(device).eval()
    print(f"[LOADED] {name}  (TransFace-{'L' if embed_dim == 768 else 'S'})")
    return ModelBundle(model, name, _HW, _HW, _MEAN, _STD)


def _load_swinface(path: str, device: torch.device) -> ModelBundle:
    cfg   = swin_subnets.SwinFaceCfg()
    model = swin_subnets.build_model(cfg)
    ckpt  = torch.load(path, map_location=device, weights_only=False)
    model.backbone.load_state_dict(ckpt["state_dict_backbone"])
    model.fam.load_state_dict(ckpt["state_dict_fam"])
    model.tss.load_state_dict(ckpt["state_dict_tss"])
    model.om.load_state_dict(ckpt["state_dict_om"])
    # Return only the face recognition embedding, not the full 43-output dict
    model.set_output_type("Recognition")
    model.to(device).eval()
    print("[LOADED] SwinFace")
    return ModelBundle(model, "SwinFace", _HW, _HW, _MEAN, _STD)


def _load_generic_pt(path: str, device: torch.device) -> ModelBundle:
    # weights_only=False required: these files contain a full serialised model
    # object, not just a state dict (triggered by PyTorch 2.6 default change).
    model = torch.load(path, map_location=device, weights_only=False).eval()
    name  = os.path.splitext(os.path.basename(path))[0]
    print(f"[LOADED] {name}")
    return ModelBundle(model, name, _HW, _HW, _MEAN, _STD)


# ── Router for a single file ──────────────────────────────────────────────────

def load_model_file(path: str, device: torch.device) -> ModelBundle:
    """
    Load one model file (.pt or .pth) by inspecting its filename.
    Raises FileNotFoundError if the file does not exist.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Model file not found: {path}")

    name = os.path.splitext(os.path.basename(path))[0]
    ext  = os.path.splitext(path)[1].lower()

    if ext == ".pth":
        if "IResNet_R18" in name:
            return _load_iresnet(path, 18, device)
        if any(x in name for x in ("IResNet_R50", "IDiffFace_CF50", "SFace_IResNet50")):
            return _load_iresnet(path, 50, device)
        if "IResNet_R100" in name:
            return _load_iresnet(path, 100, device)
        if "VIT" in name:
            return _load_vit(path, device)
        # Check for locally cached HSFace weights (e.g. manually placed .pth files)
        if name.lower() in HSFACE_VARIANTS:
            model = iresnet_attn.iresnet50_attn()
            state = torch.load(path, map_location=device, weights_only=True)
            model.load_state_dict(state, strict=True)
            model.to(device).eval()
            print(f"[LOADED] {name}")
            return ModelBundle(model, name, _HW, _HW, _MEAN, _STD)
        # Unknown .pth — try generic load
        return _load_generic_pt(path, device)

    if ext == ".pt":
        if "SwinFace" in name:
            return _load_swinface(path, device)
        if "TransFace" in name:
            return _load_transface(path, device)
        return _load_generic_pt(path, device)

    raise ValueError(f"Unsupported extension '{ext}' for {path}")


# ── Bulk loader ───────────────────────────────────────────────────────────────

def load_all_models(
    model_dir: str,
    device: torch.device,
    include_library_models: bool = True,
    skip_on_error: bool = True,
) -> list[ModelBundle]:
    """
    Load every .pt / .pth file in model_dir, then (optionally) all library
    models (EdgeFace, FaceNet VGG, FaceNet CASIA, HSFace10k, HSFace300k).

    Parameters
    ──────────
    model_dir             : folder containing .pt / .pth weight files
    device                : torch device
    include_library_models: whether to also load edgeface + facenets + hsface
    skip_on_error         : if True, print a warning and continue on failure;
                            if False, re-raise the exception

    Returns
    ───────
    list[ModelBundle]  – one entry per successfully loaded model
    """
    bundles: list[ModelBundle] = []

    # ── File-based models ─────────────────────────────────────────────────────
    candidates = sorted(
        f for f in os.listdir(model_dir)
        if os.path.splitext(f)[1].lower() in (".pt", ".pth")
    )

    for filename in candidates:
        path = os.path.join(model_dir, filename)
        try:
            bundles.append(load_model_file(path, device))
        except Exception as exc:
            if skip_on_error:
                print(f"[WARN] Skipped {filename}: {exc}")
            else:
                raise

    # ── Library models ────────────────────────────────────────────────────────
    # HyperFace/Langevin variants are listed first so that any loading errors
    # surface early in the run rather than after all other models have loaded.
    if include_library_models:
        for loader_fn in [
            *[lambda v=v: _load_hyperface(v, device) for v in HYPERFACE_VARIANTS],
            *[lambda v=v: _load_edgeface(v, device) for v in EDGEFACE_VARIANTS],
            *[lambda v=v: _load_hsface(v, device) for v in HSFACE_VARIANTS],
            lambda: _load_facenet("vggface2",     device),
            lambda: _load_facenet("casia-webface", device),
        ]:
            try:
                bundles.append(loader_fn())
            except Exception as exc:
                if skip_on_error:
                    print(f"[WARN] Skipped library model: {exc}")
                else:
                    raise

    return bundles


# ── Spec-based API (memory-safe, one model at a time) ─────────────────────────

@dataclass
class ModelSpec:
    """Lightweight descriptor — no weights loaded yet."""
    name: str
    path: str | None  # None for library models


def list_model_specs(
    model_dir: str,
    include_library_models: bool = True,
) -> list[ModelSpec]:
    """
    Return a list of ModelSpec objects without loading any weights.
    Pass each spec to load_model_by_spec() one at a time inside your eval loop,
    deleting the bundle and calling release_model() between iterations to keep
    only one model on GPU at a time.

    Ordering: HyperFace/Langevin variants are always first so that any
    loading errors surface at the very start of the run.  File-based models
    follow, then the remaining library models (EdgeFace, HSFace, FaceNet).
    """
    # ── 1. HyperFace / Langevin first ─────────────────────────────────────────
    hyperface_specs: list[ModelSpec] = []
    if include_library_models:
        for variant in HYPERFACE_VARIANTS:
            hyperface_specs.append(ModelSpec(name=variant, path=None))

    # ── 2. File-based models ───────────────────────────────────────────────────
    file_specs: list[ModelSpec] = []
    candidates = sorted(
        f for f in os.listdir(model_dir)
        if os.path.splitext(f)[1].lower() in (".pt", ".pth")
    )
    for filename in candidates:
        file_specs.append(ModelSpec(
            name=os.path.splitext(filename)[0],
            path=os.path.join(model_dir, filename),
        ))

    # ── 3. Remaining library models ────────────────────────────────────────────
    other_library_specs: list[ModelSpec] = []
    if include_library_models:
        for variant in EDGEFACE_VARIANTS:
            other_library_specs.append(ModelSpec(name=variant, path=None))
        for variant in HSFACE_VARIANTS:
            other_library_specs.append(ModelSpec(name=variant, path=None))
        other_library_specs.append(ModelSpec(name="InceptionResnetV1_VGG",   path=None))
        other_library_specs.append(ModelSpec(name="InceptionResnetV1_CASIA", path=None))

    return hyperface_specs + file_specs + other_library_specs


def load_model_by_spec(
    spec: ModelSpec,
    device: torch.device,
    skip_on_error: bool = True,
) -> ModelBundle | None:
    """Load a single model from a ModelSpec. Returns None on failure."""
    try:
        if spec.path is not None:
            return load_model_file(spec.path, device)
        # Library models identified by name
        if spec.name in EDGEFACE_VARIANTS:
            return _load_edgeface(spec.name, device)
        if spec.name in HSFACE_VARIANTS:
            return _load_hsface(spec.name, device)
        if spec.name in HYPERFACE_VARIANTS:
            return _load_hyperface(spec.name, device)
        if spec.name == "InceptionResnetV1_VGG":
            return _load_facenet("vggface2", device)
        if spec.name == "InceptionResnetV1_CASIA":
            return _load_facenet("casia-webface", device)
        raise ValueError(f"Unknown library model: {spec.name}")
    except Exception as exc:
        if skip_on_error:
            print(f"[WARN] Skipped {spec.name}: {exc}")
            return None
        raise


def release_model(bundle: ModelBundle) -> None:
    """
    Move model off GPU, delete it, and free the CUDA cache.
    Call this immediately after finishing all datasets for one model.
    """
    bundle.model.cpu()
    del bundle.model
    torch.cuda.empty_cache()
