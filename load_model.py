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
import Pretrained_Models.vit_face as vit
import Pretrained_Models.vit_trans as trans
from Pretrained_Models import swin_subnets

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
    "edgeface_xs_q",
    "edgeface_xs_gamma_06",
    "edgeface_xxs",
    "edgeface_xxs_q",
]

# Library model names (no local file required)
LIBRARY_MODELS = EDGEFACE_VARIANTS + ["InceptionResnetV1_VGG", "InceptionResnetV1_CASIA"]

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


def _load_iresnet(path: str, depth: int, device: torch.device) -> ModelBundle:
    builder = {18: iresnet.iresnet18, 50: iresnet.iresnet50, 100: iresnet.iresnet100}[depth]
    model   = builder()
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True), strict=True)
    model.to(device).eval()
    name = os.path.splitext(os.path.basename(path))[0]
    print(f"[LOADED] {name}")
    return ModelBundle(model, name, _HW, _HW, _MEAN, _STD)


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
    model = trans.VisionTransformer(
        img_size=112, patch_size=9, num_classes=512, embed_dim=512,
        depth=12, num_heads=8, drop_path_rate=0.05, norm_layer="ln", mask_ratio=0.0,
    )
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True), strict=True)
    model.to(device).eval()
    name = os.path.splitext(os.path.basename(path))[0]
    print(f"[LOADED] {name}")
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
    models (EdgeFace, FaceNet VGG, FaceNet CASIA).

    Parameters
    ──────────
    model_dir             : folder containing .pt / .pth weight files
    device                : torch device
    include_library_models: whether to also load edgeface + facenets
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
    if include_library_models:
        for loader_fn in [
            *[lambda v=v: _load_edgeface(v, device) for v in EDGEFACE_VARIANTS],
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
    """
    specs: list[ModelSpec] = []

    candidates = sorted(
        f for f in os.listdir(model_dir)
        if os.path.splitext(f)[1].lower() in (".pt", ".pth")
    )
    for filename in candidates:
        specs.append(ModelSpec(
            name=os.path.splitext(filename)[0],
            path=os.path.join(model_dir, filename),
        ))

    if include_library_models:
        for variant in EDGEFACE_VARIANTS:
            specs.append(ModelSpec(name=variant, path=None))
        specs.append(ModelSpec(name="InceptionResnetV1_VGG",   path=None))
        specs.append(ModelSpec(name="InceptionResnetV1_CASIA", path=None))

    return specs


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
