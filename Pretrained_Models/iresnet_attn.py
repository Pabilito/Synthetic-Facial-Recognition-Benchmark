"""
iresnet_attn.py
─────────────────────────────────────────────────────────────────────────────
Attention-augmented IResNet-50 used by the Vec2Face training pipeline
(HaiyuWu/SOTA-Face-Recognition-Train-and-Test).

The architecture differs from plain iresnet.py in one way only:
the *first block of each of the 4 stages* (layer1[0] … layer4[0]) carries a
Conv2d-based squeeze-and-excite channel attention gate on its residual branch:

    out = bn3(conv2(...))       # residual  shape: (B, C, H, W)
    out = ChannelAttn(out)      # ← Conv2d SE gate, bias-free, reduction=16
    out = out + identity        # skip connection

Checkpoint evidence (from actual saved weight shapes):
    layer1.0.attn.fc1.weight  →  [4,   64,  1, 1]  (Conv2d, C_out=C//16=4)
    layer1.0.attn.fc2.weight  →  [64,  4,   1, 1]  (Conv2d, C_out=C=64)
    layer2.0.attn.fc1.weight  →  [8,   128, 1, 1]
    layer2.0.attn.fc2.weight  →  [128, 8,   1, 1]
    layer3.0.attn.fc1.weight  →  [16,  256, 1, 1]
    layer3.0.attn.fc2.weight  →  [256, 16,  1, 1]
    layer4.0.attn.fc1.weight  →  [32,  512, 1, 1]
    layer4.0.attn.fc2.weight  →  [512, 32,  1, 1]

The keys are named fc1/fc2 because the Vec2Face repo uses Conv2d(kernel_size=1)
— equivalent to a channel-wise linear projection — but names the parameters
fc1/fc2 by convention.

All other blocks are unchanged plain IBasicBlocks from iresnet.py.

Public API
──────────
    from Pretrained_Models.iresnet_attn import iresnet50_attn

    model = iresnet50_attn()
    model.load_state_dict(torch.load("hsface10k.pth", weights_only=True))
"""

import types

import torch
import torch.nn as nn

import Pretrained_Models.iresnet as _base


# ── Channel-attention module ──────────────────────────────────────────────────

class ChannelAttn(nn.Module):
    """
    Conv2d-based squeeze-and-excite channel attention.

    The weights are 4-dimensional [C_out, C_in, 1, 1] matching the checkpoint.
    Parameters are named fc1/fc2 (Conv2d with kernel_size=1) to match the
    exact key names in the saved state-dict: layerN.0.attn.fc1.weight etc.
    bias=False throughout (no bias keys present in the checkpoint).

    reduction=16 — confirmed by checkpoint shapes:
        layer1: C=64  → C//16=4   → [4,  64,  1,1] and [64,  4,  1,1]
        layer2: C=128 → C//16=8   → [8,  128, 1,1] and [128, 8,  1,1]
        layer3: C=256 → C//16=16  → [16, 256, 1,1] and [256, 16, 1,1]
        layer4: C=512 → C//16=32  → [32, 512, 1,1] and [512, 32, 1,1]
    """
    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1, bias=False)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Global average pool: (B, C, H, W) → (B, C, 1, 1)
        gap  = x.mean(dim=[2, 3], keepdim=True)
        gate = torch.sigmoid(self.fc2(torch.relu(self.fc1(gap))))
        return x * gate


# ── Patched forward for attention blocks ─────────────────────────────────────

def _attn_block_forward(self, x: torch.Tensor) -> torch.Tensor:
    """
    Replacement forward for IBasicBlock instances that carry a .attn module.
    Mirrors the standard IBasicBlock forward but inserts the attention gate
    on the residual branch before the skip-connection add.
    """
    identity = x
    out = self.bn1(x)
    out = self.conv1(out)
    out = self.bn2(out)
    out = self.prelu(out)
    out = self.conv2(out)
    out = self.bn3(out)
    out = self.attn(out)          # channel-attention gate
    if self.downsample is not None:
        identity = self.downsample(x)
    out += identity
    return out


# ── Factory ───────────────────────────────────────────────────────────────────

# Channel widths of block-0 in each stage for IResNet-50
_STAGE_CHANNELS = {
    "layer1": 64,
    "layer2": 128,
    "layer3": 256,
    "layer4": 512,
}


def iresnet50_attn() -> nn.Module:
    """
    Return an IResNet-50 with ChannelAttn inserted at block 0 of each stage.

    Built on top of the stock iresnet50(); only the four attention blocks are
    patched in-place.  All other weights and keys are identical to iresnet50().
    The eight additional keys are:
        layerN.0.attn.fc{1,2}.weight  for N in {1, 2, 3, 4}
    (Conv2d weights with shape [C_out, C_in, 1, 1])
    """
    model = _base.iresnet50()

    for layer_name, channels in _STAGE_CHANNELS.items():
        block = getattr(model, layer_name)[0]
        block.attn = ChannelAttn(channels)
        block.forward = types.MethodType(_attn_block_forward, block)

    return model
