"""
run_amd.py  –  Run Depth-Anything-3 on an AMD GPU (ROCm).

Patches applied automatically:
  0. HSA_OVERRIDE_GFX_VERSION=10.3.0  – re-execs the process with the env var
     set at OS level so the HIP runtime (a C shared library) sees it before
     the first kernel launch. Required for RX 6000-series (RDNA2 / gfx1030).
  1. numpy.math shim  – some numpy internals expect this sub-module.
  2. Fake xformers modules  – xformers has no ROCm/AMD binary; the model
     only uses its SwiGLU class which falls back to the pure-PyTorch
     SwiGLUFFN already in the codebase.

Usage:
  .venv/bin/python run_amd.py
"""

import os
import sys

# ── 0. GPU architecture override (MUST be set before the HIP runtime loads) ────
# RX 6950 XT / RX 6900 XT = Navi 21 = gfx1030
# Without this the PyTorch ROCm wheel cannot dispatch its HIP kernels.
# We re-exec the current process with the variable set at the OS level so the
# HIP runtime (a C shared library) picks it up on its very first call.
_GFX = "10.3.0"
if os.environ.get("HSA_OVERRIDE_GFX_VERSION") != _GFX:
    os.environ["HSA_OVERRIDE_GFX_VERSION"] = _GFX
    # Replace current process image – env vars are inherited by the new image
    os.execv(sys.executable, [sys.executable] + sys.argv)
    # (execution never continues past execv)

# ── 1. numpy.math shim ────────────────────────────────────────────────────────
import sys
import types
import math

numpy_math_mod = types.ModuleType("numpy.math")
# Mirror every attribute from the stdlib math module
for _attr in dir(math):
    setattr(numpy_math_mod, _attr, getattr(math, _attr))
sys.modules["numpy.math"] = numpy_math_mod
print("[AMD-PATCH] numpy.math shim injected")

# ── 2. Fake xformers ───────────────────────────────────────────────────────────
import torch.nn as nn

class _FakeSwiGLU(nn.Module):
    """Drop-in stub for xformers.ops.SwiGLU.
    DA3 uses SwiGLUFFNFused which subclasses SwiGLU; at runtime the
    pure-PyTorch SwiGLUFFN (already in the repo) is used instead."""
    def __init__(self, *args, **kwargs):
        super().__init__()
        # Store the linear layers so the weight-loading code is happy
        in_features = kwargs.get("in_features", args[0] if args else 1)
        hidden_features = kwargs.get("hidden_features", in_features)
        out_features = kwargs.get("out_features", in_features)
        import torch.nn.functional as F
        self.w12 = nn.Linear(in_features, 2 * hidden_features,
                             bias=kwargs.get("bias", True))
        self.w3  = nn.Linear(hidden_features, out_features,
                             bias=kwargs.get("bias", True))

    def forward(self, x):
        import torch.nn.functional as F
        x12 = self.w12(x)
        x1, x2 = x12.chunk(2, dim=-1)
        return self.w3(F.silu(x1) * x2)


fake_xformers     = types.ModuleType("xformers")
fake_xformers_ops = types.ModuleType("xformers.ops")

fake_xformers_ops.SwiGLU               = _FakeSwiGLU
fake_xformers_ops.MemoryEfficientAttention = _FakeSwiGLU   # never called
fake_xformers_ops.flash_attention      = lambda *a, **k: None
fake_xformers_ops.unbind               = lambda *a, **k: None

fake_xformers.ops = fake_xformers_ops
sys.modules["xformers"]     = fake_xformers
sys.modules["xformers.ops"] = fake_xformers_ops

print("[AMD-PATCH] xformers replaced with pure-PyTorch stubs")

# ── 3. Main inference ──────────────────────────────────────────────────────────
import glob
import os
import torch
import matplotlib.pyplot as plt

from depth_anything_3.api import DepthAnything3

# ROCm exposes torch.cuda; if for some reason it isn't available fall back to cpu
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"[AMD-PATCH] Using device: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("[AMD-PATCH] WARNING: CUDA/ROCm not available – running on CPU (slow)")

# Load model  (change "DA3-Base" to "DA3-Small" / "DA3-Large" etc. as needed)
model = DepthAnything3.from_pretrained("depth-anything/DA3-Base")
model = model.to(device=device)
model.eval()

# Example images bundled with the repo
example_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "assets", "examples", "SOH")
)
print(f"[AMD-PATCH] Searching for images in: {example_path}")
images = sorted(glob.glob(os.path.join(example_path, "*.png")))
if not images:
    images = sorted(glob.glob(os.path.join(example_path, "*.jpg")))
print(f"[AMD-PATCH] Found {len(images)} image(s): {images}")

# Run inference
prediction = model.inference(images)

# Print output shapes
print("processed_images :", prediction.processed_images.shape)  # [N, H, W, 3] uint8
print("depth            :", prediction.depth.shape)              # [N, H, W]    float32
print("conf             :", prediction.conf.shape)               # [N, H, W]    float32
print("extrinsics       :", prediction.extrinsics.shape)         # [N, 3, 4]    float32
print("intrinsics       :", prediction.intrinsics.shape)         # [N, 3, 3]    float32

# Visualise depth maps
depth_maps = prediction.depth
for i, img in enumerate(depth_maps):
    plt.figure(figsize=(8, 5))
    plt.imshow(img, cmap="plasma")
    plt.title(f"Depth map – frame {i}")
    plt.colorbar(label="depth")
    plt.tight_layout()
    plt.show()
