"""Quick sanity check: GPU, SSL4EO backbone, model forward pass."""

import sys

import torch

print(f"Python:          {sys.executable}")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available:  {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU device:      {torch.cuda.get_device_name(0)}")
    props = torch.cuda.get_device_properties(0)
    mem = getattr(props, "total_memory", None) or getattr(props, "total_mem", 0)
    print(f"GPU memory:      {mem / 1e9:.1f} GB")
else:
    print("WARNING: No CUDA GPU detected. Training will be very slow on CPU.")

from models import build_model, verify_ssl4eo_backbone

print("\n--- SSL4EO-S12 backbone check ---")
ok, source, shape = verify_ssl4eo_backbone(14, strict=True)
print(f"SSL4EO OK: {ok} | source: {source} | conv1: {shape}")
if source == "imagenet_fallback":
    print("FAIL: ImageNet fallback detected — fix SSL4EO before training.")
    sys.exit(1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for name in ("taunet_resnet50", "taunet", "resunext"):
    two_head = name in ("taunet", "taunet_resnet50")
    use_ssag = name != "unet_resnet50"
    model = build_model(
        name, in_channels=14, two_head=two_head, use_ssag=use_ssag, ssl4eo_strict=True,
    ).to(device)
    backbone_src = getattr(model, "backbone_source", "n/a")
    x = torch.randn(2, 14, 256, 256, device=device)
    with torch.no_grad():
        out = model(x)
    if isinstance(out, dict):
        seg_shape = out["seg"].shape
    else:
        seg_shape = out.shape
    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  {name:22s} | seg={seg_shape} | {params:.1f}M params | backbone={backbone_src} | OK")
    del model, x, out
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

print("\nAll checks passed (CUDA + SSL4EO + forward).")
