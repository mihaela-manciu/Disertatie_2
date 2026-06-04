"""Quick sanity check: GPU availability, model instantiation, forward pass."""

import torch

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available:  {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU device:      {torch.cuda.get_device_name(0)}")
    print(f"GPU memory:      {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
else:
    print("WARNING: No CUDA GPU detected. Training will be very slow on CPU.")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from models import build_model

for name in ("taunet_resnet50", "taunet", "resunext"):
    two_head = name in ("taunet", "taunet_resnet50")
    use_ssag = name != "unet_resnet50"
    model = build_model(name, in_channels=14, two_head=two_head, use_ssag=use_ssag).to(device)
    x = torch.randn(2, 14, 256, 256, device=device)
    with torch.no_grad():
        out = model(x)
    if isinstance(out, dict):
        seg_shape = out["seg"].shape
    else:
        seg_shape = out.shape
    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  {name:22s} | seg={seg_shape} | {params:.1f}M params | OK")
    del model, x, out
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

print("\nAll models passed forward-pass check.")
