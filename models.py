import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


def _get_groups(out_channels, max_groups=32):
    groups = min(max_groups, max(1, out_channels // 4))
    while out_channels % groups != 0 and groups > 1:
        groups -= 1
    return groups


def init_multispectral_stem(conv_layer, pretrained_rgb_weight, in_channels=13):
    with torch.no_grad():
        if pretrained_rgb_weight is not None:
            rgb_mean = pretrained_rgb_weight.mean(dim=1, keepdim=True)
            repeated = rgb_mean.repeat(1, in_channels, 1, 1)
            conv_layer.weight.copy_(repeated / in_channels)
        else:
            nn.init.kaiming_normal_(conv_layer.weight, mode="fan_out", nonlinearity="relu")


# SSL4EO-S12 band order (13 bands): B01 B02 B03 B04 B05 B06 B07 B08 B8A B09 B10 B11 B12
# MARIDA band order (11 bands):     B01 B02 B03 B04 B05 B06 B07 B08 B8A          B11 B12
_SSL4EO_TO_MARIDA = [0, 1, 2, 3, 4, 5, 6, 7, 8, 11, 12]  # indices into 13-band SSL4EO
_SSL4EO_RESNET50_URL = (
    "https://hf.co/torchgeo/resnet50_sentinel2_all_moco/resolve/"
    "da4f3c9dbe09272eb902f3b37f46635fa4726879/resnet50_sentinel2_all_moco-df8b932e.pth"
)
_SSL4EO_CACHE_NAME = "resnet50_sentinel2_all_moco.pth"


def _ssl4eo_cache_dir():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "pretrained_weights")


def _fetch_ssl4eo_state_dict():
    """
    Load SSL4EO-S12 MoCo ResNet-50 weights.
    1) torchgeo API (fast when GDAL stack works)
    2) direct HuggingFace download (avoids rasterio/GDAL DLL issues on Windows)
    """
    try:
        from torchgeo.models import ResNet50_Weights as GeoWeights
        from torchgeo.models import resnet50 as geo_resnet50

        geo_model = geo_resnet50(weights=GeoWeights.SENTINEL2_ALL_MOCO)
        return geo_model.state_dict(), "torchgeo"
    except Exception as torchgeo_err:
        print(f"[backbone] torchgeo path failed ({torchgeo_err}); trying HF cache…")
        cache_dir = _ssl4eo_cache_dir()
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = os.path.join(cache_dir, _SSL4EO_CACHE_NAME)
        try:
            if os.path.isfile(cache_path):
                try:
                    state = torch.load(cache_path, map_location="cpu", weights_only=True)
                except TypeError:
                    state = torch.load(cache_path, map_location="cpu")
            else:
                state = torch.hub.load_state_dict_from_url(
                    _SSL4EO_RESNET50_URL,
                    model_dir=cache_dir,
                    file_name=_SSL4EO_CACHE_NAME,
                    map_location="cpu",
                )
            if "conv1.weight" not in state:
                raise KeyError("downloaded checkpoint missing conv1.weight")
            return state, "hf_cache"
        except Exception as hf_err:
            raise RuntimeError(
                "SSL4EO-S12 weights could not be loaded. "
                f"torchgeo error: {torchgeo_err}; HF error: {hf_err}"
            ) from hf_err


def verify_ssl4eo_backbone(in_channels=14, strict=True):
    """Return (ok, source, conv1_shape). Raises if strict=True and SSL4EO unavailable."""
    try:
        state, source = _fetch_ssl4eo_state_dict()
        shape = tuple(state["conv1.weight"].shape)
        ok = shape[1] == 13
        if not ok:
            msg = f"SSL4EO conv1 has {shape[1]} channels, expected 13"
            if strict:
                raise RuntimeError(msg)
            return False, source, shape
        return True, source, shape
    except Exception as exc:
        if strict:
            raise
        return False, "failed", None


def _load_ssl4eo_resnet50(in_channels=14, *, strict=True):
    """
    Load ResNet-50 with SSL4EO-S12 Sentinel-2 MoCo pretrained weights (13-band).
    Remap conv1 to match MARIDA's 11 bands + 3 index channels (NDVI, FDI, PI).
    """
    try:
        s2_state, source = _fetch_ssl4eo_state_dict()
        s2_conv1 = s2_state["conv1.weight"]
        print(
            f"[backbone] Loaded SSL4EO-S12 MoCo via {source} "
            f"(conv1 shape: {tuple(s2_conv1.shape)})"
        )
    except Exception as e:
        if strict:
            raise RuntimeError(
                "SSL4EO-S12 is required but unavailable. "
                "Fix torchgeo/GDAL or allow ImageNet with backbone='imagenet'. "
                f"Details: {e}"
            ) from e
        print(f"[backbone] SSL4EO-S12 unavailable ({e}), falling back to ImageNet stem init")
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        stem_conv = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        init_multispectral_stem(stem_conv, resnet.conv1.weight, in_channels)
        return resnet, stem_conv, "imagenet_fallback"

    resnet = models.resnet50(weights=None)
    rn_state = resnet.state_dict()

    mapped_state = {}
    for k, v in s2_state.items():
        if k == "conv1.weight":
            continue
        if k == "fc.weight" or k == "fc.bias":
            if k in rn_state and rn_state[k].shape == v.shape:
                mapped_state[k] = v
            continue
        if k in rn_state:
            mapped_state[k] = v

    msg = resnet.load_state_dict(mapped_state, strict=False)
    real_missing = [k for k in msg.missing_keys
                    if k not in ("conv1.weight", "fc.weight", "fc.bias")]
    if real_missing:
        print(f"[backbone] WARNING: {len(real_missing)} encoder keys not loaded "
              f"from SSL4EO: {real_missing[:5]}…")
    else:
        print(f"[backbone] All encoder layers loaded from SSL4EO "
              f"({len(mapped_state)} params)")
    if msg.unexpected_keys:
        print(f"[backbone] Unexpected keys ignored: {msg.unexpected_keys[:5]}")

    with torch.no_grad():
        marida_weights = s2_conv1[:, _SSL4EO_TO_MARIDA, :, :]  # (64, 11, 7, 7)
        index_init = s2_conv1.mean(dim=1, keepdim=True).repeat(1, in_channels - 11, 1, 1)
        index_init *= 0.1  # small magnitude for derived channels
        full_weights = torch.cat([marida_weights, index_init], dim=1)  # (64, 14, 7, 7)

    stem_conv = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
    stem_conv.weight = nn.Parameter(full_weights)
    print(f"[backbone] Sentinel-2 stem: mapped 11/13 bands + {in_channels - 11} index channels "
          f"-> conv1 shape {tuple(stem_conv.weight.shape)}")

    return resnet, stem_conv, source


def _upsample_logits(logits, size):
    return F.interpolate(logits, size=size, mode="bilinear", align_corners=False)


# ============================================================================
# Spectral-Spatial Attention Gate (SSAG) — original contribution
# ============================================================================
# Standard channel attention (SE, CBAM) treats feature channels generically.
# In multispectral satellite imagery every band has a known wavelength and
# physical meaning (e.g. B04 = red, B08 = NIR).  Hand-crafted indices like
# NDVI = (NIR-Red)/(NIR+Red) are powerful discriminators — but they are fixed.
#
# SSAG learns K "pseudo-spectral-indices" as differentiable 1x1 convolutions
# on the raw input bands, then generates spatial attention maps that are
# injected into decoder skip connections.  This fuses domain knowledge (band
# structure) with learned features, giving the decoder physics-informed
# spatial priors about where debris / ships / vegetation are likely located.
# ============================================================================

class SpectralSpatialAttentionGate(nn.Module):
    """
    Spectral-Spatial Attention Gate (SSAG).

    Given the raw multispectral input `x_spec` (B, S, H, W) and a decoder
    feature map `x_feat` (B, C, H', W'), the module:
      1. Learns K pseudo-spectral indices via 1x1 convolutions on x_spec.
      2. Resizes them to match x_feat's spatial size.
      3. Projects K index maps → C-channel attention via a small MLP.
      4. Multiplies x_feat element-wise by the sigmoid attention.
    """

    def __init__(self, spectral_channels: int, feat_channels: int, num_indices: int = 6):
        super().__init__()
        self.index_gen = nn.Sequential(
            nn.Conv2d(spectral_channels, num_indices, kernel_size=1, bias=True),
            nn.BatchNorm2d(num_indices),
            nn.ReLU(inplace=True),
        )
        self.spatial_squeeze = nn.Sequential(
            nn.Conv2d(num_indices, num_indices, kernel_size=3, padding=1,
                      groups=num_indices, bias=False),
            nn.BatchNorm2d(num_indices),
            nn.ReLU(inplace=True),
        )
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(num_indices, feat_channels // 4),
            nn.ReLU(inplace=True),
            nn.Linear(feat_channels // 4, feat_channels),
            nn.Sigmoid(),
        )
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(num_indices, 1, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x_spec, x_feat):
        indices = self.index_gen(x_spec)
        indices = F.interpolate(indices, size=x_feat.shape[2:],
                                mode="bilinear", align_corners=False)
        indices = self.spatial_squeeze(indices)

        ch_attn = self.channel_gate(indices).unsqueeze(-1).unsqueeze(-1)
        sp_attn = self.spatial_gate(indices)

        return x_feat * ch_attn * sp_attn


# ============================================================================
# Physics-Informed Index Fusion (PIIF) — extended SSAG (original)
# ============================================================================
# Papers use fixed indices (FDI, PI, NDVI); we compute them in the dataset
# as extra input channels.  PIIF takes those explicit index channels AND the
# learned pseudo-indices from SSAG, then learns a per-patch weighting via a
# lightweight attention network.  This bridges physics and learning.
# ============================================================================

class PhysicsInformedIndexFusion(nn.Module):
    """
    Fuses K_phys explicit physics indices (FDI, PI, NDVI) with K_learn
    learned pseudo-indices from SSAG's index_gen.  Outputs a fused
    K_out-channel index volume for downstream spatial/channel gates.

    The fusion learns per-patch weights over [physics; learned] channels.
    """

    def __init__(self, num_physics: int = 3, num_learned: int = 6, num_out: int = 8):
        super().__init__()
        k_total = num_physics + num_learned
        self.fuse = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(k_total, k_total),
            nn.ReLU(inplace=True),
            nn.Linear(k_total, k_total),
            nn.Sigmoid(),
        )
        self.project = nn.Sequential(
            nn.Conv2d(k_total, num_out, kernel_size=1, bias=False),
            nn.BatchNorm2d(num_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, physics_indices, learned_indices):
        """
        physics_indices: (B, K_phys, H, W) — FDI, PI, NDVI
        learned_indices: (B, K_learn, H, W) — from SSAG index_gen
        Both must be at the same spatial resolution.
        """
        cat = torch.cat([physics_indices, learned_indices], dim=1)
        w = self.fuse(cat).unsqueeze(-1).unsqueeze(-1)
        return self.project(cat * w)


class SSAGv2(nn.Module):
    """
    SSAG v2 with Physics-Informed Index Fusion (PIIF).

    Takes raw spectral bands, extracts physics indices (last 3 channels =
    NDVI, FDI, PI), learns pseudo-indices, fuses them via PIIF, then
    produces channel + spatial attention on the decoder feature map.
    """

    def __init__(self, spectral_channels: int, feat_channels: int,
                 num_physics: int = 3, num_learned: int = 6, num_fused: int = 8):
        super().__init__()
        self.num_physics = num_physics
        self.num_learned = num_learned

        self.index_gen = nn.Sequential(
            nn.Conv2d(spectral_channels, num_learned, kernel_size=1, bias=True),
            nn.BatchNorm2d(num_learned),
            nn.ReLU(inplace=True),
        )
        self.piif = PhysicsInformedIndexFusion(num_physics, num_learned, num_fused)
        self.spatial_squeeze = nn.Sequential(
            nn.Conv2d(num_fused, num_fused, kernel_size=3, padding=1,
                      groups=num_fused, bias=False),
            nn.BatchNorm2d(num_fused),
            nn.ReLU(inplace=True),
        )
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(num_fused, feat_channels // 4),
            nn.ReLU(inplace=True),
            nn.Linear(feat_channels // 4, feat_channels),
            nn.Sigmoid(),
        )
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(num_fused, 1, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x_spec, x_feat):
        """
        x_spec: (B, S, H, W) — full spectral input (last 3 channels are physics indices)
        x_feat: (B, C, H', W') — decoder feature map
        """
        physics = x_spec[:, -self.num_physics:]
        learned = self.index_gen(x_spec)

        physics_rs = F.interpolate(physics, size=x_feat.shape[2:],
                                   mode="bilinear", align_corners=False)
        learned_rs = F.interpolate(learned, size=x_feat.shape[2:],
                                   mode="bilinear", align_corners=False)

        fused = self.piif(physics_rs, learned_rs)
        fused = self.spatial_squeeze(fused)

        ch_attn = self.channel_gate(fused).unsqueeze(-1).unsqueeze(-1)
        sp_attn = self.spatial_gate(fused)

        return x_feat * ch_attn * sp_attn


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.0):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        self.double_conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.double_conv(x)


class UNetResNet50(nn.Module):
    def __init__(self, in_channels=13, out_classes=4, decoder_dropout=0.2, deep_supervision=False):
        super().__init__()
        self.deep_supervision = deep_supervision

        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        stem_conv = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        init_multispectral_stem(stem_conv, resnet.conv1.weight, in_channels)

        self.encoder0 = nn.Sequential(stem_conv, resnet.bn1, resnet.relu)
        self.pool0 = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        dd = decoder_dropout
        self.up4 = nn.ConvTranspose2d(2048, 1024, kernel_size=2, stride=2)
        self.up_conv4 = DoubleConv(2048, 1024, dropout=dd)
        self.up3 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.up_conv3 = DoubleConv(1024, 512, dropout=dd)
        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.up_conv2 = DoubleConv(512, 256, dropout=dd)
        self.up1 = nn.ConvTranspose2d(256, 64, kernel_size=2, stride=2)
        self.up_conv1 = DoubleConv(128, 64, dropout=dd)
        self.up0 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.up_conv0 = DoubleConv(64, 64, dropout=dd)
        self.out_conv = nn.Conv2d(64, out_classes, kernel_size=1)

        if deep_supervision:
            self.aux_head4 = nn.Conv2d(1024, out_classes, kernel_size=1)
            self.aux_head3 = nn.Conv2d(512, out_classes, kernel_size=1)
            self.aux_head2 = nn.Conv2d(256, out_classes, kernel_size=1)

    def freeze_encoder(self, freeze=True):
        for module in (self.encoder1, self.encoder2):
            for param in module.parameters():
                param.requires_grad = not freeze

    def forward(self, x):
        size = x.shape[2:]
        e0 = self.encoder0(x)
        e1 = self.encoder1(self.pool0(e0))
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        d4 = self.up_conv4(torch.cat([self.up4(e4), e3], dim=1))
        d3 = self.up_conv3(torch.cat([self.up3(d4), e2], dim=1))
        d2 = self.up_conv2(torch.cat([self.up2(d3), e1], dim=1))
        d1 = self.up_conv1(torch.cat([self.up1(d2), e0], dim=1))
        d0 = self.up_conv0(self.up0(d1))
        seg = self.out_conv(d0)

        if self.deep_supervision:
            aux = [
                _upsample_logits(self.aux_head4(d4), size),
                _upsample_logits(self.aux_head3(d3), size),
                _upsample_logits(self.aux_head2(d2), size),
            ]
            return {"seg": seg, "aux": aux}
        return seg


class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        reduced = max(1, in_channels // reduction)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, reduced, bias=False),
            nn.SiLU(inplace=True),
            nn.Linear(reduced, in_channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class ResNeXtBlock(nn.Module):
    def __init__(self, in_channels, out_channels, groups=None):
        super().__init__()
        if groups is None:
            groups = _get_groups(out_channels)
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(
                out_channels, out_channels, kernel_size=3, padding=1, groups=groups, bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )
        self.se = SEBlock(out_channels)
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        out = self.se(self.block(x))
        out += self.shortcut(x)
        return torch.nn.functional.silu(out)


def _init_weights(module):
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.BatchNorm2d, nn.InstanceNorm2d)):
        if module.weight is not None:
            nn.init.ones_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class ResUNext(nn.Module):
    def __init__(self, in_channels=14, out_classes=4, deep_supervision=False,
                 use_ssag=False):
        super().__init__()
        self.deep_supervision = deep_supervision
        self.use_ssag = use_ssag

        self.down1 = ResNeXtBlock(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.down2 = ResNeXtBlock(64, 128)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.down3 = ResNeXtBlock(128, 256)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.down4 = ResNeXtBlock(256, 512)
        self.pool4 = nn.MaxPool2d(2, 2)
        self.bottleneck = ResNeXtBlock(512, 1024)

        self.skip_se4 = SEBlock(512)
        self.skip_se3 = SEBlock(256)
        self.skip_se2 = SEBlock(128)
        self.skip_se1 = SEBlock(64)

        if use_ssag:
            self.ssag4 = SSAGv2(in_channels, 512)
            self.ssag3 = SSAGv2(in_channels, 256)
            self.ssag2 = SSAGv2(in_channels, 128)
            self.ssag1 = SSAGv2(in_channels, 64)

        self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.up_conv4 = ResNeXtBlock(1024, 512)
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.up_conv3 = ResNeXtBlock(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.up_conv2 = ResNeXtBlock(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.up_conv1 = ResNeXtBlock(128, 64)
        self.out_conv = nn.Conv2d(64, out_classes, kernel_size=1)

        if deep_supervision:
            self.aux_head4 = nn.Conv2d(512, out_classes, kernel_size=1)
            self.aux_head3 = nn.Conv2d(256, out_classes, kernel_size=1)
            self.aux_head2 = nn.Conv2d(128, out_classes, kernel_size=1)

        self.apply(_init_weights)

    def forward(self, x):
        size = x.shape[2:]
        x_spec = x

        x1 = self.down1(x)
        x2 = self.down2(self.pool1(x1))
        x3 = self.down3(self.pool2(x2))
        x4 = self.down4(self.pool3(x3))
        bn = self.bottleneck(self.pool4(x4))

        s4 = self.skip_se4(x4)
        s3 = self.skip_se3(x3)
        s2 = self.skip_se2(x2)
        s1 = self.skip_se1(x1)

        if self.use_ssag:
            s4 = self.ssag4(x_spec, s4)
            s3 = self.ssag3(x_spec, s3)
            s2 = self.ssag2(x_spec, s2)
            s1 = self.ssag1(x_spec, s1)

        d4 = self.up_conv4(torch.cat([self.up4(bn), s4], dim=1))
        d3 = self.up_conv3(torch.cat([self.up3(d4), s3], dim=1))
        d2 = self.up_conv2(torch.cat([self.up2(d3), s2], dim=1))
        d1 = self.up_conv1(torch.cat([self.up1(d2), s1], dim=1))
        seg = self.out_conv(d1)

        if self.deep_supervision:
            aux = [
                _upsample_logits(self.aux_head4(d4), size),
                _upsample_logits(self.aux_head3(d3), size),
                _upsample_logits(self.aux_head2(d2), size),
            ]
            return {"seg": seg, "aux": aux}
        return seg


class DoubleConvIN(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.0):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        self.double_conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.double_conv(x)


class SelfAttention2D(nn.Module):
    def __init__(self, embed_dim, num_heads=4):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm = nn.InstanceNorm2d(embed_dim, affine=True)

    def forward(self, x):
        b, c, h, w = x.shape
        x_flat = x.view(b, c, h * w).permute(0, 2, 1)
        attn_out, _ = self.mha(x_flat, x_flat, x_flat)
        attn_out = attn_out.permute(0, 2, 1).view(b, c, h, w)
        return self.norm(x + attn_out)


class CrossAttention2D(nn.Module):
    def __init__(self, embed_dim, num_heads=4):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm = nn.InstanceNorm2d(embed_dim, affine=True)

    def forward(self, query, key_val):
        b, c, h, w = query.shape
        q_flat = query.view(b, c, h * w).permute(0, 2, 1)
        kv_flat = key_val.view(b, c, h * w).permute(0, 2, 1)
        attn_out, _ = self.mha(query=q_flat, key=kv_flat, value=kv_flat)
        attn_out = attn_out.permute(0, 2, 1).view(b, c, h, w)
        return self.norm(key_val + attn_out)


class TAUNet(nn.Module):
    """Original TAUNet — from-scratch encoder + transformer-attention decoder."""

    def __init__(
        self,
        in_channels=14,
        out_classes=4,
        two_head=True,
        deep_supervision=False,
        attn_levels=("up4", "up3"),
        use_ssag=False,
    ):
        super().__init__()
        self.two_head = two_head
        self.deep_supervision = deep_supervision
        self.attn_levels = set(attn_levels)
        self.use_ssag = use_ssag

        self.down1 = DoubleConvIN(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.down2 = DoubleConvIN(64, 128)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.down3 = DoubleConvIN(128, 256)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.down4 = DoubleConvIN(256, 512)
        self.pool4 = nn.MaxPool2d(2, 2)

        self.bottleneck = DoubleConvIN(512, 1024)
        self.self_attn = SelfAttention2D(embed_dim=1024, num_heads=4)
        self.attn_dropout = nn.Dropout2d(0.1)

        SSAGClass = SSAGv2 if use_ssag else None
        if use_ssag:
            self.ssag4 = SSAGClass(in_channels, 512)
            self.ssag3 = SSAGClass(in_channels, 256)
            self.ssag2 = SSAGClass(in_channels, 128)
            self.ssag1 = SSAGClass(in_channels, 64)

        self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.cross_attn4 = (
            CrossAttention2D(embed_dim=512, num_heads=4) if "up4" in self.attn_levels else None
        )
        self.up_conv4 = DoubleConvIN(1024, 512)

        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.cross_attn3 = (
            CrossAttention2D(embed_dim=256, num_heads=4) if "up3" in self.attn_levels else None
        )
        self.up_conv3 = DoubleConvIN(512, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.cross_attn2 = (
            CrossAttention2D(embed_dim=128, num_heads=2) if "up2" in self.attn_levels else None
        )
        self.up_conv2 = DoubleConvIN(256, 128)

        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.cross_attn1 = (
            CrossAttention2D(embed_dim=64, num_heads=2) if "up1" in self.attn_levels else None
        )
        self.up_conv1 = DoubleConvIN(128, 64)

        self.out_conv = nn.Conv2d(64, out_classes, kernel_size=1)
        if two_head:
            self.debris_head = nn.Conv2d(64, 1, kernel_size=1)
            self.type_head = nn.Conv2d(64, 3, kernel_size=1)

        if deep_supervision:
            self.aux_head4 = nn.Conv2d(512, out_classes, kernel_size=1)
            self.aux_head3 = nn.Conv2d(256, out_classes, kernel_size=1)
            self.aux_head2 = nn.Conv2d(128, out_classes, kernel_size=1)

        self.apply(_init_weights)

    def _decode(self, x, skip, cross_attn, up, up_conv, ssag=None, x_spec=None):
        x = up(x)
        if cross_attn is not None:
            skip_attended = cross_attn(query=x, key_val=skip)
        else:
            skip_attended = skip
        if ssag is not None and x_spec is not None:
            skip_attended = ssag(x_spec, skip_attended)
        return up_conv(torch.cat([x, skip_attended], dim=1))

    def forward(self, x):
        size = x.shape[2:]
        x_spec = x

        x1 = self.down1(x)
        x2 = self.down2(self.pool1(x1))
        x3 = self.down3(self.pool2(x2))
        x4 = self.down4(self.pool3(x3))

        bn = self.bottleneck(self.pool4(x4))
        bn = self.attn_dropout(self.self_attn(bn))

        ssag4 = getattr(self, "ssag4", None) if self.use_ssag else None
        ssag3 = getattr(self, "ssag3", None) if self.use_ssag else None
        ssag2 = getattr(self, "ssag2", None) if self.use_ssag else None
        ssag1 = getattr(self, "ssag1", None) if self.use_ssag else None

        d4 = self._decode(bn, x4, self.cross_attn4, self.up4, self.up_conv4, ssag4, x_spec)
        d3 = self._decode(d4, x3, self.cross_attn3, self.up3, self.up_conv3, ssag3, x_spec)
        d2 = self._decode(d3, x2, self.cross_attn2, self.up2, self.up_conv2, ssag2, x_spec)
        d1 = self._decode(d2, x1, self.cross_attn1, self.up1, self.up_conv1, ssag1, x_spec)

        seg = self.out_conv(d1)
        result = {"seg": seg}
        if self.two_head:
            result["debris"] = self.debris_head(d1)
            result["type"] = self.type_head(d1)
        if self.deep_supervision:
            result["aux"] = [
                _upsample_logits(self.aux_head4(d4), size),
                _upsample_logits(self.aux_head3(d3), size),
                _upsample_logits(self.aux_head2(d2), size),
            ]
        if self.two_head or self.deep_supervision:
            return result
        return seg


# ============================================================================
# TAUNet-ResNet50: Pretrained ResNet-50 encoder + TAUNet attention decoder
# ============================================================================
# This is the main model for achieving competitive MARIDA results.
# The encoder uses SSL4EO-S12 Sentinel-2 MoCo pretrained weights (13-band),
# with band-mapped stem for MARIDA's 11 bands + 3 spectral index channels.
# The decoder retains TAUNet's cross-attention and SSAGv2 modules.
# ============================================================================

class TAUNetResNet50(nn.Module):
    """
    TAUNet with a pretrained ResNet-50 encoder.

    Encoder: ResNet-50 with SSL4EO-S12 Sentinel-2 MoCo weights (default)
             or ImageNet weights (fallback). Band-mapped multispectral stem.
    Decoder: TAUNet-style with cross-attention on skip connections,
             SSAGv2 (physics-informed index fusion), deep supervision,
             and optional two-head output.
    """

    def __init__(
        self,
        in_channels=14,
        out_classes=4,
        two_head=False,
        deep_supervision=True,
        decoder_dropout=0.15,
        attn_levels=("up4", "up3"),
        use_ssag=True,
        backbone="ssl4eo",
        ssl4eo_strict=True,
    ):
        super().__init__()
        self.two_head = two_head
        self.deep_supervision = deep_supervision
        self.attn_levels = set(attn_levels)
        self.use_ssag = use_ssag
        self.in_channels = in_channels
        self.backbone_name = backbone
        self.backbone_source = backbone

        # --- Pretrained ResNet-50 encoder ---
        if backbone == "ssl4eo":
            resnet, stem_conv, source = _load_ssl4eo_resnet50(in_channels, strict=ssl4eo_strict)
            self.backbone_source = source
        else:
            resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            stem_conv = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
            init_multispectral_stem(stem_conv, resnet.conv1.weight, in_channels)
            self.backbone_source = "imagenet"

        self.encoder0 = nn.Sequential(stem_conv, resnet.bn1, resnet.relu)  # -> 64, /2
        self.pool0 = resnet.maxpool                                        # -> 64, /4
        self.encoder1 = resnet.layer1   # -> 256, /4
        self.encoder2 = resnet.layer2   # -> 512, /8
        self.encoder3 = resnet.layer3   # -> 1024, /16
        self.encoder4 = resnet.layer4   # -> 2048, /32

        # --- Bottleneck self-attention ---
        self.bottleneck_attn = SelfAttention2D(embed_dim=2048, num_heads=8)
        self.attn_dropout = nn.Dropout2d(0.1)

        # --- SSAGv2 on skip connections ---
        if use_ssag:
            self.ssag4 = SSAGv2(in_channels, 1024)
            self.ssag3 = SSAGv2(in_channels, 512)
            self.ssag2 = SSAGv2(in_channels, 256)
            self.ssag1 = SSAGv2(in_channels, 64)

        # --- Decoder with cross-attention ---
        dd = decoder_dropout

        self.up4 = nn.ConvTranspose2d(2048, 1024, kernel_size=2, stride=2)
        self.cross_attn4 = (
            CrossAttention2D(embed_dim=1024, num_heads=4) if "up4" in self.attn_levels else None
        )
        self.up_conv4 = DoubleConv(2048, 1024, dropout=dd)

        self.up3 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.cross_attn3 = (
            CrossAttention2D(embed_dim=512, num_heads=4) if "up3" in self.attn_levels else None
        )
        self.up_conv3 = DoubleConv(1024, 512, dropout=dd)

        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.cross_attn2 = (
            CrossAttention2D(embed_dim=256, num_heads=4) if "up2" in self.attn_levels else None
        )
        self.up_conv2 = DoubleConv(512, 256, dropout=dd)

        self.up1 = nn.ConvTranspose2d(256, 64, kernel_size=2, stride=2)
        self.cross_attn1 = (
            CrossAttention2D(embed_dim=64, num_heads=2) if "up1" in self.attn_levels else None
        )
        self.up_conv1 = DoubleConv(128, 64, dropout=dd)

        self.up0 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.up_conv0 = DoubleConv(64, 64, dropout=dd)

        self.out_conv = nn.Conv2d(64, out_classes, kernel_size=1)
        if two_head:
            self.debris_head = nn.Conv2d(64, 1, kernel_size=1)
            self.type_head = nn.Conv2d(64, 3, kernel_size=1)

        if deep_supervision:
            self.aux_head4 = nn.Conv2d(1024, out_classes, kernel_size=1)
            self.aux_head3 = nn.Conv2d(512, out_classes, kernel_size=1)
            self.aux_head2 = nn.Conv2d(256, out_classes, kernel_size=1)

    def freeze_encoder(self, freeze=True):
        for module in (self.encoder0, self.encoder1, self.encoder2,
                       self.encoder3, self.encoder4):
            for param in module.parameters():
                param.requires_grad = not freeze

    def encoder_parameters(self):
        """Parameters belonging to the pretrained encoder (low LR group)."""
        encoder_modules = [self.encoder0, self.pool0, self.encoder1,
                           self.encoder2, self.encoder3, self.encoder4]
        seen = set()
        for mod in encoder_modules:
            for p in mod.parameters():
                if id(p) not in seen:
                    seen.add(id(p))
                    yield p

    def decoder_parameters(self):
        """Parameters belonging to the decoder, SSAG, and heads (high LR group)."""
        encoder_ids = {id(p) for p in self.encoder_parameters()}
        for p in self.parameters():
            if id(p) not in encoder_ids:
                yield p

    def _decode(self, x, skip, cross_attn, up, up_conv, ssag=None, x_spec=None):
        x = up(x)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
        if cross_attn is not None:
            skip_attended = cross_attn(query=x, key_val=skip)
        else:
            skip_attended = skip
        if ssag is not None and x_spec is not None:
            skip_attended = ssag(x_spec, skip_attended)
        return up_conv(torch.cat([x, skip_attended], dim=1))

    def forward(self, x):
        size = x.shape[2:]
        x_spec = x

        e0 = self.encoder0(x)        # (B, 64, H/2, W/2)
        e1 = self.encoder1(self.pool0(e0))  # (B, 256, H/4, W/4)
        e2 = self.encoder2(e1)        # (B, 512, H/8, W/8)
        e3 = self.encoder3(e2)        # (B, 1024, H/16, W/16)
        e4 = self.encoder4(e3)        # (B, 2048, H/32, W/32)

        bn = self.attn_dropout(self.bottleneck_attn(e4))

        ssag4 = getattr(self, "ssag4", None) if self.use_ssag else None
        ssag3 = getattr(self, "ssag3", None) if self.use_ssag else None
        ssag2 = getattr(self, "ssag2", None) if self.use_ssag else None
        ssag1 = getattr(self, "ssag1", None) if self.use_ssag else None

        d4 = self._decode(bn, e3, self.cross_attn4, self.up4, self.up_conv4, ssag4, x_spec)
        d3 = self._decode(d4, e2, self.cross_attn3, self.up3, self.up_conv3, ssag3, x_spec)
        d2 = self._decode(d3, e1, self.cross_attn2, self.up2, self.up_conv2, ssag2, x_spec)
        d1 = self._decode(d2, e0, self.cross_attn1, self.up1, self.up_conv1, ssag1, x_spec)
        d0 = self.up_conv0(self.up0(d1))

        seg = self.out_conv(d0)
        result = {"seg": seg}
        if self.two_head:
            result["debris"] = self.debris_head(d0)
            result["type"] = self.type_head(d0)
        if self.deep_supervision:
            result["aux"] = [
                _upsample_logits(self.aux_head4(d4), size),
                _upsample_logits(self.aux_head3(d3), size),
                _upsample_logits(self.aux_head2(d2), size),
            ]
        if self.two_head or self.deep_supervision:
            return result
        return seg


def build_model(name, in_channels=14, out_classes=4, two_head=True,
                deep_supervision=True, use_ssag=False, backbone="ssl4eo",
                ssl4eo_strict=True):
    """Factory for all segmentation architectures."""
    name = name.lower()
    if name in ("unet_resnet50", "unetresnet50", "resnet50"):
        return UNetResNet50(
            in_channels=in_channels,
            out_classes=out_classes,
            deep_supervision=deep_supervision,
        )
    if name in ("resunext", "res_unext"):
        return ResUNext(
            in_channels=in_channels,
            out_classes=out_classes,
            deep_supervision=deep_supervision,
            use_ssag=use_ssag,
        )
    if name in ("taunet", "tau"):
        return TAUNet(
            in_channels=in_channels,
            out_classes=out_classes,
            two_head=two_head,
            deep_supervision=deep_supervision,
            use_ssag=use_ssag,
        )
    if name in ("taunet_resnet50", "taunetresnet50", "tau_resnet50"):
        return TAUNetResNet50(
            in_channels=in_channels,
            out_classes=out_classes,
            two_head=two_head,
            deep_supervision=deep_supervision,
            use_ssag=use_ssag,
            backbone=backbone,
            ssl4eo_strict=ssl4eo_strict,
        )
    raise ValueError(
        f"Model necunoscut: {name}. "
        f"Alege: taunet, taunet_resnet50, resunext, unet_resnet50"
    )


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for model_name in ("taunet", "taunet_resnet50", "resunext", "unet_resnet50"):
        two_head = model_name in ("taunet", "taunet_resnet50")
        use_ssag = model_name != "unet_resnet50"
        model = build_model(model_name, in_channels=14, two_head=two_head,
                            use_ssag=use_ssag).to(device)
        x = torch.randn(2, 14, 256, 256, device=device)
        out = model(x)
        if isinstance(out, dict):
            print(model_name, "seg", out["seg"].shape, "aux", len(out.get("aux", [])))
        else:
            print(model_name, out.shape)
        total_params = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"  Parameters: {total_params:.1f}M")
