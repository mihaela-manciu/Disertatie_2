# Thesis Roadmap — Disertatie_2

## Goal

Achieve **0.65–0.73 MARIDA MD-class IoU** (binary marine debris) and **0.40–0.55 mIoU foreground** on the 4-class task (water / plastic / natural debris / ships), up from ~0.28 in Disertatie_1.

**Target hardware:** AMD Ryzen 7 5800X3D · 36 GB RAM · RTX 3080 Ti (12 GB VRAM)

---

## Architecture of Changes (what's new vs Disertatie_1)

| Component | File(s) | Change | Paper source(s) |
|-----------|---------|--------|-----------------|
| **Pretrained encoder** | `models.py` → `TAUNetResNet50` | **SSL4EO-S12 Sentinel-2 MoCo** ResNet-50 encoder (13-band native) with band-mapped 14-channel stem. Falls back to ImageNet if torchgeo unavailable. | SSL4EO-S12 (Wang et al. 2022), SeCo (Mañas et al. 2021 ICCV), MFPN 2024, Hybrid UNet-ResNeXt50 2025 |
| **14-channel input** | `dataset.py` | Added PI (Plastic Index) as 14th channel alongside NDVI+FDI | Biermann 2020, Themistocleous 2020, Torres TAU-Net 2025 |
| **SSAGv2 + PIIF** | `models.py` → `SSAGv2`, `PhysicsInformedIndexFusion` | Fuses explicit physics indices (FDI, PI, NDVI) with learned pseudo-indices via attention — *original contribution* | SAMSelect 2025 (search), SSAG (ours) |
| **Rarity-aware sampling** | `dataset.py` → `build_debris_weighted_sampler` | Separate plastic_boost (12×) and debris_boost (5×) weights | Bouchelaghem 2026, ResAttUNet 2022 |
| **`pretrained_strong` recipe** | `train.py` | LR 5e-5, 8-epoch warmup, 5-epoch encoder freeze, grad accum 4, 150 epochs, copy-paste 0.7 | Tuned for pretrained + 3080 Ti |
| **Thermal guard** | `train.py` | GPU temp monitoring + adaptive pause (RTX 3080 Ti can throttle) | — |

---

## Idea-by-Idea Mapping

### Tier A — High Impact (must-do)

| # | Idea | Expected Δ | File(s) | Paper(s) |
|---|------|-----------|---------|----------|
| **A1** | SSL4EO-S12 Sentinel-2 pretrained ResNet-50 encoder | +5–15 pts mIoU | `models.py` (`TAUNetResNet50`, `_load_ssl4eo_resnet50`) | SSL4EO-S12 (Wang et al. 2022): 13-band S2 MoCo pretraining, 91.8% BigEarthNet. SeCo (Mañas et al. ICCV 2021): in-domain pretraining > ImageNet for RS. MFPN 2024: mIoU=0.71. |
| **A2** | Extra input: PI (Plastic Index) | +1–3 pts plastic IoU | `dataset.py` (`_compute_spectral_indices`) | Themistocleous 2020: PI=NIR/(NIR+Red) highlights plastics. Biermann 2020: FDI. Torres 2025: NDVI+FDI in TAU-Net. |
| **A3** | Rarity-aware sampling (plastic 12×, debris 5×) | +2–5 pts plastic recall | `dataset.py` (`build_debris_weighted_sampler`) | ResAttUNet 2022: class-weighted focal. Bouchelaghem 2026: rarity-aware training for binary. |
| **A4** | Focal + Lovász + boundary loss combo | +1–3 pts edge accuracy | `losses.py`, `train.py` | ResAttUNet 2022: focal γ=2. Lovász-Softmax: direct IoU optimization. |
| **A5** | Multi-scale TTA at eval (0.75, 1.0, 1.25) | +2–5 pts test mIoU | `segmentation_utils.py` | MFPN 2024, standard practice. |
| **A6** | Longer training (150 ep) + cosine warmup | Stability, no epoch-11 collapse | `train.py` (`LinearWarmupCosineScheduler`) | Standard for pretrained fine-tuning. |

### Tier B — Strong Thesis Story

| # | Idea | File(s) | Paper(s) |
|---|------|---------|----------|
| **B1** | Ablation: scratch TAUNet vs TAUNet-ResNet50 | `run_experiments.py` (run both) | Compare with MARIDA baselines (RF, UNet). |
| **B2** | Two-head (binary debris + type) ablation | `run_experiments.py --two-head` | Torres TAU-Net 2025, our `two_head` design. |
| **B3** | Copy-paste with plastic-only donors at 0.7 prob | `dataset.py` (`apply_copy_paste`) | PLP field campaigns, augmentation literature. |
| **B4** | Report MARIDA-native binary MD metrics | `segmentation_utils.py` (`calculate_debris_metrics`) | MARIDA 2022, ResAttUNet 2022. |

### Tier C — Original Contribution (novelty)

| # | Idea | Why novel | File(s) | Builds on |
|---|------|-----------|---------|-----------|
| **C1** | **SSAGv2 + Physics-Informed Index Fusion (PIIF)** | Papers use *fixed* indices OR *learned-only* attention. PIIF bridges both: learns per-patch weights over {FDI, PI, NDVI, pseudo-indices} | `models.py` (`SSAGv2`, `PhysicsInformedIndexFusion`) | SAMSelect 2025 (band search) + original SSAG |
| **C2** | 14-channel architecture-level integration | No MARIDA paper uses PI as an explicit model input | `dataset.py`, `models.py` | Themistocleous 2020, Torres 2025 |

---

## Recommended Training Commands

### Primary run (pretrained TAUNet-ResNet50, single-head, aug + no_aug)
```bash
python run_experiments.py \
    --marida "D:\TAID\Disertatie\MARIDA" \
    --recipe pretrained_strong \
    --models taunet_resnet50 \
    --num-workers 6 \
    --pin-memory
```

### Ablation: scratch TAUNet (for comparison)
```bash
python run_experiments.py \
    --marida "D:\TAID\Disertatie\MARIDA" \
    --recipe strong \
    --models taunet \
    --num-workers 6 \
    --pin-memory
```

### Two-head ablation
```bash
python run_experiments.py \
    --marida "D:\TAID\Disertatie\MARIDA" \
    --recipe pretrained_strong \
    --models taunet_resnet50 \
    --two-head \
    --num-workers 6 \
    --pin-memory
```

### Quick smoke test (3 epochs, 1 step each)
```bash
python run_experiments.py \
    --marida "D:\TAID\Disertatie\MARIDA" \
    --models taunet_resnet50 \
    --fast --smoke
```

---

## 3080 Ti Settings

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `batch_size` | **4** (fallback: 2) | TAUNet-ResNet50 + SSAGv2 + deep_sup ≈ 8–10 GB at batch 4 |
| `grad_accum_steps` | **4** | Effective batch = 16 |
| `num_workers` | **6** | Ryzen 7 5800X3D has 8 cores; leave 2 for system |
| `pin_memory` | **True** | Faster host→device transfers with dedicated GPU |
| `use_amp` | **True** | FP16 saves ~30% VRAM |
| `freeze_encoder_epochs` | **3** | Let decoder warm up before fine-tuning encoder |
| `warmup_epochs` | **3** | Short warmup; SSL4EO features need less protection than ImageNet |
| `lr` | **1e-4** (encoder), **1e-3** (decoder) | Differential LR: 10× multiplier for randomly initialized decoder |
| `epochs` | **150** | With patience 35 and cosine schedule |
| `copy_paste_prob` | **0.7** | Aggressive debris augmentation |
| `plastic_boost` | **12** | Oversample plastic patches 12× vs water-only |

---

## Expected Results

| Metric | Disertatie_1 (results_3) | Target (conservative) | Target (optimistic) |
|--------|-------------------------|----------------------|---------------------|
| mIoU foreground (1–3) | 0.282 | 0.40–0.50 | 0.50–0.60 |
| Binary debris IoU (1+2) | 0.203 | 0.35–0.45 | 0.50–0.65 |
| Plastic IoU (class 1) | 0.101 | 0.20–0.30 | 0.30–0.45 |
| MARIDA MD IoU (if eval on MD label) | ~0.30 est. | 0.55–0.65 | 0.65–0.73 |

---

## Thesis Framing

1. **SSAGv2 + PIIF** is the original contribution (Tier C).
2. **TAUNet-ResNet50** is the competitive backbone that closes the gap with MARIDA benchmarks.
3. Always report **plastic IoU** and **binary debris IoU** separately from mIoU foreground — ships dominate the foreground metric.
4. Compare fairly: cite MFPN's 0.71, ResAttUNet's 0.67, and the 2026 binary paper's 0.89 F1 — but note they use different class sets, metrics, or binary formulations.
5. Ablation table in thesis: scratch vs pretrained, 13ch vs 14ch, SSAG vs SSAGv2, single vs two-head.

---

## File Map

| File | Role |
|------|------|
| `models.py` | All architectures: TAUNet, TAUNet-ResNet50, ResUNext, UNet-ResNet50, SSAG, SSAGv2, PIIF |
| `dataset.py` | 14-channel MARIDA loader with PI, rarity-aware sampler, copy-paste augmentation |
| `losses.py` | Focal, Dice, Lovász, Boundary, two-head, OHEM losses |
| `train.py` | Training loop, recipes (`pretrained_strong`, `strong`, etc.), thermal guard |
| `evaluate.py` | Test evaluation, qualitative panels, training curve plots |
| `segmentation_utils.py` | TTA, CRF post-processing, metrics, threshold tuning |
| `visual_reporting.py` | Zoom comparison panels for aug vs no_aug |
| `run_experiments.py` | Full automated experiment pipeline (train → eval → compare) |

---

## Paper References

| Short name | Full title | Year | Key takeaway |
|-----------|-----------|------|-------------|
| MARIDA | MARIDA: A benchmark for Marine Debris detection from Sentinel-2 | 2022 | Dataset + RF/UNet baselines (IoU=0.69) |
| ResAttUNet | Detecting Marine Debris using Attention-Activated Residual UNet | 2022 | CBAM + residual + focal; IoU=0.67 on MARIDA |
| MFPN | Marine debris detection using a multi-feature pyramid network | 2024 | ResNet-101 pretrained; mIoU=0.71, F1=0.80 on MARIDA |
| MADOS/MariNeXt | Detecting Marine pollutants with Deep learning in Sentinel-2 | 2024 | SegNeXt-style; +12% F1 vs baselines; 174 scenes |
| Hybrid UNet-ResNeXt50 | Hybrid Deep Learning for Marine Debris in Satellite Imagery | 2025 | Pretrained ResNeXt-50; strong recall on debris |
| Torres TAU-Net | Transformer Assisted U-Net for Marine Litter on Sentinel-2 | 2025 | Cross-attention decoder + NDVI/FDI; FloatingObjects dataset |
| SAMSelect | Spectral Index Search for Marine Debris using SAM | 2025 | Automated band/index selection for visualization |
| Bouchelaghem 2026 | Binary reformulation on combined MARIDA and MADOS | 2026 | Cross-dataset F1=0.89; rarity-aware binary training |
| SSL4EO-S12 | SSL4EO-S12: A Large-Scale Multi-Modal Multi-Task Dataset for Self-Supervised Learning in EO | 2022 | ResNet-50 pretrained on 13-band Sentinel-2 via MoCo/DINO; 91.8% BigEarthNet |
| SeCo | Seasonal Contrast: Unsupervised Pre-Training from Uncurated Remote Sensing Data | 2021 | Domain-specific pretraining > ImageNet for RS; ICCV 2021 |
| Biermann 2020 | Finding plastic patches using optical satellite data | 2020 | FDI (Floating Debris Index) definition |
| Themistocleous 2020 | Investigating floating plastic litter from space | 2020 | PI (Plastic Index) = NIR/(NIR+Red) |
| Octonion 2023 | Marine Debris Segmentation Using Parameter-Efficient Octonion | 2023 | +9.9% IoU vs SOTA on MARIDA with hypercomplex nets |
| EPFL Large-scale 2023 | Large-scale Detection of Marine Debris in Coastal Areas | 2023 | Data-centric: negative sampling, label refinement |
