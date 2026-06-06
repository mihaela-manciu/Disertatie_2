import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
if os.name != "nt":
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import time

import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm


_PYNVML_AVAILABLE = False
try:
    import pynvml
    pynvml.nvmlInit()
    _PYNVML_AVAILABLE = True
except Exception:
    pass


def _gpu_temp_celsius(device_index: int = 0) -> float | None:
    if not _PYNVML_AVAILABLE:
        return None
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
        return float(pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU))
    except Exception:
        return None


def thermal_guard(device_index=0, warn_temp=78.0, critical_temp=85.0,
                  poll_interval=10.0, max_wait=300.0):
    temp = _gpu_temp_celsius(device_index)
    if temp is None or temp < warn_temp:
        return
    level = "CRITIC" if temp >= critical_temp else "AVERTIZARE"
    print(f"\n[thermal] {level}: GPU la {temp:.0f} °C (prag {warn_temp:.0f} °C). Pauză adaptivă…")
    waited = 0.0
    while waited < max_wait:
        time.sleep(poll_interval)
        waited += poll_interval
        temp = _gpu_temp_celsius(device_index)
        if temp is None:
            break
        print(f"[thermal] GPU: {temp:.0f} °C (așteptat {waited:.0f}s / max {max_wait:.0f}s)")
        if temp < warn_temp:
            print(f"[thermal] GPU răcit la {temp:.0f} °C — reluare antrenare.")
            break
    else:
        print(f"[thermal] Timeout {max_wait:.0f}s, reluare oricum.")


def _make_amp_helpers(device, use_amp):
    if hasattr(torch.amp, "autocast"):
        autocast_ctx = lambda: torch.amp.autocast(device.type, enabled=use_amp)
    else:
        from torch.cuda.amp import autocast as legacy_autocast
        autocast_ctx = lambda: legacy_autocast(enabled=use_amp)

    if hasattr(torch.amp, "GradScaler"):
        scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    else:
        from torch.cuda.amp import GradScaler as LegacyScaler
        scaler = LegacyScaler(enabled=use_amp)
    return autocast_ctx, scaler


from dataset import (
    MARIDACropMiningDataset,
    MARIDADataset,
    build_debris_weighted_sampler,
    get_training_augmentation,
    get_validation_augmentation,
    load_channel_stats,
    get_inverse_freq_weights,
    marida_pad_collate,
    NUM_CHANNELS,
)
from losses import (
    BoundaryLoss,
    DeepSupervisionHybridLoss,
    MDOutlierLoss,
    OhemTwoHeadHybridLoss,
    TwoHeadHybridLoss,
)
from metrics_reporting import plot_training_dashboard
from models import UNetResNet50, TAUNetResNet50, build_model
from experiment_timing import fmt_duration, format_timing_dict, print_timing_block, sec_since
from segmentation_utils import compute_md_outlier_metrics, compute_val_metrics
from training_stability import (
    MetricEMA,
    ModelEMA,
    checkpoint_score,
    md_plastic_checkpoint_score,
    set_training_seed,
)


class LinearWarmupCosineScheduler(optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_epochs, total_epochs, eta_min=1e-6, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            alpha = (self.last_epoch + 1) / max(1, self.warmup_epochs)
            return [base_lr * alpha for base_lr in self.base_lrs]
        progress = (self.last_epoch - self.warmup_epochs) / max(1, self.total_epochs - self.warmup_epochs)
        import math
        cos_factor = 0.5 * (1 + math.cos(math.pi * progress))
        return [self.eta_min + (base_lr - self.eta_min) * cos_factor for base_lr in self.base_lrs]


def _sanitize_batch(images, masks, num_classes=4):
    images = torch.nan_to_num(images, nan=0.0, posinf=0.0, neginf=0.0)
    masks = masks.long().clamp(0, num_classes - 1)
    return images, masks


def train_one_epoch(model, dataloader, criterion, optimizer, device, *,
                    use_amp=False, max_steps=None, grad_accum_steps=1,
                    model_ema=None):
    model.train()
    running_loss = 0.0
    valid_steps = 0
    skipped_steps = 0
    pasted_total = 0
    pixels_total = 0
    ema_updates = 0
    autocast_ctx, scaler = _make_amp_helpers(device, use_amp)
    loop = tqdm(dataloader, desc="Training", leave=False)
    steps = 0
    accum_count = 0
    optimizer.zero_grad(set_to_none=True)

    for batch in loop:
        if max_steps is not None and steps >= max_steps:
            break
        steps += 1
        if len(batch) == 3:
            images, masks, meta = batch
            if isinstance(meta, dict) and "pasted_pixels" in meta and "total_pixels" in meta:
                pasted_total += int(meta["pasted_pixels"].sum().item())
                pixels_total += int(meta["total_pixels"].sum().item())
            else:
                try:
                    pasted_total += int(sum(int(m["pasted_pixels"]) for m in meta))
                    pixels_total += int(sum(int(m["total_pixels"]) for m in meta))
                except Exception:
                    pass
        else:
            images, masks = batch
        images = images.to(device)
        masks = masks.to(device, dtype=torch.long)
        images, masks = _sanitize_batch(images, masks)

        with autocast_ctx():
            outputs = model(images)
        if isinstance(outputs, dict):
            outputs = {k: v.float() if torch.is_tensor(v) else v for k, v in outputs.items()}
        elif torch.is_tensor(outputs):
            outputs = outputs.float()
        try:
            loss = criterion(outputs, masks)
        except RuntimeError as exc:
            if steps <= 1:
                import traceback
                masks_cpu = masks.detach().cpu()
                print(
                    f"\n[train] batch {steps} failed: masks dtype={masks_cpu.dtype} "
                    f"shape={tuple(masks_cpu.shape)} "
                    f"min={int(masks_cpu.min())} max={int(masks_cpu.max())}"
                )
                traceback.print_exc()
            raise exc
        loss = loss / grad_accum_steps

        if not torch.isfinite(loss):
            skipped_steps += 1
            if skipped_steps <= 3 or skipped_steps % 50 == 0:
                print(f"\n[train] batch {steps}: loss non-finite, skip (total skipped={skipped_steps})")
            continue

        if use_amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        accum_count += 1
        if accum_count >= grad_accum_steps:
            stepped = False
            if use_amp:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                stepped = True
            else:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                if torch.isfinite(grad_norm):
                    optimizer.step()
                    stepped = True
                else:
                    skipped_steps += 1
            if stepped and model_ema is not None:
                model_ema.update(model)
                ema_updates += 1
            optimizer.zero_grad(set_to_none=True)
            accum_count = 0

        loss_val = float(loss.item()) * grad_accum_steps
        running_loss += loss_val
        valid_steps += 1
        loop.set_postfix(loss=loss_val, skip=skipped_steps)

    if accum_count > 0:
        stepped = False
        if use_amp:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            stepped = True
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            if torch.isfinite(grad_norm):
                optimizer.step()
                stepped = True
        if stepped and model_ema is not None:
            model_ema.update(model)
            ema_updates += 1
        optimizer.zero_grad(set_to_none=True)

    if skipped_steps > 0:
        print(f"[train] epocă: {skipped_steps} batch-uri sărite (loss/grad non-finite)")

    avg_loss = running_loss / max(1, valid_steps)
    paste_ratio = (pasted_total / max(1, pixels_total)) if pixels_total > 0 else 0.0
    return avg_loss, {
        "pasted_pixels": int(pasted_total),
        "total_pixels": int(pixels_total),
        "paste_ratio": float(paste_ratio),
        "skipped_batches": int(skipped_steps),
        "ema_updates": int(ema_updates),
    }


def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    valid_steps = 0
    with torch.no_grad():
        for images, masks in tqdm(dataloader, desc="Validation", leave=False):
            images = images.to(device)
            masks = masks.to(device, dtype=torch.long)
            images, masks = _sanitize_batch(images, masks)
            outputs = model(images)
            if isinstance(outputs, dict):
                outputs = {k: v.float() if torch.is_tensor(v) else v for k, v in outputs.items()}
            elif torch.is_tensor(outputs):
                outputs = outputs.float()
            loss = criterion(outputs, masks)
            if torch.isfinite(loss):
                running_loss += loss.item()
                valid_steps += 1
    return running_loss / max(1, valid_steps)


def build_criterion(class_weights, two_head, use_ohem, deep_sup_weights, device,
                    *, use_focal=True, lovasz_weight=0.3, label_smoothing=0.0,
                    debris_weight=2.0, type_weight=1.5, seg_weight=0.5,
                    binary_pos_weight=20.0, focal_gamma=1.5,
                    training_mode="standard", binary_lovasz_weight=0.25):
    if training_mode == "md_outlier":
        return MDOutlierLoss(
            class_weight=class_weights,
            debris_weight=debris_weight,
            type_weight=type_weight,
            seg_weight=seg_weight,
            binary_lovasz_weight=binary_lovasz_weight,
            deep_sup_weights=deep_sup_weights,
            use_focal=use_focal,
            lovasz_weight=lovasz_weight,
            binary_pos_weight=binary_pos_weight,
            focal_gamma=focal_gamma,
        ).to(device)
    if two_head:
        if use_ohem:
            return OhemTwoHeadHybridLoss(
                class_weight=class_weights,
                debris_weight=debris_weight,
                type_weight=type_weight,
                seg_weight=seg_weight,
                deep_sup_weights=deep_sup_weights,
                use_focal=use_focal,
                lovasz_weight=lovasz_weight,
                keep_ratio=0.25,
            ).to(device)
        return TwoHeadHybridLoss(
            class_weight=class_weights,
            debris_weight=debris_weight,
            type_weight=type_weight,
            seg_weight=seg_weight,
            deep_sup_weights=deep_sup_weights,
            use_focal=use_focal,
            lovasz_weight=lovasz_weight,
            binary_pos_weight=binary_pos_weight,
            focal_gamma=focal_gamma,
        ).to(device)
    return DeepSupervisionHybridLoss(
        deep_sup_weights=deep_sup_weights,
        weight=class_weights,
        use_focal=use_focal,
        lovasz_weight=lovasz_weight,
        label_smoothing=label_smoothing,
    ).to(device)


DEEP_SUP_WEIGHTS = [0.08, 0.05, 0.02]

RECIPE_PRESETS = {
    "balanced": {
        "class_weights": "auto",
        "use_focal": False,
        "lovasz_weight": 0.0,
        "use_ohem": False,
        "use_deep_sup": False,
        "two_head_default": False,
        "debris_boost": 3.0,
        "plastic_boost": 6.0,
        "lr": 3e-4,
        "warmup_epochs": 5,
        "label_smoothing": 0.05,
        "class_weights_from_data": True,
        "freeze_encoder_epochs": 0,
        "default_epochs": 80,
        "default_patience": 20,
        "use_ssag": False,
        "boundary_weight": 0.0,
        "grad_accum_steps": 1,
        "copy_paste_prob": 0.2,
        "tta_scales_eval": [1.0],
    },
    "strong": {
        "class_weights": "auto",
        "use_focal": True,
        "lovasz_weight": 0.2,
        "use_ohem": False,
        "use_deep_sup": True,
        "two_head_default": False,
        "debris_boost": 5.0,
        "plastic_boost": 10.0,
        "lr": 1e-4,
        "warmup_epochs": 5,
        "label_smoothing": 0.05,
        "class_weights_from_data": True,
        "freeze_encoder_epochs": 0,
        "default_epochs": 120,
        "default_patience": 30,
        "use_ssag": True,
        "boundary_weight": 0.3,
        "grad_accum_steps": 4,
        "copy_paste_prob": 0.6,
        "tta_scales_eval": [0.75, 1.0, 1.25],
    },
    "strong_two_head": {
        "class_weights": "auto",
        "use_focal": True,
        "lovasz_weight": 0.2,
        "use_ohem": False,
        "use_deep_sup": True,
        "two_head_default": True,
        "debris_boost": 5.0,
        "plastic_boost": 10.0,
        "lr": 1e-4,
        "warmup_epochs": 5,
        "label_smoothing": 0.05,
        "class_weights_from_data": True,
        "freeze_encoder_epochs": 0,
        "default_epochs": 120,
        "default_patience": 30,
        "use_ssag": True,
        "boundary_weight": 0.3,
        "grad_accum_steps": 4,
        "copy_paste_prob": 0.6,
        "tta_scales_eval": [0.75, 1.0, 1.25],
    },
    "pretrained_strong": {
        "class_weights": "auto",
        "use_focal": True,
        "lovasz_weight": 0.20,
        "use_ohem": False,
        "use_deep_sup": True,
        "two_head_default": False,
        "debris_boost": 5.0,
        "plastic_boost": 12.0,
        "lr": 1e-4,
        "decoder_lr_mult": 10.0,
        "warmup_epochs": 5,
        "label_smoothing": 0.05,
        "class_weights_from_data": True,
        "class_weight_max": 8.0,
        "freeze_encoder_epochs": 5,
        "encoder_unfreeze_ramp_epochs": 3,
        "encoder_lr_ramp_mult": 0.25,
        "default_epochs": 150,
        "default_patience": 35,
        "use_ssag": True,
        "boundary_weight": 0.1,
        "grad_accum_steps": 4,
        "copy_paste_prob": 0.4,
        "tta_scales_eval": [0.75, 1.0, 1.25],
        "backbone": "ssl4eo",
        "ema_decay": 0.999,
        "metric_ema_decay": 0.85,
        "checkpoint_miou_weight": 0.25,
        "checkpoint_plastic_weight": 0.35,
        "checkpoint_fg_binary_weight": 0.40,
        "val_fg_threshold": 0.20,
        "focal_gamma": 2.0,
        "seed": 42,
    },
    "pretrained_strong_two_head": {
        "class_weights": "auto",
        "use_focal": True,
        "lovasz_weight": 0.20,
        "use_ohem": False,
        "use_deep_sup": True,
        "two_head_default": True,
        "debris_boost": 5.0,
        "plastic_boost": 12.0,
        "lr": 1e-4,
        "decoder_lr_mult": 10.0,
        "warmup_epochs": 5,
        "label_smoothing": 0.05,
        "class_weights_from_data": True,
        "class_weight_max": 8.0,
        "freeze_encoder_epochs": 5,
        "encoder_unfreeze_ramp_epochs": 3,
        "encoder_lr_ramp_mult": 0.25,
        "default_epochs": 150,
        "default_patience": 35,
        "use_ssag": True,
        "boundary_weight": 0.1,
        "grad_accum_steps": 4,
        "copy_paste_prob": 0.4,
        "tta_scales_eval": [0.75, 1.0, 1.25],
        "backbone": "ssl4eo",
        "ema_decay": 0.999,
        "metric_ema_decay": 0.85,
        "checkpoint_miou_weight": 0.25,
        "checkpoint_plastic_weight": 0.35,
        "checkpoint_fg_binary_weight": 0.40,
        "val_fg_threshold": 0.20,
        "binary_pos_weight": 10.0,
        "focal_gamma": 2.0,
        "debris_weight": 1.0,
        "type_weight": 0.5,
        "seg_weight": 1.0,
        "seed": 42,
    },
    "md_outlier": {
        "training_mode": "md_outlier",
        "class_weights": "auto",
        "use_focal": True,
        "lovasz_weight": 0.10,
        "use_ohem": False,
        "use_deep_sup": True,
        "two_head_default": True,
        "debris_boost": 8.0,
        "plastic_boost": 30.0,
        "lr": 1e-4,
        "decoder_lr_mult": 10.0,
        "warmup_epochs": 5,
        "label_smoothing": 0.05,
        "class_weights_from_data": True,
        "class_weight_max": 8.0,
        "freeze_encoder_epochs": 5,
        "encoder_unfreeze_ramp_epochs": 3,
        "encoder_lr_ramp_mult": 0.25,
        "default_epochs": 150,
        "default_patience": 35,
        "use_ssag": True,
        "boundary_weight": 0.05,
        "grad_accum_steps": 4,
        "copy_paste_prob": 0.7,
        "tta_scales_eval": [0.75, 1.0, 1.25],
        "backbone": "ssl4eo",
        "ema_decay": 0.999,
        "metric_ema_decay": 0.85,
        "checkpoint_md_weight": 0.45,
        "checkpoint_plastic_weight": 0.30,
        "checkpoint_plastic_recall_weight": 0.25,
        "val_debris_threshold": 0.35,
        "debris_weight": 3.0,
        "type_weight": 0.25,
        "seg_weight": 0.25,
        "binary_lovasz_weight": 0.25,
        "binary_pos_weight": 15.0,
        "focal_gamma": 2.0,
        "crop_mining": True,
        "crop_size": 128,
        "full_patch_prob": 0.5,
        "crop_prob": 0.5,
        "plastic_focus_prob": 0.75,
        "checkpoint_mode": "md_plastic",
        "checkpoint_md_iou_weight": 0.5,
        "checkpoint_plastic_iou_weight": 0.5,
        "variants": ["no_aug"],
        "copy_paste_on_no_aug": True,
        "ssl4eo_strict": True,
        "seed": 42,
    },
}


def print_recipe_config(recipe: str, preset: dict, *, two_head: bool, model_key: str) -> None:
    import losses as _losses_mod
    losses_path = os.path.abspath(_losses_mod.__file__)
    dtype_guard = hasattr(_losses_mod, "_type_labels_from_target")
    print(
        f"[config] build={os.path.basename(os.path.dirname(os.path.abspath(__file__)))} "
        f"recipe={recipe} model={model_key} two_head={two_head} "
        f"mode={preset.get('training_mode', 'standard')} "
        f"lr={preset.get('lr')} freeze={preset.get('freeze_encoder_epochs')} "
        f"debris_boost={preset.get('debris_boost')} plastic_boost={preset.get('plastic_boost')} "
        f"copy_paste={preset.get('copy_paste_prob')} ema={preset.get('ema_decay', 0)} "
        f"lovasz={preset.get('lovasz_weight')} crop_mining={preset.get('crop_mining', False)} "
        f"full_patch_prob={preset.get('full_patch_prob', 0)} "
        f"ckpt_mode={preset.get('checkpoint_mode', 'score_ema')} "
        f"loss_dtype_guard={dtype_guard}"
    )
    print(f"[config] losses module: {losses_path}")


def resolve_two_head(model_key, recipe, two_head=None):
    model_key = model_key.lower()
    if model_key not in ("taunet", "taunet_resnet50"):
        if two_head is True:
            print(f"[two_head] Ignorat pentru {model_key} — disponibil doar pe TAUNet.")
        return False
    if two_head is not None:
        return bool(two_head)
    if recipe not in RECIPE_PRESETS:
        raise ValueError(f"Recipe necunoscut: {recipe}")
    return bool(RECIPE_PRESETS[recipe].get("two_head_default", False))


def run_training(
    marida_path,
    model_key,
    *,
    epochs=50,
    batch_size=4,
    lr=None,
    patience=10,
    two_head=None,
    use_deep_sup=None,
    use_ohem=None,
    debris_boost=None,
    saved_models_dir="saved_models",
    run_dir=None,
    device=None,
    augmentation_enabled=True,
    use_amp=True,
    num_workers=0,
    pin_memory=None,
    cooldown_between_epochs=0.0,
    resume=True,
    max_steps_per_epoch=None,
    recipe="pretrained_strong",
    ssl4eo_strict=None,
):
    model_key = model_key.lower()
    if recipe not in RECIPE_PRESETS:
        raise ValueError(f"Recipe necunoscut: {recipe}. Disponibile: {list(RECIPE_PRESETS)}")
    preset = RECIPE_PRESETS[recipe]

    if two_head is None:
        two_head = (model_key in ("taunet", "taunet_resnet50")) and preset["two_head_default"]
    if use_deep_sup is None:
        use_deep_sup = preset["use_deep_sup"]
    if use_ohem is None:
        use_ohem = preset["use_ohem"]
    if debris_boost is None:
        debris_boost = preset["debris_boost"]
    plastic_boost = preset.get("plastic_boost", debris_boost * 2)
    if lr is None:
        lr = preset["lr"]

    deep_sup_weights = DEEP_SUP_WEIGHTS if use_deep_sup else []

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(use_amp and device.type == "cuda")
    if pin_memory is None:
        pin_memory = device.type == "cuda" and num_workers > 0
    os.makedirs(saved_models_dir, exist_ok=True)
    if run_dir:
        os.makedirs(run_dir, exist_ok=True)

    checkpoint_path = os.path.join(saved_models_dir, f"{model_key}_best.pth")
    checkpoint_ema_path = os.path.join(saved_models_dir, f"{model_key}_best_score_ema.pth")
    state_path = os.path.join(saved_models_dir, f"{model_key}_state.pth")
    checkpoint_mode = preset.get("checkpoint_mode", "score_ema")
    ckpt_md_iou_w = float(preset.get("checkpoint_md_iou_weight", 0.5))
    ckpt_plastic_iou_w = float(preset.get("checkpoint_plastic_iou_weight", 0.5))
    history_path = (
        os.path.join(run_dir, "istoric_antrenare.csv")
        if run_dir
        else f"istoric_antrenare_{model_key}.csv"
    )

    print_recipe_config(recipe, preset, two_head=two_head, model_key=model_key)
    print(f"\n{'=' * 60}\nAntrenare: {model_key} | device={device} | two_head={two_head}\n{'=' * 60}")

    seed = int(preset.get("seed", 42))
    set_training_seed(seed)
    print(f"[config] random seed={seed}")

    t_run = time.perf_counter()
    timing = {}

    cp_prob = preset.get("copy_paste_prob", 0.5)

    t_step = time.perf_counter()
    enable_copy_paste = augmentation_enabled or preset.get("copy_paste_on_no_aug", False)
    train_base = MARIDADataset(
        marida_path,
        split="train",
        transform=get_training_augmentation(enable=augmentation_enabled),
        enable_copy_paste=enable_copy_paste,
        copy_paste_prob=cp_prob if enable_copy_paste else 0.0,
        prefer_plastic=True,
        return_paste_stats=True,
    )
    use_crop_mining = preset.get("crop_mining", False)
    train_collate = None
    if use_crop_mining:
        full_patch_prob = float(preset.get("full_patch_prob", 0.5))
        train_dataset = MARIDACropMiningDataset(
            train_base,
            crop_size=preset.get("crop_size", 128),
            crop_prob=preset.get("crop_prob", 0.5),
            plastic_focus_prob=preset.get("plastic_focus_prob", 0.75),
            full_patch_prob=full_patch_prob,
        )
        train_collate = marida_pad_collate
        print(
            f"[crop_mining] mixed schedule: full_patch={full_patch_prob:.0%}, "
            f"crop={preset.get('crop_size', 128)}px, crop_prob={preset.get('crop_prob', 0.5)}, "
            f"plastic_focus={preset.get('plastic_focus_prob', 0.75)}"
        )
    else:
        train_dataset = train_base
    val_dataset = MARIDADataset(marida_path, split="val", transform=get_validation_augmentation())
    timing["setup_datasets_sec"] = sec_since(t_step)

    t_step = time.perf_counter()
    sampler = build_debris_weighted_sampler(
        train_base, debris_boost=debris_boost, plastic_boost=plastic_boost
    )
    timing["setup_sampler_sec"] = sec_since(t_step)

    t_step = time.perf_counter()
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=train_collate,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    timing["setup_dataloaders_sec"] = sec_since(t_step)

    t_step = time.perf_counter()
    use_ssag = preset.get("use_ssag", False)
    backbone = preset.get("backbone", "ssl4eo")
    if ssl4eo_strict is None:
        ssl4eo_strict = bool(preset.get("ssl4eo_strict", backbone == "ssl4eo"))
    if backbone == "ssl4eo":
        from models import verify_ssl4eo_backbone
        ok, src, shape = verify_ssl4eo_backbone(NUM_CHANNELS, strict=ssl4eo_strict)
        print(f"[backbone] SSL4EO preflight: ok={ok} source={src} conv1={shape}")
    model = build_model(
        model_key,
        in_channels=NUM_CHANNELS,
        out_classes=4,
        two_head=two_head,
        deep_supervision=use_deep_sup,
        use_ssag=use_ssag,
        backbone=backbone,
        ssl4eo_strict=ssl4eo_strict,
    ).to(device)
    if hasattr(model, "backbone_source"):
        print(f"[backbone] active encoder weights: {model.backbone_source}")
    timing["setup_model_sec"] = sec_since(t_step)

    t_step = time.perf_counter()
    if preset.get("class_weights_from_data") or preset["class_weights"] == "auto":
        stats = load_channel_stats(marida_path)
        weight_list = get_inverse_freq_weights(
            stats, max_weight=preset.get("class_weight_max", 50.0),
        )
        print(f"[recipe={recipe}] data-driven class weights: {[f'{w:.2f}' for w in weight_list]}")
    else:
        weight_list = preset["class_weights"]

    class_weights = torch.tensor(weight_list, dtype=torch.float32).to(device)
    label_smoothing = preset.get("label_smoothing", 0.0)
    training_mode = preset.get("training_mode", "standard")
    base_criterion = build_criterion(
        class_weights, two_head, use_ohem, deep_sup_weights, device,
        use_focal=preset["use_focal"],
        lovasz_weight=preset["lovasz_weight"],
        label_smoothing=label_smoothing,
        debris_weight=preset.get("debris_weight", 2.0),
        type_weight=preset.get("type_weight", 1.5),
        seg_weight=preset.get("seg_weight", 0.5),
        binary_pos_weight=preset.get("binary_pos_weight", 20.0),
        focal_gamma=preset.get("focal_gamma", 1.5),
        training_mode=training_mode,
        binary_lovasz_weight=preset.get("binary_lovasz_weight", 0.25),
    )

    boundary_weight = preset.get("boundary_weight", 0.0)
    boundary_loss_fn = BoundaryLoss().to(device) if boundary_weight > 0 else None
    grad_accum_steps = preset.get("grad_accum_steps", 1)

    class _CombinedCriterion(torch.nn.Module):
        def __init__(self, main, bdry, bdry_w):
            super().__init__()
            self.main = main
            self.bdry = bdry
            self.bdry_w = bdry_w
        def forward(self, outputs, target):
            loss = self.main(outputs, target)
            if self.bdry is not None and self.bdry_w > 0:
                seg = outputs["seg"] if isinstance(outputs, dict) else outputs
                loss = loss + self.bdry_w * self.bdry(seg, target)
            return loss

    criterion = _CombinedCriterion(base_criterion, boundary_loss_fn, boundary_weight)

    print(
        f"[recipe={recipe}] weights={[f'{w:.2f}' for w in weight_list]} "
        f"focal={preset['use_focal']} lovasz={preset['lovasz_weight']} "
        f"ohem={use_ohem} deep_sup={use_deep_sup} two_head={two_head} "
        f"debris_boost={debris_boost} plastic_boost={plastic_boost} lr={lr:.0e} "
        f"decoder_lr_mult={preset.get('decoder_lr_mult', 1.0)} "
        f"label_smoothing={label_smoothing} warmup={preset.get('warmup_epochs', 0)} "
        f"ssag={use_ssag} boundary={boundary_weight} accum={grad_accum_steps}"
    )

    decoder_lr_mult = preset.get("decoder_lr_mult", 1.0)
    if decoder_lr_mult > 1.0 and hasattr(model, "encoder_parameters") and hasattr(model, "decoder_parameters"):
        encoder_params = list(model.encoder_parameters())
        decoder_params = list(model.decoder_parameters())
        decoder_lr = lr * decoder_lr_mult
        param_groups = [
            {"params": encoder_params, "lr": lr, "group": "encoder"},
            {"params": decoder_params, "lr": decoder_lr, "group": "decoder"},
        ]
        optimizer = optim.AdamW(param_groups, weight_decay=1e-4)
        print(f"[optimizer] Differential LR: encoder={lr:.1e}, decoder={decoder_lr:.1e} "
              f"({len(encoder_params)} enc params, {len(decoder_params)} dec params)")
    else:
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    warmup_epochs = preset.get("warmup_epochs", 0)
    if warmup_epochs > 0:
        scheduler = LinearWarmupCosineScheduler(optimizer, warmup_epochs, epochs)
    else:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    timing["setup_optimizer_sec"] = sec_since(t_step)

    ema_decay = float(preset.get("ema_decay", 0.999))
    metric_ema_decay = float(preset.get("metric_ema_decay", 0.85))
    ckpt_miou_w = float(preset.get("checkpoint_miou_weight", 0.25))
    ckpt_plastic_w = float(preset.get("checkpoint_plastic_weight", 0.35))
    ckpt_fg_w = float(preset.get("checkpoint_fg_binary_weight", 0.40))
    ckpt_md_w = preset.get("checkpoint_md_weight")
    ckpt_plastic_rec_w = preset.get("checkpoint_plastic_recall_weight")
    val_fg_threshold = preset.get("val_fg_threshold", 0.20)
    val_debris_threshold = preset.get("val_debris_threshold", 0.35)
    encoder_unfreeze_ramp = int(preset.get("encoder_unfreeze_ramp_epochs", 0))
    encoder_lr_ramp_mult = float(preset.get("encoder_lr_ramp_mult", 0.25))
    score_ema = MetricEMA(decay=metric_ema_decay)

    best_val_miou = -1.0
    best_val_md_iou = -1.0
    best_val_plastic_iou = -1.0
    best_val_md_plastic_score = -1.0
    best_val_md_plastic_epoch = 0
    best_checkpoint_score = -1.0
    best_checkpoint_score_ema = -1.0
    epochs_no_improve = 0
    epochs_ran = 0
    start_epoch = 0
    istoric = {
        "train_loss": [], "val_loss": [], "val_miou_fg": [], "val_fg_binary_iou": [],
        "val_md_iou": [], "val_binary_debris_iou": [], "val_plastic_iou": [],
        "val_plastic_recall": [],
        "val_md_plastic_ckpt_score": [],
        "val_checkpoint_score": [], "val_checkpoint_score_ema": [], "lr": [],
        "epoch_sec": [], "train_sec": [], "val_sec": [], "val_miou_sec": [],
        "epoch_fmt": [], "train_fmt": [], "val_fmt": [],
    }
    paste_stats_all = {"pasted_pixels": 0, "total_pixels": 0}

    t_step = time.perf_counter()
    if resume and os.path.isfile(state_path):
        try:
            ckpt = torch.load(state_path, map_location=device)
            model.load_state_dict(ckpt["model"])
            optimizer.load_state_dict(ckpt["optimizer"])
            scheduler.load_state_dict(ckpt["scheduler"])
            start_epoch = int(ckpt.get("epoch", 0))
            best_val_miou = float(ckpt.get("best_val_miou", -1.0))
            epochs_no_improve = int(ckpt.get("epochs_no_improve", 0))
            istoric = ckpt.get("istoric", istoric)
            for key in (
                "epoch_sec", "train_sec", "val_sec", "val_miou_sec",
                "epoch_fmt", "train_fmt", "val_fmt",
                "val_fg_binary_iou", "val_md_iou", "val_binary_debris_iou",
                "val_plastic_iou", "val_plastic_recall", "val_md_plastic_ckpt_score",
                "val_checkpoint_score", "val_checkpoint_score_ema",
            ):
                istoric.setdefault(key, [])
            best_checkpoint_score = float(ckpt.get("best_checkpoint_score", best_checkpoint_score))
            best_checkpoint_score_ema = float(
                ckpt.get("best_checkpoint_score_ema", best_checkpoint_score_ema)
            )
            best_val_md_iou = float(ckpt.get("best_val_md_iou", best_val_md_iou))
            best_val_plastic_iou = float(ckpt.get("best_val_plastic_iou", best_val_plastic_iou))
            best_val_md_plastic_score = float(
                ckpt.get("best_val_md_plastic_score", best_val_md_plastic_score)
            )
            best_val_md_plastic_epoch = int(ckpt.get("best_val_md_plastic_epoch", best_val_md_plastic_epoch))
            paste_stats_all = ckpt.get("paste_stats_all", paste_stats_all)
            print(f"Resumed from {state_path} at epoch {start_epoch + 1}")
        except Exception as exc:
            print(f"Resume failed ({exc}); starting from scratch.")
    timing["resume_sec"] = sec_since(t_step)
    model_ema = ModelEMA(model, decay=ema_decay) if ema_decay > 0 else None

    _script_dir = os.path.dirname(os.path.abspath(__file__))
    stop_file = os.path.join(_script_dir, "STOP")

    def _save_state(epoch_done):
        try:
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "epoch": epoch_done,
                    "best_val_miou": best_val_miou,
                    "best_val_md_iou": best_val_md_iou,
                    "best_val_plastic_iou": best_val_plastic_iou,
                    "best_val_md_plastic_score": best_val_md_plastic_score,
                    "best_val_md_plastic_epoch": best_val_md_plastic_epoch,
                    "best_checkpoint_score": best_checkpoint_score,
                    "best_checkpoint_score_ema": best_checkpoint_score_ema,
                    "checkpoint_mode": checkpoint_mode,
                    "epochs_no_improve": epochs_no_improve,
                    "istoric": istoric,
                    "paste_stats_all": paste_stats_all,
                },
                state_path,
            )
        except Exception as exc:
            print(f"[warning] State save failed ({exc}).")

    interrupted = False
    t_loop = time.perf_counter()
    for epoch in range(start_epoch, epochs):
        t_epoch = time.perf_counter()
        epochs_ran = epoch + 1
        print(f"\nEpoch {epoch + 1}/{epochs}")
        lr_curent = scheduler.get_last_lr()[0]

        freeze_enc_epochs = preset.get("freeze_encoder_epochs", 0)
        if hasattr(model, "freeze_encoder") and freeze_enc_epochs > 0:
            if epoch < freeze_enc_epochs:
                model.freeze_encoder(True)
            elif epoch == freeze_enc_epochs:
                model.freeze_encoder(False)
                print("Encoder deblocat — fine-tuning complet.")
                if encoder_unfreeze_ramp > 0:
                    for pg in optimizer.param_groups:
                        if pg.get("group") == "encoder":
                            pg["lr"] = lr * encoder_lr_ramp_mult
                    print(
                        f"[optimizer] Encoder LR ramp: {lr * encoder_lr_ramp_mult:.1e} "
                        f"for {encoder_unfreeze_ramp} epochs"
                    )
            elif (
                encoder_unfreeze_ramp > 0
                and epoch == freeze_enc_epochs + encoder_unfreeze_ramp
            ):
                for pg in optimizer.param_groups:
                    if pg.get("group") == "encoder":
                        pg["lr"] = lr
                print(f"[optimizer] Encoder LR restored to {lr:.1e}")

        try:
            t_train = time.perf_counter()
            train_loss, paste_stats = train_one_epoch(
                model, train_loader, criterion, optimizer, device,
                use_amp=use_amp,
                max_steps=max_steps_per_epoch,
                grad_accum_steps=grad_accum_steps,
                model_ema=model_ema,
            )
            train_sec = sec_since(t_train)

            t_val = time.perf_counter()
            val_loss = validate(model, val_loader, criterion, device)
            val_part_sec = sec_since(t_val)

            t_miou = time.perf_counter()
            if training_mode == "md_outlier":
                val_metrics = compute_md_outlier_metrics(
                    model, val_loader, device,
                    debris_threshold=val_debris_threshold, verbose=True,
                )
            else:
                val_metrics = compute_val_metrics(
                    model, val_loader, device, two_head=False,
                    fg_threshold=val_fg_threshold, verbose=True,
                )
                if two_head:
                    th_metrics = compute_val_metrics(
                        model, val_loader, device, two_head=True, verbose=False,
                    )
                    print(
                        f"  [two-head decode] mIoU (fg): {th_metrics['mIoU_foreground']:.4f} | "
                        f"Plastic IoU: {th_metrics['plastic_IoU']:.4f} | "
                        f"Fg-bin IoU: {th_metrics['binary_foreground_IoU']:.4f}"
                    )
            val_miou_fg = val_metrics["mIoU_foreground"]
            val_plastic_iou = val_metrics["plastic_IoU"]
            val_fg_binary = val_metrics["binary_foreground_IoU"]
            val_md_iou = val_metrics.get("marida_md_IoU", val_metrics.get("binary_debris_IoU", 0.0))
            val_binary_debris = val_metrics.get("binary_debris_IoU", val_md_iou)
            val_plastic_recall = val_metrics.get("plastic_recall", 0.0)
            if ckpt_md_w is not None:
                val_ckpt_score = checkpoint_score(
                    val_metrics,
                    md_weight=float(ckpt_md_w),
                    plastic_weight=ckpt_plastic_w,
                    plastic_recall_weight=float(ckpt_plastic_rec_w or 0.0),
                )
            else:
                val_ckpt_score = checkpoint_score(
                    val_metrics,
                    miou_weight=ckpt_miou_w,
                    plastic_weight=ckpt_plastic_w,
                    fg_binary_weight=ckpt_fg_w,
                )
            val_ckpt_score_ema = score_ema.update(val_ckpt_score)
            val_md_plastic_ckpt = md_plastic_checkpoint_score(
                val_metrics,
                md_weight=ckpt_md_iou_w,
                plastic_weight=ckpt_plastic_iou_w,
            )
            val_miou_sec = sec_since(t_miou)
            val_sec = round(val_part_sec + val_miou_sec, 2)
            epoch_sec = sec_since(t_epoch)
        except KeyboardInterrupt:
            print("\n[STOP] Ctrl+C detectat. Salvare stare și oprire…")
            _save_state(epoch)
            interrupted = True
            break

        paste_stats_all["pasted_pixels"] += paste_stats["pasted_pixels"]
        paste_stats_all["total_pixels"] += paste_stats["total_pixels"]

        ema_info = ""
        if paste_stats.get("ema_updates"):
            ema_info = f" | EMA steps: {paste_stats['ema_updates']}"
        md_line = f" | MD IoU: {val_md_iou:.4f}" if training_mode == "md_outlier" else ""
        print(
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"Val mIoU (fg): {val_miou_fg:.4f} | Plastic IoU: {val_plastic_iou:.4f} "
            f"(R={val_plastic_recall:.3f}) | Fg-bin IoU: {val_fg_binary:.4f}{md_line} | "
            f"Score: {val_ckpt_score:.4f} (EMA {val_ckpt_score_ema:.4f}) | "
            f"MD+Pl: {val_md_plastic_ckpt:.4f} | LR: {lr_curent:.6f}{ema_info} | "
            f"Time: train {fmt_duration(train_sec)} val {fmt_duration(val_sec)} "
            f"(ep {fmt_duration(epoch_sec)})"
        )

        if val_miou_fg > best_val_miou:
            best_val_miou = val_miou_fg

        if val_ckpt_score_ema > best_checkpoint_score_ema:
            best_checkpoint_score_ema = val_ckpt_score_ema
            best_checkpoint_score = val_ckpt_score
            torch.save(model.state_dict(), checkpoint_ema_path)
            if model_ema is not None:
                ema_path = checkpoint_path.replace("_best.pth", "_ema_best.pth")
                torch.save(model_ema.state_dict(model), ema_path)

        use_md_plastic_ckpt = checkpoint_mode == "md_plastic"
        ckpt_improved = (
            val_md_plastic_ckpt > best_val_md_plastic_score
            if use_md_plastic_ckpt
            else val_ckpt_score_ema > best_checkpoint_score_ema
        )
        if ckpt_improved:
            if use_md_plastic_ckpt:
                best_val_md_plastic_score = val_md_plastic_ckpt
                best_val_md_iou = val_md_iou
                best_val_plastic_iou = val_plastic_iou
                best_val_md_plastic_epoch = epoch + 1
            epochs_no_improve = 0
            torch.save(model.state_dict(), checkpoint_path)
            if use_md_plastic_ckpt:
                save_msg = (
                    f"*** Salvat MD+plastic (score={best_val_md_plastic_score:.4f}, "
                    f"MD={val_md_iou:.4f}, plastic={val_plastic_iou:.4f}, ep={epoch + 1}) "
                    f"-> {checkpoint_path} ***"
                )
            else:
                save_msg = (
                    f"*** Salvat (score EMA={best_checkpoint_score_ema:.4f}, "
                    f"mIoU={val_miou_fg:.4f}, plastic={val_plastic_iou:.4f}, "
                    f"fg-bin={val_fg_binary:.4f}) -> {checkpoint_path} ***"
                )
            print(save_msg)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping după {patience} epoci fără îmbunătățire.")
                break

        istoric["train_loss"].append(train_loss)
        istoric["val_loss"].append(val_loss)
        istoric["val_miou_fg"].append(val_miou_fg)
        istoric["val_fg_binary_iou"].append(val_fg_binary)
        istoric["val_md_iou"].append(val_md_iou)
        istoric["val_binary_debris_iou"].append(val_binary_debris)
        istoric["val_plastic_iou"].append(val_plastic_iou)
        istoric["val_plastic_recall"].append(val_plastic_recall)
        istoric["val_md_plastic_ckpt_score"].append(val_md_plastic_ckpt)
        istoric["val_checkpoint_score"].append(val_ckpt_score)
        istoric["val_checkpoint_score_ema"].append(val_ckpt_score_ema)
        istoric["lr"].append(lr_curent)
        istoric["epoch_sec"].append(epoch_sec)
        istoric["train_sec"].append(train_sec)
        istoric["val_sec"].append(val_sec)
        istoric["val_miou_sec"].append(val_miou_sec)
        istoric["epoch_fmt"].append(fmt_duration(epoch_sec))
        istoric["train_fmt"].append(fmt_duration(train_sec))
        istoric["val_fmt"].append(fmt_duration(val_sec))
        scheduler.step()
        if device.type == "cuda":
            torch.cuda.empty_cache()

        _save_state(epoch + 1)

        if device.type == "cuda":
            thermal_guard(warn_temp=78.0, critical_temp=85.0)
        if cooldown_between_epochs > 0:
            print(f"[cooldown] sleeping {cooldown_between_epochs:.0f}s …")
            time.sleep(cooldown_between_epochs)

        if os.path.isfile(stop_file):
            print(f"\n[STOP] Fișier STOP detectat ({stop_file}). Oprire grațioasă.")
            try:
                os.remove(stop_file)
            except OSError:
                pass
            interrupted = True
            break

    timing["training_loop_sec"] = sec_since(t_loop)
    timing["total_sec"] = sec_since(t_run)
    if istoric["epoch_sec"]:
        timing["avg_epoch_sec"] = round(sum(istoric["epoch_sec"]) / len(istoric["epoch_sec"]), 2)
        timing["sum_train_sec"] = round(sum(istoric["train_sec"]), 2)
        timing["sum_val_sec"] = round(sum(istoric["val_sec"]), 2)

    pd.DataFrame(istoric).to_csv(history_path, index=False)
    print(f"Istoric salvat: {history_path}")
    dashboard_dir = run_dir or saved_models_dir
    try:
        dash_paths = plot_training_dashboard(
            history_path, dashboard_dir, f"{model_key} ({recipe})",
        )
        if dash_paths:
            print(f"Dashboard metrici: {', '.join(os.path.basename(p) for p in dash_paths)}")
    except Exception as exc:
        print(f"[warning] plot_training_dashboard failed: {exc}")
    paste_ratio = paste_stats_all["pasted_pixels"] / max(1, paste_stats_all["total_pixels"])

    timing = format_timing_dict(timing)
    print_timing_block(f"Antrenare {model_key}", timing)

    return {
        "model_key": model_key,
        "checkpoint_path": checkpoint_path,
        "history_path": history_path,
        "best_val_miou_fg": best_val_miou,
        "best_val_md_iou": best_val_md_iou,
        "best_val_plastic_iou": best_val_plastic_iou,
        "best_val_md_plastic_score": best_val_md_plastic_score,
        "best_val_md_plastic_epoch": best_val_md_plastic_epoch,
        "checkpoint_mode": checkpoint_mode,
        "checkpoint_ema_path": checkpoint_ema_path,
        "best_checkpoint_score": best_checkpoint_score,
        "best_checkpoint_score_ema": best_checkpoint_score_ema,
        "epochs_ran": epochs_ran,
        "interrupted": interrupted,
        "two_head": two_head,
        "use_amp": use_amp,
        "recipe": recipe,
        "training_mode": training_mode,
        "backbone": preset.get("backbone", "ssl4eo"),
        "backbone_source": getattr(model, "backbone_source", None),
        "class_weights": weight_list,
        "use_focal": preset["use_focal"],
        "lovasz_weight": preset["lovasz_weight"],
        "use_ohem": use_ohem,
        "use_deep_sup": use_deep_sup,
        "use_ssag": use_ssag,
        "boundary_weight": boundary_weight,
        "grad_accum_steps": grad_accum_steps,
        "augmentation_stats": {
            "copy_paste_pasted_pixels": int(paste_stats_all["pasted_pixels"]),
            "copy_paste_total_pixels": int(paste_stats_all["total_pixels"]),
            "copy_paste_paste_ratio": float(paste_ratio),
        },
        "duration_sec": timing["total_sec"],
        "duration_fmt": timing.get("total_fmt", fmt_duration(timing["total_sec"])),
        "timing": timing,
    }


def parse_args():
    p = argparse.ArgumentParser(description="Antrenare segmentare MARIDA — Disertatie_2")
    p.add_argument("--marida", type=str, default=r"D:\TAID\Disertatie\MARIDA")
    p.add_argument("--model", type=str, default="taunet_resnet50",
                   choices=["taunet", "taunet_resnet50", "resunext", "unet_resnet50"])
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--batch-size", type=int, default=6)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--patience", type=int, default=35)
    p.add_argument("--no-two-head", action="store_true")
    p.add_argument("--no-deep-sup", action="store_true")
    p.add_argument("--no-ohem", action="store_true")
    p.add_argument("--debris-boost", type=float, default=None)
    p.add_argument("--num-workers", type=int, default=6)
    p.add_argument("--pin-memory", action="store_true")
    p.add_argument("--recipe", type=str, default="md_outlier",
                   choices=list(RECIPE_PRESETS.keys()))
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result = run_training(
        args.marida,
        args.model.lower(),
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience,
        two_head=(args.model.lower() in ("taunet", "taunet_resnet50")) and not args.no_two_head,
        use_deep_sup=False if args.no_deep_sup else None,
        use_ohem=False if args.no_ohem else None,
        debris_boost=args.debris_boost,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        recipe=args.recipe,
    )
    print(f"\nFinalizat. Cel mai bun mIoU foreground: {result['best_val_miou_fg']:.4f}")
