"""Training stability utilities: EMA weights, reproducibility, checkpoint scoring."""

from __future__ import annotations

import copy
import random

import numpy as np
import torch


def set_training_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def _ema_trackable(tensor: torch.Tensor) -> bool:
    """Only float weights/buffers are EMA-blended; BatchNorm counters stay int."""
    return tensor.is_floating_point()


class ModelEMA:
    """Exponential moving average of model weights for stable validation."""

    def __init__(self, model: torch.nn.Module, decay: float = 0.999):
        self.decay = decay
        state = model.state_dict()
        self.shadow = {
            k: v.detach().clone()
            for k, v in state.items()
            if _ema_trackable(v)
        }
        skipped = [k for k in state if k not in self.shadow]
        if skipped:
            print(f"[ema] Skipping {len(skipped)} non-float state keys (e.g. num_batches_tracked)")

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        for key, value in model.state_dict().items():
            if key not in self.shadow:
                continue
            self.shadow[key].mul_(self.decay).add_(value.detach(), alpha=1.0 - self.decay)

    def state_dict(self, model: torch.nn.Module) -> dict:
        """EMA floats merged with the model's live integer buffers for checkpointing."""
        merged = {k: v.detach().clone() for k, v in self.shadow.items()}
        for key, value in model.state_dict().items():
            if key not in merged:
                merged[key] = value.detach().clone()
        return merged

    def apply(self, model: torch.nn.Module) -> dict:
        backup = {k: v.detach().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(self.state_dict(model), strict=True)
        return backup

    @staticmethod
    def restore(model: torch.nn.Module, backup: dict) -> None:
        model.load_state_dict(backup, strict=True)


def checkpoint_score(
    metrics: dict,
    *,
    miou_weight: float = 0.25,
    plastic_weight: float = 0.35,
    fg_binary_weight: float = 0.40,
    md_weight: float | None = None,
    plastic_recall_weight: float | None = None,
) -> float:
    """Composite validation score — higher is better."""
    plastic = float(metrics.get("plastic_IoU", metrics.get("per_class", {}).get(1, {}).get("IoU", 0.0)))
    plastic_rec = float(
        metrics.get("plastic_recall", metrics.get("per_class", {}).get(1, {}).get("Recall", 0.0))
    )
    if md_weight is not None:
        md = float(metrics.get("marida_md_IoU", metrics.get("binary_debris_IoU", 0.0)))
        pr_w = plastic_recall_weight if plastic_recall_weight is not None else 0.0
        total_w = md_weight + plastic_weight + pr_w
        if total_w <= 0:
            return md
        return (md_weight * md + plastic_weight * plastic + pr_w * plastic_rec) / total_w

    miou = float(metrics.get("mIoU_foreground", 0.0))
    fg_bin = float(metrics.get("binary_foreground_IoU", metrics.get("binary_debris_IoU", 0.0)))
    total_w = miou_weight + plastic_weight + fg_binary_weight
    if total_w <= 0:
        return miou
    return (miou_weight * miou + plastic_weight * plastic + fg_binary_weight * fg_bin) / total_w


class MetricEMA:
    """Smooth validation metrics across epochs for checkpoint / early stopping."""

    def __init__(self, decay: float = 0.9):
        self.decay = decay
        self.value: float | None = None

    def update(self, raw: float) -> float:
        if self.value is None:
            self.value = float(raw)
        else:
            self.value = self.decay * self.value + (1.0 - self.decay) * float(raw)
        return self.value
