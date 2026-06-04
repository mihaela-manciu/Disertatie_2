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


class ModelEMA:
    """Exponential moving average of model weights for stable validation."""

    def __init__(self, model: torch.nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        for key, value in model.state_dict().items():
            self.shadow[key].mul_(self.decay).add_(value.detach(), alpha=1.0 - self.decay)

    def apply(self, model: torch.nn.Module) -> dict:
        backup = {k: v.detach().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(self.shadow, strict=True)
        return backup

    @staticmethod
    def restore(model: torch.nn.Module, backup: dict) -> None:
        model.load_state_dict(backup, strict=True)


def checkpoint_score(metrics: dict, *, miou_weight: float = 0.5, fg_binary_weight: float = 0.5) -> float:
    """Composite validation score — higher is better."""
    miou = float(metrics.get("mIoU_foreground", 0.0))
    fg_bin = float(metrics.get("binary_foreground_IoU", metrics.get("binary_debris_IoU", 0.0)))
    return miou_weight * miou + fg_binary_weight * fg_bin


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
