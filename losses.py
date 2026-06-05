"""Loss functions for MARIDA multi-class segmentation."""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _as_long_target(target, num_classes=4):
    """Cross-entropy, one_hot, and type labels require integer class indices."""
    if not torch.is_tensor(target):
        target = torch.as_tensor(target, dtype=torch.long)
    if target.dtype != torch.long:
        target = target.long()
    if target.ndim == 4 and target.shape[1] == 1:
        target = target.squeeze(1)
    elif target.ndim > 3:
        target = target.reshape(target.shape[0], *target.shape[-2:])
    return target.clamp(0, num_classes - 1)


def _type_labels_from_target(target, num_classes=4):
    """Type head labels: 255 = ignore (water), else class_id - 1."""
    target = _as_long_target(target, num_classes)
    fg = target > 0
    type_vals = torch.where(fg, target - 1, torch.zeros_like(target))
    return torch.where(fg, type_vals, torch.full_like(target, 255))


def _lovasz_grad(gt_sorted):
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1.0 - intersection / union.clamp(min=1e-8)
    if p > 1:
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def lovasz_softmax_flat(probas, labels, classes="present"):
    if probas.numel() == 0:
        return probas * 0.0
    c = probas.size(0)
    losses = []
    class_range = range(c) if classes == "all" else range(1, c)
    for cls in class_range:
        fg = (labels == cls).float()
        if classes == "present" and fg.sum() == 0:
            continue
        class_pred = probas[cls]
        errors = (fg - class_pred).abs()
        errors_sorted, perm = torch.sort(errors, descending=True)
        fg_sorted = fg[perm]
        losses.append(torch.dot(errors_sorted, _lovasz_grad(fg_sorted)))
    if not losses:
        return probas.sum() * 0.0
    return torch.mean(torch.stack(losses))


def lovasz_softmax(probas, labels, classes="present"):
    labels = _as_long_target(labels, probas.shape[1])
    losses = []
    for prob, lab in zip(probas, labels):
        c = prob.shape[0]
        prob_flat = prob.reshape(c, -1)
        lab_flat = lab.reshape(-1)
        losses.append(lovasz_softmax_flat(prob_flat, lab_flat, classes=classes))
    return torch.mean(torch.stack(losses))


class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0, ignore_index=255):
        super().__init__()
        self.weight = weight
        self.gamma = gamma
        self.ignore_index = ignore_index

    def forward(self, logits, target):
        target = _as_long_target(target, logits.shape[1])
        ce = F.cross_entropy(
            logits, target, weight=self.weight, reduction="none", ignore_index=self.ignore_index
        )
        pt = torch.exp(-ce)
        focal = ((1 - pt) ** self.gamma) * ce
        valid = target != self.ignore_index
        if valid.sum() == 0:
            return focal.sum() * 0.0
        return focal[valid].mean()


class HybridLoss(nn.Module):
    """Weighted CE or Focal + foreground Dice + optional Lovász."""

    def __init__(
        self,
        weight=None,
        ce_weight=1.0,
        dice_weight=1.0,
        lovasz_weight=0.0,
        use_focal=True,
        focal_gamma=2.0,
        label_smoothing=0.0,
    ):
        super().__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.lovasz_weight = lovasz_weight
        if use_focal:
            self.cls_loss = FocalLoss(weight=weight, gamma=focal_gamma)
        else:
            self.cls_loss = nn.CrossEntropyLoss(weight=weight, label_smoothing=label_smoothing)

    def dice_loss(self, pred, target, smooth=1e-5):
        pred = pred.float()
        pred_probs = F.softmax(pred, dim=1)
        target = _as_long_target(target, pred.shape[1])
        target_one_hot = F.one_hot(target, num_classes=pred.shape[1]).permute(0, 3, 1, 2).float()
        intersection = (pred_probs * target_one_hot).sum(dim=(2, 3))
        union = pred_probs.sum(dim=(2, 3)) + target_one_hot.sum(dim=(2, 3))
        dice_score = (2.0 * intersection + smooth) / (union + smooth)
        dice_score = torch.clamp(dice_score, 0.0, 1.0)

        fg_scores = dice_score[:, 1:]
        fg_present = target_one_hot[:, 1:].sum(dim=(2, 3)) > 0
        if fg_present.any():
            return 1.0 - fg_scores[fg_present].mean()
        return pred.sum() * 0.0

    def forward(self, pred, target):
        if isinstance(pred, dict):
            pred = pred["seg"]
        pred = pred.float()
        target = _as_long_target(target, pred.shape[1])
        ce = self.cls_loss(pred, target)
        dice = self.dice_loss(pred, target)
        loss = self.ce_weight * ce + self.dice_weight * dice
        if self.lovasz_weight > 0:
            loss = loss + self.lovasz_weight * lovasz_softmax(
                F.softmax(pred, dim=1), target, classes="present"
            )
        return loss


class DeepSupervisionHybridLoss(HybridLoss):
    def __init__(self, deep_sup_weights=None, **kwargs):
        super().__init__(**kwargs)
        self.deep_sup_weights = deep_sup_weights or []

    def forward(self, outputs, target):
        if isinstance(outputs, dict):
            loss = super().forward(outputs["seg"], target)
            for w, aux in zip(self.deep_sup_weights, outputs.get("aux", [])):
                loss = loss + w * super().forward(aux, target)
            return loss
        return super().forward(outputs, target)


class TwoHeadHybridLoss(nn.Module):
    def __init__(
        self,
        class_weight=None,
        debris_weight=2.0,
        type_weight=1.5,
        seg_weight=0.5,
        deep_sup_weights=None,
        use_focal=True,
        lovasz_weight=0.3,
        binary_pos_weight=20.0,
        focal_gamma=1.5,
    ):
        super().__init__()
        self.seg_loss = HybridLoss(
            weight=class_weight,
            use_focal=use_focal,
            lovasz_weight=lovasz_weight,
            focal_gamma=focal_gamma,
        )
        self.seg_weight = seg_weight
        self.debris_weight = debris_weight
        self.type_weight = type_weight
        pos_w = torch.tensor([binary_pos_weight], dtype=torch.float32)
        self.register_buffer("_pos_weight", pos_w)
        self.deep_sup_weights = deep_sup_weights or []

    def forward(self, outputs, target):
        target = _as_long_target(target, outputs["seg"].shape[1])
        loss = self.seg_weight * self.seg_loss(outputs, target)
        fg_target = (target > 0).float().unsqueeze(1)
        loss = loss + self.debris_weight * F.binary_cross_entropy_with_logits(
            outputs["debris"], fg_target, pos_weight=self._pos_weight,
        )
        type_labels = _type_labels_from_target(target, outputs["seg"].shape[1])
        if (type_labels != 255).any():
            loss = loss + self.type_weight * F.cross_entropy(
                outputs["type"], type_labels, ignore_index=255,
            )
        for w, aux_logits in zip(self.deep_sup_weights, outputs.get("aux", [])):
            loss = loss + w * self.seg_weight * self.seg_loss(aux_logits, target)
        return loss


def ohem_cross_entropy(logits, target, weight=None, keep_ratio=0.25, min_kept=4096):
    target = _as_long_target(target, logits.shape[1])
    per_pixel = F.cross_entropy(logits, target, weight=weight, reduction="none")
    flat = per_pixel.view(-1)
    n = flat.numel()
    k = max(min_kept, int(n * keep_ratio))
    k = min(k, n)
    if k == 0:
        return flat.mean()
    hard, _ = torch.topk(flat, k)
    return hard.mean()


class OhemHybridLoss(nn.Module):
    def __init__(self, weight=None, dice_weight=1.0, lovasz_weight=0.3,
                 keep_ratio=0.25, use_focal=False):
        super().__init__()
        self.weight = weight
        self.dice_weight = dice_weight
        self.lovasz_weight = lovasz_weight
        self.keep_ratio = keep_ratio
        self.use_focal = use_focal
        self.base = HybridLoss(weight=weight, ce_weight=0.0, dice_weight=0.0,
                               lovasz_weight=0.0, use_focal=False)

    def forward(self, pred, target):
        if isinstance(pred, dict):
            pred = pred["seg"]
        target = _as_long_target(target, pred.shape[1])
        if self.use_focal:
            ce = FocalLoss(weight=self.weight)(pred, target)
        else:
            ce = ohem_cross_entropy(pred, target, weight=self.weight, keep_ratio=self.keep_ratio)
        loss = ce + self.dice_weight * self.base.dice_loss(pred, target)
        if self.lovasz_weight > 0:
            loss = loss + self.lovasz_weight * lovasz_softmax(
                F.softmax(pred, dim=1), target, classes="present"
            )
        return loss


class BoundaryLoss(nn.Module):
    def __init__(self, num_classes=4, foreground_classes=(1, 2, 3)):
        super().__init__()
        self.num_classes = num_classes
        self.fg_classes = foreground_classes

    @staticmethod
    def _boundary_distance_map(mask, cls_id, sigma=3.0):
        import numpy as _np
        from scipy.ndimage import distance_transform_edt
        binary = (mask == cls_id).astype(_np.float32)
        if binary.sum() < 1:
            return _np.zeros_like(binary)
        inner = distance_transform_edt(binary)
        outer = distance_transform_edt(1.0 - binary)
        dist = _np.minimum(inner, outer)
        return _np.exp(-dist / sigma)

    def forward(self, pred_logits, target):
        import numpy as _np
        pred_probs = F.softmax(pred_logits.float(), dim=1)
        target_np = target.detach().cpu().numpy()
        batch_loss = []
        for b in range(pred_logits.shape[0]):
            sample_loss = []
            for cls_id in self.fg_classes:
                w_map = self._boundary_distance_map(target_np[b], cls_id)
                if w_map.sum() < 1e-6:
                    continue
                w_tensor = torch.from_numpy(w_map).to(
                    pred_logits.device, dtype=pred_logits.dtype
                )
                wrong_prob = 1.0 - pred_probs[b, cls_id]
                sample_loss.append((wrong_prob * w_tensor).mean())
            if sample_loss:
                batch_loss.append(torch.stack(sample_loss).mean())
        if not batch_loss:
            return torch.tensor(0.0, device=pred_logits.device, requires_grad=True)
        return torch.stack(batch_loss).mean()


def _binary_lovasz_hinge(logits, target_bool):
    """Lovász hinge for single-channel binary logits (both must be B×H×W)."""
    if logits.ndim == 4:
        logits = logits.squeeze(1)
    if target_bool.ndim == 4:
        target_bool = target_bool.squeeze(1)
    prob = torch.sigmoid(logits)
    fg = target_bool.float()
    errors = (fg - prob).abs()
    errors_sorted, perm = torch.sort(errors.reshape(-1), descending=True)
    fg_sorted = fg.reshape(-1)[perm]
    if fg_sorted.sum() == 0:
        return logits.sum() * 0.0
    grad = _lovasz_grad(fg_sorted)
    return torch.dot(errors_sorted, grad)


class MDOutlierLoss(nn.Module):
    """
    Outlier-first MARIDA loss: binary MD detection (debris head) is primary;
    light 4-class seg + type heads refine plastic / natural / ship on foreground.
    """

    def __init__(
        self,
        class_weight=None,
        debris_weight=3.0,
        type_weight=0.5,
        seg_weight=0.25,
        binary_lovasz_weight=0.25,
        deep_sup_weights=None,
        use_focal=True,
        lovasz_weight=0.10,
        binary_pos_weight=15.0,
        focal_gamma=2.0,
        ohem_ratio=0.25,
    ):
        super().__init__()
        self.debris_weight = debris_weight
        self.type_weight = type_weight
        self.seg_weight = seg_weight
        self.binary_lovasz_weight = binary_lovasz_weight
        self.ohem_ratio = ohem_ratio
        self.seg_loss = HybridLoss(
            weight=class_weight,
            use_focal=use_focal,
            lovasz_weight=lovasz_weight,
            focal_gamma=focal_gamma,
        )
        pos_w = torch.tensor([binary_pos_weight], dtype=torch.float32)
        self.register_buffer("_pos_weight", pos_w)
        self.deep_sup_weights = deep_sup_weights or []

    def _md_target(self, target):
        target = _as_long_target(target, 4)
        return ((target == 1) | (target == 2)).float().unsqueeze(1)

    def forward(self, outputs, target):
        target = _as_long_target(target, outputs["seg"].shape[1])
        md_target = self._md_target(target)
        debris_logits = outputs["debris"]
        bce = F.binary_cross_entropy_with_logits(
            debris_logits, md_target, pos_weight=self._pos_weight,
        )
        loss = self.debris_weight * bce
        if self.binary_lovasz_weight > 0:
            debris_sq = debris_logits.squeeze(1) if debris_logits.ndim == 4 else debris_logits
            loss = loss + self.binary_lovasz_weight * _binary_lovasz_hinge(
                debris_sq, md_target.squeeze(1).bool(),
            )
        if self.seg_weight > 0:
            loss = loss + self.seg_weight * self.seg_loss(outputs["seg"], target)
        type_labels = _type_labels_from_target(target, outputs["seg"].shape[1])
        if self.type_weight > 0 and (type_labels != 255).any():
            loss = loss + self.type_weight * F.cross_entropy(
                outputs["type"], type_labels, ignore_index=255,
            )
        for w, aux_logits in zip(self.deep_sup_weights, outputs.get("aux", [])):
            loss = loss + w * self.seg_weight * self.seg_loss(aux_logits, target)
        return loss


class OhemTwoHeadHybridLoss(TwoHeadHybridLoss):
    def __init__(self, keep_ratio=0.25, class_weight=None, **kwargs):
        super().__init__(class_weight=class_weight, **kwargs)
        self.keep_ratio = keep_ratio
        self.class_weight = class_weight

    def forward(self, outputs, target):
        target = _as_long_target(target, outputs["seg"].shape[1])
        seg_logits = outputs["seg"]
        ce = ohem_cross_entropy(seg_logits, target, weight=self.class_weight,
                                keep_ratio=self.keep_ratio)
        loss = ce
        loss = loss + self.seg_loss.dice_weight * self.seg_loss.dice_loss(seg_logits, target)
        if self.seg_loss.lovasz_weight > 0:
            loss = loss + self.seg_loss.lovasz_weight * lovasz_softmax(
                F.softmax(seg_logits, dim=1), target, classes="present"
            )
        fg_target = (target > 0).float().unsqueeze(1)
        loss = loss + self.debris_weight * F.binary_cross_entropy_with_logits(
            outputs["debris"], fg_target, pos_weight=self._pos_weight,
        )
        type_labels = _type_labels_from_target(target, seg_logits.shape[1])
        if (type_labels != 255).any():
            loss = loss + self.type_weight * F.cross_entropy(
                outputs["type"], type_labels, ignore_index=255,
            )
        for w, aux_logits in zip(self.deep_sup_weights, outputs.get("aux", [])):
            loss = loss + w * self.seg_weight * self.seg_loss(aux_logits, target)
        return loss
