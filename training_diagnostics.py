"""Training-time diagnostics: head stats, decode funnel, multi-decode IoU comparison."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from segmentation_utils import (
    CLASE_NUME,
    calculate_debris_metrics,
    decode_two_head,
    decode_softmax,
    model_forward,
    update_confusion_matrix,
)


def _pixel_counts(arr, num_classes=4):
    flat = np.asarray(arr).flatten()
    counts = {c: int((flat == c).sum()) for c in range(num_classes)}
    counts["total"] = int(flat.size)
    return counts


def _metrics_from_preds(target, pred):
    conf = np.zeros((4, 4), dtype=np.int64)
    update_confusion_matrix(conf, target, pred)
    return calculate_debris_metrics(conf)


def _head_stats(seg_logits, debris_logits, type_logits, device):
    with torch.no_grad():
        seg_p = torch.softmax(seg_logits.float(), dim=1)
        debris_p = torch.sigmoid(debris_logits.float())
        if debris_p.ndim == 4:
            debris_p = debris_p.squeeze(1)
        type_p = torch.softmax(type_logits.float(), dim=1)
        type_conf, type_cls = torch.max(type_p, dim=1)
        seg_pred = torch.argmax(seg_p, dim=1)

        d = debris_p.detach()
        stats = {
            "debris_min": float(d.min()),
            "debris_mean": float(d.mean()),
            "debris_max": float(d.max()),
            "debris_p95": float(torch.quantile(d.flatten(), 0.95)),
            "pct_debris_ge_35": float((d >= 0.35).float().mean()),
            "pct_debris_ge_42": float((d >= 0.42).float().mean()),
            "pct_debris_ge_50": float((d >= 0.50).float().mean()),
            "seg_fg_pct": float((seg_pred > 0).float().mean()),
            "seg_plastic_pct": float((seg_pred == 1).float().mean()),
            "seg_natural_pct": float((seg_pred == 2).float().mean()),
            "seg_ships_pct": float((seg_pred == 3).float().mean()),
            "type_conf_mean": float(type_conf.mean()),
            "type_conf_max": float(type_conf.max()),
            "pct_type_conf_ge_50": float((type_conf >= 0.50).float().mean()),
        }

        gate_debris = d >= 0.35
        gate_seg = gate_debris & (seg_pred > 0)
        gate_full = gate_seg & (type_conf >= 0.50)
        stats["gate_debris_35_px"] = int(gate_debris.sum())
        stats["gate_debris_and_seg_px"] = int(gate_seg.sum())
        stats["gate_full_px"] = int(gate_full.sum())
        stats["total_px"] = int(d.numel())
        return stats, seg_p, debris_p.unsqueeze(1) if debris_p.ndim == 3 else debris_p, type_p


def print_train_batch_diagnostics(images, masks, *, tag="train"):
    """One-time sanity check on a single training batch."""
    gt = masks.detach().cpu().numpy()
    if gt.ndim == 4:
        gt = gt[:, 0]
    counts = {c: 0 for c in range(4)}
    for b in range(gt.shape[0]):
        for c in range(4):
            counts[c] += int((gt[b] == c).sum())
    total = sum(counts.values())
    img = images.detach().float()
    print(f"\n[diag] === {tag} batch ===")
    print(f"[diag] image shape={tuple(img.shape)} dtype={img.dtype} "
          f"min={float(img.min()):.4f} max={float(img.max()):.4f} "
          f"mean={float(img.mean()):.4f} nan={bool(torch.isnan(img).any())}")
    print(f"[diag] mask  shape={tuple(masks.shape)} dtype={masks.dtype} "
          f"min={int(masks.min())} max={int(masks.max())}")
    for c in range(4):
        pct = 100.0 * counts[c] / max(1, total)
        print(f"[diag] GT class {c}: {counts[c]:6d} px ({pct:5.2f}%)")


def print_validation_diagnostics(
    model,
    dataloader,
    device,
    *,
    epoch: int,
    debris_threshold: float = 0.35,
    gated_debris_threshold: float = 0.42,
    gated_type_confidence: float = 0.50,
    gated_require_seg_fg: bool = True,
    max_batches: int = 2,
):
    """
    Print where the pipeline breaks: GT counts, head activations, gate funnel,
    IoU under seg-only / legacy two-head / strict gated decode.
    """
    model.eval()
    agg_gt = {c: 0 for c in range(4)}
    agg_head = None
    variants = {
        "seg_argmax": {"two_head": False},
        "two_head_legacy": {
            "two_head": True,
            "debris_threshold": debris_threshold,
            "type_confidence_threshold": 0.0,
            "require_seg_fg": False,
        },
        "two_head_gated": {
            "two_head": True,
            "debris_threshold": gated_debris_threshold,
            "type_confidence_threshold": gated_type_confidence,
            "require_seg_fg": gated_require_seg_fg,
        },
    }
    variant_preds = {k: [] for k in variants}
    variant_masks = []
    batches_seen = 0

    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(dataloader):
            if batch_idx >= max_batches:
                break
            batches_seen += 1
            images = images.to(device)
            masks_np = masks.numpy()
            if masks_np.ndim == 2:
                masks_np = masks_np[None, ...]

            out = model_forward(model, images)
            seg_logits = out["seg"]
            debris_logits = out["debris"]
            type_logits = out["type"]

            batch_stats, seg_p, debris_p, type_p = _head_stats(
                seg_logits, debris_logits, type_logits, device,
            )
            if agg_head is None:
                agg_head = {k: 0.0 for k in batch_stats}
            for k, v in batch_stats.items():
                if isinstance(v, (int, float)):
                    agg_head[k] = agg_head.get(k, 0) + v

            for b in range(masks_np.shape[0]):
                gt = masks_np[b]
                for c in range(4):
                    agg_gt[c] += int((gt == c).sum())
                variant_masks.append(gt)

                seg_only = decode_softmax(seg_p[b : b + 1])[0]
                variant_preds["seg_argmax"].append(seg_only)

                for name in ("two_head_legacy", "two_head_gated"):
                    cfg = variants[name]
                    pred = decode_two_head(
                        seg_p[b : b + 1],
                        debris_p[b : b + 1],
                        type_p[b : b + 1],
                        debris_threshold=cfg["debris_threshold"],
                        type_confidence_threshold=cfg["type_confidence_threshold"],
                        require_seg_fg=cfg["require_seg_fg"],
                    )[0]
                    variant_preds[name].append(pred)

    if agg_head is None:
        print("[diag] validation diagnostics: no batches")
        return

    n_batches = batches_seen
    for k in list(agg_head.keys()):
        if k.endswith("_px") or k == "total_px":
            continue
        agg_head[k] /= max(1, n_batches)

    total_gt = sum(agg_gt.values())
    print(f"\n[diag] === Epoch {epoch} validation diagnostics ({n_batches} batches) ===")
    print("[diag] --- Ground truth (sampled val pixels) ---")
    for c in range(4):
        pct = 100.0 * agg_gt[c] / max(1, total_gt)
        print(f"[diag]   GT {CLASE_NUME[c]}: {agg_gt[c]:8d} ({pct:5.2f}%)")

    print("[diag] --- Head activations (mean over batches) ---")
    print(f"[diag]   debris sigmoid: min={agg_head['debris_min']:.4f} "
          f"mean={agg_head['debris_mean']:.4f} max={agg_head['debris_max']:.4f} "
          f"p95={agg_head['debris_p95']:.4f}")
    print(f"[diag]   debris>=0.35: {100*agg_head['pct_debris_ge_35']:.2f}%  "
          f">=0.42: {100*agg_head['pct_debris_ge_42']:.2f}%  "
          f">=0.50: {100*agg_head['pct_debris_ge_50']:.2f}%")
    print(f"[diag]   seg predicts fg: {100*agg_head['seg_fg_pct']:.2f}%  "
          f"plastic={100*agg_head['seg_plastic_pct']:.3f}%  "
          f"natural={100*agg_head['seg_natural_pct']:.3f}%  "
          f"ships={100*agg_head['seg_ships_pct']:.3f}%")
    print(f"[diag]   type confidence: mean={agg_head['type_conf_mean']:.4f} "
          f"max={agg_head['type_conf_max']:.4f}  "
          f">=0.50: {100*agg_head['pct_type_conf_ge_50']:.2f}%")

    print("[diag] --- Decode gate funnel (thr=0.35 + seg_fg + type>=0.5) ---")
    print(f"[diag]   pixels passing debris>=0.35:     {agg_head['gate_debris_35_px']}")
    print(f"[diag]   + seg foreground:                  {agg_head['gate_debris_and_seg_px']}")
    print(f"[diag]   + type conf >= 0.5 (R15 gated):    {agg_head['gate_full_px']}")
    print(f"[diag]   (if gate_full ≈ 0 → gated metrics will be 0; use legacy decode for training)")

    print("[diag] --- IoU by decode variant (same val sample) ---")
    for name, preds in variant_preds.items():
        conf = np.zeros((4, 4), dtype=np.int64)
        for gt, pred in zip(variant_masks, preds):
            update_confusion_matrix(conf, gt, pred)
        m = calculate_debris_metrics(conf)
        pred_counts = _pixel_counts(np.concatenate([p.flatten() for p in preds]))
        print(
            f"[diag]   {name:18s} | MD IoU={m['marida_md_IoU']:.4f} "
            f"plastic={m['plastic_IoU']:.4f} mIoU_fg={m['mIoU_foreground']:.4f} | "
            f"pred px: water={pred_counts[0]} pl={pred_counts[1]} "
            f"nat={pred_counts[2]} ship={pred_counts[3]}"
        )


def print_loss_breakdown(model, criterion, images, masks, device, *, tag="val"):
    """Print per-term loss on one batch (MDOutlierLoss or generic)."""
    model.eval()
    images = images.to(device)
    masks = masks.to(device, dtype=torch.long)
    with torch.no_grad():
        out = model_forward(model, images)
        total = float(criterion(out, masks).item())
    print(f"\n[diag] === Loss breakdown ({tag} batch) total={total:.4f} ===")
    crit = criterion.main if hasattr(criterion, "main") else criterion
    if crit.__class__.__name__ != "MDOutlierLoss":
        print(f"[diag]   criterion={crit.__class__.__name__} — no term breakdown implemented")
        return

    from losses import _as_long_target, _binary_weighted_bce, _type_labels_from_target

    target = _as_long_target(masks, 4)
    md_target = ((target == 1) | (target == 2)).float().unsqueeze(1)
    md_pos_pct = 100.0 * float(md_target.mean())
    fg_pos_pct = 100.0 * float((target > 0).float().mean())
    debris_logits = torch.nan_to_num(
        out["debris"].float(), nan=0.0, posinf=12.0, neginf=-12.0,
    )
    debris_p = torch.sigmoid(debris_logits)
    d = debris_p.squeeze(1) if debris_p.ndim == 4 else debris_p
    print(
        f"[diag]   GT: md_pos={md_pos_pct:.3f}% fg_pos={fg_pos_pct:.3f}% | "
        f"debris logit mean={float(debris_logits.mean()):.3f} "
        f"prob mean={float(d.mean()):.4f} max={float(d.max()):.4f}"
    )
    with torch.no_grad():
        bce = _binary_weighted_bce(debris_logits, md_target, crit._pos_weight)
        terms = {"debris_bce": float(bce.item()) * crit.debris_weight}
        if crit.seg_weight > 0:
            seg_l = crit.seg_loss(out["seg"], target)
            terms["seg"] = float(seg_l.item()) * crit.seg_weight
        type_labels = _type_labels_from_target(target, out["seg"].shape[1])
        if crit.type_weight > 0 and (type_labels != 255).any():
            type_l = F.cross_entropy(out["type"], type_labels, ignore_index=255)
            terms["type"] = float(type_l.item()) * crit.type_weight
        for w, aux_logits in zip(crit.deep_sup_weights, out.get("aux", [])):
            aux_l = crit.seg_loss(aux_logits, target)
            terms[f"aux@{w}"] = float(aux_l.item()) * w * crit.seg_weight
    for name, val in terms.items():
        print(f"[diag]   {name:12s} = {val:.4f}")
