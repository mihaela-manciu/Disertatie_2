"""Shared decoding, metrics, TTA, and post-processing for MARIDA segmentation."""

import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage
from tqdm import tqdm


CLASE_NUME = [
    "0: Background",
    "1: Artificial Debris (Plastic)",
    "2: Natural Debris",
    "3: Ships (Nave)",
]


def model_forward(model, images):
    out = model(images)
    if isinstance(out, dict):
        return out
    return {"seg": out}


def _tta_geometric_pairs():
    return [
        (lambda x: x, lambda x: x),
        (lambda x: torch.rot90(x, k=1, dims=(2, 3)), lambda x: torch.rot90(x, k=-1, dims=(2, 3))),
        (lambda x: torch.flip(x, dims=[3]), lambda x: torch.flip(x, dims=[3])),
        (lambda x: torch.flip(x, dims=[2]), lambda x: torch.flip(x, dims=[2])),
    ]


def _scale_image(image, scale, size_hw):
    if abs(scale - 1.0) < 1e-6:
        return image
    h, w = size_hw
    new_h, new_w = int(h * scale), int(w * scale)
    scaled = F.interpolate(image, size=(new_h, new_w), mode="bilinear", align_corners=False)
    return F.interpolate(scaled, size=(h, w), mode="bilinear", align_corners=False)


def _rgb_from_tensor(image_tensor):
    img = image_tensor.detach().cpu()
    if img.ndim == 4:
        img = img[0]
    r = (img[3].numpy() * 255).astype(np.uint8)
    g = (img[2].numpy() * 255).astype(np.uint8)
    b = (img[1].numpy() * 255).astype(np.uint8)
    return np.stack([r, g, b], axis=-1)


def apply_crf_refinement(seg_probs, image_tensor, use_crf=True):
    if not use_crf:
        return seg_probs
    batched = seg_probs.ndim == 4
    probs_np = seg_probs[0].detach().cpu().numpy() if batched else seg_probs.detach().cpu().numpy()
    c, h, w = probs_np.shape
    unary = np.clip(probs_np, 1e-6, 1.0)
    rgb = _rgb_from_tensor(image_tensor)

    try:
        import pydensecrf.densecrf as dcrf
        from pydensecrf.utils import unary_from_softmax
        u = unary_from_softmax(unary)
        d = dcrf.DenseCRF2D(w, h, c)
        d.setUnaryEnergy(u)
        d.addPairwiseGaussian(sxy=3, compat=3)
        d.addPairwiseBilateral(sxy=30, srgb=10, rgbim=rgb, compat=5)
        q = d.inference(5)
        refined = np.array(q).reshape(c, h, w)
        out = torch.from_numpy(refined).float().to(seg_probs.device)
        return out.unsqueeze(0) if batched else out
    except ImportError:
        pass

    try:
        import cv2
        refined = np.zeros_like(unary)
        for ch in range(c):
            ch_u8 = (unary[ch] * 255).astype(np.uint8)
            refined[ch] = cv2.bilateralFilter(ch_u8, d=5, sigmaColor=50, sigmaSpace=50) / 255.0
        out = torch.from_numpy(refined).float().to(seg_probs.device)
        return out.unsqueeze(0) if batched else out
    except ImportError:
        return seg_probs


def predict_with_tta_softmax(model, image, device, scales=(1.0,), use_crf=False):
    model.eval()
    size_hw = image.shape[2:]
    def _probs(img):
        out = model_forward(model, img)
        return torch.softmax(out["seg"], dim=1)
    with torch.no_grad():
        accum = []
        for scale in scales:
            img_s = _scale_image(image, scale, size_hw)
            for aug, inv in _tta_geometric_pairs():
                seg = inv(_probs(aug(img_s)))
                accum.append(seg)
        mean_probs = torch.stack(accum, dim=0).mean(dim=0)
        return apply_crf_refinement(mean_probs, image, use_crf=use_crf)


def predict_with_tta_two_head(model, image, device, scales=(1.0,), use_crf=False):
    model.eval()
    size_hw = image.shape[2:]
    def _forward(img):
        out = model_forward(model, img)
        return (
            torch.softmax(out["seg"], dim=1),
            torch.sigmoid(out["debris"]),
            torch.softmax(out["type"], dim=1),
        )
    with torch.no_grad():
        seg_list, debris_list, type_list = [], [], []
        for scale in scales:
            img_s = _scale_image(image, scale, size_hw)
            for aug, inv in _tta_geometric_pairs():
                seg, debris, typ = _forward(aug(img_s))
                seg_list.append(inv(seg))
                debris_list.append(inv(debris))
                type_list.append(inv(typ))
        seg_probs = torch.stack(seg_list, dim=0).mean(dim=0)
        seg_probs = apply_crf_refinement(seg_probs, image, use_crf=use_crf)
        debris_probs = torch.stack(debris_list, dim=0).mean(dim=0)
        type_probs = torch.stack(type_list, dim=0).mean(dim=0)
        return seg_probs, debris_probs, type_probs


def decode_softmax(seg_probs):
    if seg_probs.ndim == 4:
        return torch.argmax(seg_probs, dim=1).cpu().numpy()
    return torch.argmax(seg_probs, dim=0).cpu().numpy()


def decode_thresholds(probs, thresholds):
    h, w = probs.shape[1], probs.shape[2]
    preds = np.zeros((h, w), dtype=np.int64)
    fg = probs[1:4]
    fg_mask = fg >= np.array(thresholds)[:, None, None]
    has_fg = fg_mask.any(axis=0)
    preds[has_fg] = np.argmax(fg[:, has_fg], axis=0) + 1
    return preds


def decode_two_head(seg_probs, debris_probs, type_probs, debris_threshold=0.5):
    seg_pred = torch.argmax(seg_probs, dim=1)
    debris_mask = (debris_probs.squeeze(1) >= debris_threshold)
    type_pred = torch.argmax(type_probs, dim=1) + 1
    refined = seg_pred.clone()
    refine_mask = debris_mask & (seg_pred == 0)
    refined[refine_mask] = type_pred[refine_mask]
    return refined.cpu().numpy()


def postprocess_predictions(preds, min_component_size=8, classes=(1, 2)):
    cleaned = preds.copy()
    for cls in classes:
        mask = cleaned == cls
        if not mask.any():
            continue
        labeled, num = ndimage.label(mask)
        for label_id in range(1, num + 1):
            component = labeled == label_id
            if component.sum() < min_component_size:
                cleaned[component] = 0
    return cleaned


def update_confusion_matrix(conf_matrix, target, pred, num_classes=4):
    t = np.asarray(target)
    p = np.asarray(pred)
    if t.ndim == 3 and t.shape[0] == 1:
        t = t[0]
    if p.ndim == 3 and p.shape[0] == 1:
        p = p[0]
    if t.shape != p.shape:
        raise ValueError(f"update_confusion_matrix: target shape {t.shape} != pred shape {p.shape}")
    t_flat = t.flatten()
    p_flat = p.flatten()
    valid = (t_flat < num_classes) & (p_flat < num_classes)
    t_valid = t_flat[valid]
    p_valid = p_flat[valid]
    np.add.at(conf_matrix, (t_valid, p_valid), 1)


def calculate_metrics(conf_matrix, num_classes=4):
    metrics = {}
    for c in range(num_classes):
        tp = conf_matrix[c, c]
        fp = conf_matrix[:, c].sum() - tp
        fn = conf_matrix[c, :].sum() - tp
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
        iou = tp / (tp + fp + fn + 1e-8)
        metrics[c] = {
            "Precision": float(precision),
            "Recall": float(recall),
            "F1-Score": float(f1),
            "IoU": float(iou),
        }
    return metrics


def calculate_debris_metrics(conf_matrix):
    fg_metrics = calculate_metrics(conf_matrix)
    miou_fg = np.mean([fg_metrics[c]["IoU"] for c in (1, 2, 3)])
    debris_tp = conf_matrix[1, 1] + conf_matrix[1, 2] + conf_matrix[2, 1] + conf_matrix[2, 2]
    debris_actual = conf_matrix[1, :].sum() + conf_matrix[2, :].sum()
    debris_pred = conf_matrix[:, 1].sum() + conf_matrix[:, 2].sum()
    debris_fp = debris_pred - debris_tp
    debris_fn = debris_actual - debris_tp
    binary_debris_iou = debris_tp / (debris_tp + debris_fp + debris_fn + 1e-8)
    return {
        "mIoU_foreground": float(miou_fg),
        "binary_debris_IoU": float(binary_debris_iou),
    }


def _decode_batch(model, images, device, two_head=False, debris_threshold=0.5):
    out = model_forward(model, images)
    if two_head:
        seg_p = torch.softmax(out["seg"], dim=1)
        debris_p = torch.sigmoid(out["debris"])
        type_p = torch.softmax(out["type"], dim=1)
        pred = decode_two_head(seg_p, debris_p, type_p, debris_threshold)
    else:
        pred = decode_softmax(torch.softmax(out["seg"], dim=1))
    if pred.ndim == 2:
        pred = pred[None, ...]
    return pred


def compute_val_miou_foreground(model, dataloader, device, two_head=False,
                                 debris_threshold=0.5, use_tta=False,
                                 postprocess=False):
    model.eval()
    conf_matrix = np.zeros((4, 4), dtype=np.int64)
    with torch.no_grad():
        for images, masks in dataloader:
            images = images.to(device)
            mask_np = masks.numpy()
            if mask_np.ndim == 2:
                mask_np = mask_np[None, ...]
            if use_tta:
                if two_head:
                    seg_p, debris_p, type_p = predict_with_tta_two_head(model, images, device)
                    pred = decode_two_head(seg_p, debris_p, type_p, debris_threshold)
                else:
                    seg_p = predict_with_tta_softmax(model, images, device)
                    pred = decode_softmax(seg_p)
                if pred.ndim == 2:
                    pred = pred[None, ...]
            else:
                pred = _decode_batch(model, images, device, two_head, debris_threshold)
            for sample_idx in range(pred.shape[0]):
                p = pred[sample_idx]
                if postprocess:
                    p = postprocess_predictions(p)
                update_confusion_matrix(conf_matrix, mask_np[sample_idx], p)
    return calculate_debris_metrics(conf_matrix)["mIoU_foreground"]


def _eval_batch(model, images, mask_np, device, cfg, use_two_head):
    scales = tuple(cfg.get("tta_scales", (1.0,)))
    use_crf = cfg.get("use_crf", False)
    min_size = cfg.get("min_component_size", 8)
    if use_two_head:
        seg_p, debris_p, type_p = predict_with_tta_two_head(
            model, images, device, scales=scales, use_crf=use_crf
        )
        pred = decode_two_head(seg_p, debris_p, type_p, cfg.get("debris_threshold", 0.5))[0]
    else:
        seg_p = predict_with_tta_softmax(model, images, device, scales=scales, use_crf=use_crf)
        pred = decode_softmax(seg_p)
        if pred.ndim == 3:
            pred = pred[0]
    return postprocess_predictions(pred, min_component_size=min_size)


def _run_val_config(model, val_loader, device, cfg, use_two_head, progress_desc=None):
    conf_matrix = np.zeros((4, 4), dtype=np.int64)
    iterator = val_loader
    if progress_desc is not None:
        iterator = tqdm(val_loader, desc=progress_desc, leave=False)
    for images, masks in iterator:
        images = images.to(device)
        mask_np = masks.numpy().squeeze()
        pred = _eval_batch(model, images, mask_np, device, cfg, use_two_head)
        update_confusion_matrix(conf_matrix, mask_np, pred)
    return calculate_debris_metrics(conf_matrix)["mIoU_foreground"]


def tune_thresholds_on_val(model, val_loader, device, use_two_head=False,
                           debris_threshold_grid=None, scale_options=None,
                           crf_options=None, fast=True):
    if fast:
        if scale_options is None:
            scale_options = [(1.0,)]
        if crf_options is None:
            crf_options = [False]
        if debris_threshold_grid is None:
            debris_threshold_grid = [0.40, 0.50, 0.60]
        min_size_options = (4, 8, 16)
    else:
        if scale_options is None:
            scale_options = [(1.0,), (0.75, 1.0, 1.25)]
        if crf_options is None:
            crf_options = [False, True]
        if debris_threshold_grid is None:
            debris_threshold_grid = [0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
        min_size_options = (4, 8, 12, 16)

    model.eval()
    best_miou = -1.0
    best_cfg = {}
    configs = []
    for scales in scale_options:
        for use_crf in crf_options:
            for min_size in min_size_options:
                if use_two_head:
                    for debris_thr in debris_threshold_grid:
                        configs.append({
                            "decode_mode": "two_head",
                            "debris_threshold": debris_thr,
                            "tta_scales": list(scales),
                            "use_crf": use_crf,
                            "min_component_size": min_size,
                        })
                else:
                    configs.append({
                        "decode_mode": "softmax",
                        "tta_scales": list(scales),
                        "use_crf": use_crf,
                        "min_component_size": min_size,
                    })

    print(f"Tune: {len(configs)} configurații de încercat ({'fast' if fast else 'full'})")
    with torch.no_grad():
        for i, cfg in enumerate(tqdm(configs, desc="Tune val", leave=False)):
            miou_fg = _run_val_config(
                model, val_loader, device, cfg, use_two_head,
                progress_desc=f"cfg {i + 1}/{len(configs)}",
            )
            if miou_fg > best_miou:
                best_miou = miou_fg
                best_cfg = {**cfg, "val_mIoU_fg": miou_fg}
    return best_cfg


def save_eval_config(config, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)


def load_eval_config(path):
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
