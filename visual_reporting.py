import os

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi": 150,
})

FOREGROUND_CLASSES = {1: "Plastic", 2: "Natural Debris", 3: "Ships"}
CLASS_COLORS = {1: "#E6194B", 2: "#3CB44B", 3: "#4363D8"}
_SEG_COLORS = ["#B0D4F1", "#E6194B", "#3CB44B", "#4363D8"]
_SEG_CMAP = mcolors.ListedColormap(_SEG_COLORS, name="seg4")
_SEG_NORM = mcolors.BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], _SEG_CMAP.N)
_CLASS_LABELS = ["Water", "Plastic", "Natural Debris", "Ships"]


def tensor_to_rgb(image_tensor, channel_mean=None, channel_std=None, plow=2, phigh=98):
    rgb_indices = [3, 2, 1]
    channels = []
    for ci in rgb_indices:
        ch = image_tensor[ci].detach().cpu().numpy().astype(np.float64)
        if channel_mean is not None and channel_std is not None:
            ch = ch * channel_std[ci] + channel_mean[ci]
        channels.append(ch)
    rgb = np.stack(channels, axis=-1)
    for i in range(3):
        lo = np.percentile(rgb[:, :, i], plow)
        hi = np.percentile(rgb[:, :, i], phigh)
        if hi - lo < 1e-6:
            hi = lo + 1e-6
        rgb[:, :, i] = (rgb[:, :, i] - lo) / (hi - lo)
    return np.clip(rgb, 0.0, 1.0)


def _largest_region_center(mask_2d):
    from scipy import ndimage
    labeled, num = ndimage.label(mask_2d)
    if num == 0:
        ys, xs = np.where(mask_2d)
        return int(ys.mean()), int(xs.mean())
    sizes = ndimage.sum(mask_2d, labeled, range(1, num + 1))
    best_label = int(np.argmax(sizes)) + 1
    ys, xs = np.where(labeled == best_label)
    return int(ys.mean()), int(xs.mean())


def find_interest_boxes(gt, pred_a, pred_b, box=56, max_per_class=2):
    h, w = gt.shape
    boxes = []
    for cls_id in FOREGROUND_CLASSES:
        presence = (gt == cls_id) | (pred_a == cls_id) | (pred_b == cls_id)
        if not presence.any():
            continue
        from scipy import ndimage
        labeled, num = ndimage.label(presence)
        if num == 0:
            continue
        sizes = ndimage.sum(presence, labeled, range(1, num + 1))
        top_labels = np.argsort(sizes)[::-1][:max_per_class] + 1
        for lbl in top_labels:
            comp = labeled == lbl
            ys, xs = np.where(comp)
            cy, cx = int(ys.mean()), int(xs.mean())
            y0 = max(0, min(h - box, cy - box // 2))
            x0 = max(0, min(w - box, cx - box // 2))
            boxes.append((y0, x0, box, cls_id))
    return boxes


def _render_mask(ax, mask, title):
    ax.imshow(mask, cmap=_SEG_CMAP, norm=_SEG_NORM, interpolation="nearest")
    ax.set_title(title, fontweight="medium", pad=6)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def save_zoom_comparison(image_tensor, gt, pred_no_aug, pred_aug, out_path, *,
                         boxes=None, model_name="Model",
                         channel_mean=None, channel_std=None):
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    rgb = tensor_to_rgb(image_tensor, channel_mean=channel_mean, channel_std=channel_std)
    if boxes is None:
        boxes = find_interest_boxes(gt, pred_no_aug, pred_aug)
    if not boxes:
        return

    n = len(boxes)
    fig_h = 3.2 + n * 2.8
    fig, axes = plt.subplots(n + 1, 4, figsize=(14, fig_h), facecolor="white",
                             gridspec_kw={"height_ratios": [1.3] + [1.0] * n})
    if n + 1 == 1:
        axes = axes[np.newaxis, :]

    for col in range(4):
        ax = axes[0, col]
        if col == 0:
            ax.imshow(rgb)
            ax.set_title("Sentinel-2 RGB", fontweight="bold", pad=6)
        elif col == 1:
            _render_mask(ax, gt, "Ground Truth")
        elif col == 2:
            _render_mask(ax, pred_no_aug, "Pred — No Augmentation")
        elif col == 3:
            _render_mask(ax, pred_aug, "Pred — With Augmentation")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor("#cccccc")
            spine.set_linewidth(0.5)

    ax_rgb = axes[0, 0]
    for bi, (y0, x0, b, cls_id) in enumerate(boxes):
        color = CLASS_COLORS[cls_id]
        ax_rgb.add_patch(Rectangle((x0, y0), b, b, fill=False, linewidth=2.0,
                                   edgecolor=color, linestyle="-"))
        ax_rgb.text(x0 + 2, y0 - 3, f"{bi + 1}", color=color, fontsize=8, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.15", facecolor="white", alpha=0.85,
                              edgecolor=color, linewidth=0.5))
    for col_idx in (1, 2, 3):
        for bi, (y0, x0, b, cls_id) in enumerate(boxes):
            color = CLASS_COLORS[cls_id]
            axes[0, col_idx].add_patch(
                Rectangle((x0, y0), b, b, fill=False, linewidth=1.5, edgecolor=color, linestyle="--"))

    for bi, (y0, x0, b, cls_id) in enumerate(boxes):
        color = CLASS_COLORS[cls_id]
        cls_name = FOREGROUND_CLASSES[cls_id]
        row = bi + 1
        crop_rgb = rgb[y0: y0 + b, x0: x0 + b]
        crop_gt = gt[y0: y0 + b, x0: x0 + b]
        crop_na = pred_no_aug[y0: y0 + b, x0: x0 + b]
        crop_aug = pred_aug[y0: y0 + b, x0: x0 + b]
        titles = [f"#{bi+1} Zoom — {cls_name}", "GT (crop)",
                  "Pred No Aug (crop)", "Pred Aug (crop)"]
        data = [crop_rgb, crop_gt, crop_na, crop_aug]
        for col, (d, title) in enumerate(zip(data, titles)):
            ax = axes[row, col]
            if col == 0:
                ax.imshow(d)
            else:
                ax.imshow(d, cmap=_SEG_CMAP, norm=_SEG_NORM, interpolation="nearest")
            ax.set_title(title, fontweight="medium", fontsize=10, pad=6)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_edgecolor(color)
                spine.set_linewidth(1.5)

    legend_elements = []
    for cls_id, cls_name in FOREGROUND_CLASSES.items():
        if any(b_[3] == cls_id for b_ in boxes):
            legend_elements.append(
                Line2D([0], [0], color=CLASS_COLORS[cls_id], linewidth=3, label=f"Box: {cls_name}"))
    for i, label in enumerate(_CLASS_LABELS):
        legend_elements.append(
            Line2D([0], [0], marker="s", color="none", markerfacecolor=_SEG_COLORS[i],
                   markeredgecolor="#666666", markersize=9, label=label))
    fig.legend(handles=legend_elements, loc="lower center",
               ncol=min(len(legend_elements), 7), fontsize=9,
               frameon=True, fancybox=True, edgecolor="#cccccc", framealpha=0.95, borderpad=0.6)
    fig.suptitle(f"{model_name} — No Augmentation vs. Augmentation",
                 fontsize=14, fontweight="bold", y=0.995, color="#222222")
    fig.subplots_adjust(hspace=0.35, wspace=0.08, bottom=0.06, top=0.95)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close(fig)
