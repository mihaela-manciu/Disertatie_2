import os
import time

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import MARIDADataset, get_validation_augmentation, load_channel_stats, NUM_CHANNELS
from experiment_timing import fmt_duration, format_timing_dict, print_timing_block, sec_since
from metrics_reporting import plot_eval_metric_summary, save_metrics_json
from models import build_model
from segmentation_utils import (
    CLASE_NUME,
    _eval_batch,
    calculate_debris_metrics,
    calculate_metrics,
    eval_config_from_tune_row,
    load_eval_config,
    save_eval_config,
    training_baseline_eval_config,
    tune_thresholds_md_plastic,
    tune_thresholds_on_val,
    update_confusion_matrix,
)


def plot_training_curves(history_csv_path, save_dir, model_name):
    if not os.path.isfile(history_csv_path):
        print(f"[training curves] CSV not found: {history_csv_path}")
        return None
    df = pd.read_csv(history_csv_path)
    if df.empty or "train_loss" not in df.columns:
        print(f"[training curves] CSV empty or missing columns: {history_csv_path}")
        return None
    epochs = range(1, len(df) + 1)
    os.makedirs(save_dir, exist_ok=True)
    has_miou = "val_miou_fg" in df.columns and df["val_miou_fg"].notna().any()
    has_lr = "lr" in df.columns and df["lr"].notna().any()
    n_plots = 1 + int(has_miou) + int(has_lr)
    fig, axes = plt.subplots(1, n_plots, figsize=(5.5 * n_plots, 4.2), facecolor="white")
    if n_plots == 1:
        axes = [axes]
    ax_idx = 0

    ax = axes[ax_idx]
    ax.plot(epochs, df["train_loss"], label="Train Loss", color="#2176AE", linewidth=1.6)
    if "val_loss" in df.columns:
        ax.plot(epochs, df["val_loss"], label="Val Loss", color="#E84855", linewidth=1.6)
    ax.set_xlabel("Epoch", fontsize=10)
    ax.set_ylabel("Loss", fontsize=10)
    ax.set_title("Training & Validation Loss", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9, frameon=True, fancybox=True, edgecolor="#cccccc")
    ax.grid(True, linestyle="--", alpha=0.35, color="#cccccc")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax_idx += 1

    if has_miou:
        ax = axes[ax_idx]
        miou_vals = df["val_miou_fg"].values
        ax.plot(epochs, miou_vals, color="#3CB44B", linewidth=1.6)
        best_epoch = int(np.argmax(miou_vals)) + 1
        best_val = float(np.max(miou_vals))
        ax.axhline(best_val, color="#3CB44B", linestyle=":", alpha=0.5, linewidth=1)
        ax.annotate(f"best = {best_val:.4f} (ep {best_epoch})",
                    xy=(best_epoch, best_val),
                    xytext=(best_epoch + len(df) * 0.05, best_val - 0.02),
                    fontsize=8.5, color="#2a7e2a",
                    arrowprops=dict(arrowstyle="->", color="#2a7e2a", lw=1))
        ax.set_xlabel("Epoch", fontsize=10)
        ax.set_ylabel("mIoU (foreground)", fontsize=10)
        ax.set_title("Validation mIoU (classes 1-3)", fontsize=11, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.35, color="#cccccc")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax_idx += 1

    if has_lr:
        ax = axes[ax_idx]
        ax.plot(epochs, df["lr"], color="#F4A261", linewidth=1.6)
        ax.set_xlabel("Epoch", fontsize=10)
        ax.set_ylabel("Learning Rate", fontsize=10)
        ax.set_title("LR Schedule (Warmup + Cosine)", fontsize=11, fontweight="bold")
        ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, -3))
        ax.grid(True, linestyle="--", alpha=0.35, color="#cccccc")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle(f"{model_name} — Training History", fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    out_path = os.path.join(save_dir, f"training_curves_{model_name.lower().replace(' ', '_').replace('(', '').replace(')', '')}.png")
    fig.savefig(out_path, dpi=300, facecolor="white", edgecolor="none", bbox_inches="tight")
    plt.close(fig)
    print(f"Training curves saved: {out_path}")
    return out_path


def plot_training_curves_overlay(history_paths, save_dir):
    fig, ax = plt.subplots(figsize=(7, 4.5), facecolor="white")
    colors = ["#2176AE", "#E84855", "#3CB44B", "#F4A261", "#911EB4", "#42D4F4"]
    any_plotted = False
    for i, (label, csv_path) in enumerate(history_paths):
        if not os.path.isfile(csv_path):
            continue
        df = pd.read_csv(csv_path)
        if "val_miou_fg" not in df.columns or df.empty:
            continue
        epochs = range(1, len(df) + 1)
        c = colors[i % len(colors)]
        ax.plot(epochs, df["val_miou_fg"], label=label, color=c, linewidth=1.6)
        best_val = df["val_miou_fg"].max()
        best_ep = df["val_miou_fg"].idxmax() + 1
        ax.scatter([best_ep], [best_val], color=c, s=40, zorder=5, edgecolors="white", linewidths=0.8)
        any_plotted = True
    if not any_plotted:
        plt.close(fig)
        return None
    ax.set_xlabel("Epoch", fontsize=10)
    ax.set_ylabel("mIoU (foreground)", fontsize=10)
    ax.set_title("Validation mIoU Comparison — All Models", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, frameon=True, fancybox=True, edgecolor="#cccccc")
    ax.grid(True, linestyle="--", alpha=0.35, color="#cccccc")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, "training_curves_overlay.png")
    fig.savefig(out_path, dpi=300, facecolor="white", edgecolor="none", bbox_inches="tight")
    plt.close(fig)
    print(f"Overlay training curves saved: {out_path}")
    return out_path


def plot_metrics_bars(metrics_dict, clase_nume, salvare_dir, model_name):
    clase = [clase_nume[c] for c in metrics_dict.keys()]
    ious = [metrics_dict[c]["IoU"] for c in metrics_dict.keys()]
    f1s = [metrics_dict[c]["F1-Score"] for c in metrics_dict.keys()]
    x = np.arange(len(clase))
    width = 0.32
    fig, ax = plt.subplots(figsize=(10, 5.5), facecolor="white")
    bars1 = ax.bar(x - width / 2, ious, width, label="IoU", color="#4363D8", edgecolor="white", linewidth=0.5)
    bars2 = ax.bar(x + width / 2, f1s, width, label="F1-Score", color="#E6194B", edgecolor="white", linewidth=0.5)
    for bar_group in (bars1, bars2):
        for bar in bar_group:
            h = bar.get_height()
            if h > 0.01:
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.015, f"{h:.2f}",
                        ha="center", va="bottom", fontsize=8, color="#444444")
    ax.set_ylabel("Score (0.0 – 1.0)", fontsize=11)
    ax.set_title(f"Per-Class Performance — {model_name}", fontsize=13, fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(clase, fontsize=10)
    ax.set_ylim(0, min(1.05, max(max(ious), max(f1s)) * 1.25 + 0.05))
    ax.legend(fontsize=10, frameon=True, fancybox=True, edgecolor="#cccccc")
    ax.grid(axis="y", linestyle="--", alpha=0.4, color="#cccccc")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_edgecolor("#cccccc")
    ax.spines["bottom"].set_edgecolor("#cccccc")
    fig.tight_layout()
    safe_name = model_name.lower().replace("-", "_").replace(" ", "_")
    fig.savefig(os.path.join(salvare_dir, f"grafic_metrici_{safe_name}.png"),
                dpi=300, facecolor="white", edgecolor="none")
    plt.close(fig)


def _sentinel2_rgb(image_tensor, channel_mean=None, channel_std=None, plow=2, phigh=98):
    rgb_indices = [3, 2, 1]
    channels = []
    for ci in rgb_indices:
        ch = image_tensor[ci].cpu().numpy().astype(np.float64)
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


def save_qualitative_panel(
    image_tensor, mask_numpy, pred_numpy, index, salvare_dir, model_name,
    channel_mean=None, channel_std=None,
):
    import matplotlib.colors as mcolors
    from matplotlib.patches import Rectangle
    from matplotlib.lines import Line2D
    from scipy import ndimage

    rgb_img = _sentinel2_rgb(image_tensor, channel_mean=channel_mean, channel_std=channel_std)
    seg_colors = ["#B0D4F1", "#E6194B", "#3CB44B", "#4363D8"]
    seg_cmap = mcolors.ListedColormap(seg_colors, name="seg4")
    seg_norm = mcolors.BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], seg_cmap.N)
    class_labels = ["Water", "Plastic", "Natural Debris", "Ships"]
    box_colors = {1: "#E6194B", 2: "#3CB44B", 3: "#4363D8"}
    box_size = 56

    boxes = []
    for cls_id in (1, 2, 3):
        presence = (mask_numpy == cls_id) | (pred_numpy == cls_id)
        if not presence.any():
            continue
        labeled, num = ndimage.label(presence)
        if num == 0:
            continue
        sizes = ndimage.sum(presence, labeled, range(1, num + 1))
        best_label = int(np.argmax(sizes)) + 1
        ys, xs = np.where(labeled == best_label)
        cy, cx = int(ys.mean()), int(xs.mean())
        h, w = mask_numpy.shape
        y0 = max(0, min(h - box_size, cy - box_size // 2))
        x0 = max(0, min(w - box_size, cx - box_size // 2))
        boxes.append((y0, x0, box_size, cls_id))

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5), facecolor="white")
    fig.subplots_adjust(wspace=0.12)
    axes[0].imshow(rgb_img)
    axes[0].set_title("Sentinel-2 RGB", fontsize=12, fontweight="bold", pad=8)
    axes[1].imshow(mask_numpy, cmap=seg_cmap, norm=seg_norm, interpolation="nearest")
    axes[1].set_title("Ground Truth", fontsize=12, fontweight="bold", pad=8)
    axes[2].imshow(pred_numpy, cmap=seg_cmap, norm=seg_norm, interpolation="nearest")
    axes[2].set_title(f"Prediction — {model_name}", fontsize=12, fontweight="bold", pad=8)

    for bi, (y0, x0, b, cls_id) in enumerate(boxes):
        color = box_colors[cls_id]
        for ax in axes:
            ax.add_patch(Rectangle((x0, y0), b, b, fill=False, linewidth=2.0,
                                   edgecolor=color, linestyle="-"))
        axes[0].text(x0 + 2, y0 - 3, f"{class_labels[cls_id]}",
                     color=color, fontsize=7, fontweight="bold",
                     bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                               alpha=0.85, edgecolor=color, linewidth=0.5))

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor("#cccccc")
            spine.set_linewidth(0.5)

    legend_handles = [
        Line2D([0], [0], marker="s", color="none", markerfacecolor=seg_colors[i],
               markeredgecolor="#666666", markersize=10, label=class_labels[i])
        for i in range(4)
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=4, fontsize=10,
               frameon=True, fancybox=True, edgecolor="#cccccc", framealpha=0.95, borderpad=0.5)
    fig.suptitle(f"Qualitative Analysis — {model_name} (Patch {index})",
                 fontsize=14, fontweight="bold", y=1.01, color="#222222")
    os.makedirs(os.path.join(salvare_dir, "visual_panels"), exist_ok=True)
    safe_name = model_name.lower().replace("-", "_").replace(" ", "_")
    fig.savefig(os.path.join(salvare_dir, "visual_panels", f"panou_{safe_name}_{index}.png"),
                dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close(fig)


def _predict_batch(model, images, device, eval_config):
    two_head = getattr(model, "two_head", False)
    return _eval_batch(model, images, None, device, eval_config, two_head)


def evaluate_pipeline(
    cale_dataset, model_path, model_key="taunet_resnet50", model_name=None,
    results_dir="results", eval_config_path=None, tune_on_val=True,
    two_head=None, model_class=None, num_workers=0, fast_tune=True,
    tune_mode="miou", default_eval_config=None,
    tune_md_weight=0.5, tune_plastic_weight=0.5,
    tune_plastic_precision_weight=0.0,
    tune_prefer_threshold=None, tune_threshold_score_tol=0.01,
):
    model_key = model_key.lower()
    if two_head is None:
        two_head = model_key in ("taunet", "taunet_resnet50")
    if model_name is None:
        model_name = model_key.replace("_", " ").title()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(results_dir, exist_ok=True)
    t_run = time.perf_counter()
    timing = {}

    if eval_config_path is None:
        eval_config_path = os.path.join(results_dir, "eval_config.json")

    t_step = time.perf_counter()
    try:
        state = torch.load(model_path, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(model_path, map_location=device)
    timing["load_checkpoint_sec"] = sec_since(t_step)

    has_two_head = any(k.startswith("debris_head") or k.startswith("type_head") for k in state.keys())
    has_deep_sup = any(k.startswith("aux_head") for k in state.keys())
    has_ssag = any(k.startswith("ssag") for k in state.keys())

    if model_key in ("taunet", "taunet_resnet50"):
        if two_head is not None and two_head != has_two_head:
            print(f"[eval] two_head={two_head} cerut, dar checkpoint-ul are "
                  f"two_head={has_two_head}; folosesc valoarea din checkpoint.")
        two_head = has_two_head

    t_step = time.perf_counter()
    model = build_model(model_key, in_channels=NUM_CHANNELS, two_head=two_head,
                        deep_supervision=has_deep_sup, use_ssag=has_ssag).to(device)
    model.load_state_dict(state)
    model.eval()
    timing["build_model_sec"] = sec_since(t_step)

    t_step = time.perf_counter()
    val_dataset = MARIDADataset(cale_dataset, split="val", transform=get_validation_augmentation())
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=num_workers)

    timing["setup_val_loader_sec"] = sec_since(t_step)

    eval_config = load_eval_config(eval_config_path)
    t_step = time.perf_counter()
    if eval_config is None and tune_on_val:
        print(
            f"Calibrare hiperparametri pe setul de validare "
            f"(mode={tune_mode}, fast={fast_tune})..."
        )
        if tune_mode == "md_plastic" and two_head:
            best_row, _ = tune_thresholds_md_plastic(
                model, val_loader, device,
                use_two_head=True,
                fast=fast_tune,
                md_weight=tune_md_weight,
                plastic_weight=tune_plastic_weight,
                plastic_precision_weight=tune_plastic_precision_weight,
                preferred_threshold=tune_prefer_threshold,
                preferred_threshold_tol=tune_threshold_score_tol,
            )
            eval_config = eval_config_from_tune_row(best_row)
            print(
                f"[tune] best val MD IoU={best_row.get('val_md_iou', 0):.3f} "
                f"plastic IoU={best_row.get('val_plastic_iou', 0):.3f} "
                f"thr={eval_config.get('debris_threshold')}"
            )
        else:
            eval_config = tune_thresholds_on_val(
                model, val_loader, device,
                use_two_head=getattr(model, "two_head", False),
                fast=fast_tune,
            )
        save_eval_config(eval_config, eval_config_path)
        print(f"Config salvat: {json.dumps(eval_config, indent=2)}")
        timing["tune_on_val_sec"] = sec_since(t_step)
    elif eval_config is None:
        if default_eval_config is not None:
            eval_config = dict(default_eval_config)
        elif two_head:
            eval_config = training_baseline_eval_config(two_head=True)
        else:
            eval_config = training_baseline_eval_config(two_head=False)
        timing["tune_on_val_sec"] = 0.0
    else:
        timing["tune_on_val_sec"] = 0.0

    t_step = time.perf_counter()
    test_dataset = MARIDADataset(cale_dataset, split="test", transform=get_validation_augmentation())
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=num_workers)

    ch_stats = load_channel_stats(cale_dataset)
    ch_mean = np.array(ch_stats["mean"], dtype=np.float64)
    ch_std = np.array(ch_stats["std"], dtype=np.float64)

    timing["setup_test_loader_sec"] = sec_since(t_step)

    conf_matrix = np.zeros((4, 4), dtype=np.int64)
    panouri_salvate = 0

    t_step = time.perf_counter()
    for idx, (images, masks) in enumerate(tqdm(test_loader, desc="Evaluating")):
        images = images.to(device)
        mask_raw = masks.numpy().squeeze()
        preds = _predict_batch(model, images, device, eval_config)
        update_confusion_matrix(conf_matrix, mask_raw, preds)
        if (1 in mask_raw or 2 in mask_raw) and panouri_salvate < 5:
            save_qualitative_panel(
                images[0], mask_raw, preds, idx, results_dir, model_name,
                channel_mean=ch_mean, channel_std=ch_std,
            )
            panouri_salvate += 1
    timing["test_inference_sec"] = sec_since(t_step)

    t_step = time.perf_counter()
    toate_metricile = calculate_metrics(conf_matrix)
    debris_extra = calculate_debris_metrics(conf_matrix)

    print("\n" + "=" * 60)
    print(f" REZULTATE TEST - {model_name}")
    print(f" Config: {eval_config}")
    print("=" * 60)
    print(f"{'Clasa':<32} | {'IoU':<6} | {'Precision':<6} | {'Recall':<6} | {'F1':<6}")
    print("-" * 60)

    rows_list = []
    for c, metrici in toate_metricile.items():
        print(f"{CLASE_NUME[c]:<32} | {metrici['IoU']:.4f} | {metrici['Precision']:.4f} | "
              f"{metrici['Recall']:.4f} | {metrici['F1-Score']:.4f}")
        rows_list.append({
            "Clasa": CLASE_NUME[c],
            "IoU": metrici["IoU"],
            "Precision": metrici["Precision"],
            "Recall": metrici["Recall"],
            "F1-Score": metrici["F1-Score"],
        })

    print("-" * 60)
    print(f"mIoU foreground (1-3): {debris_extra['mIoU_foreground']:.4f}")
    print(f"MARIDA MD IoU (1+2): {debris_extra.get('marida_md_IoU', debris_extra['binary_debris_IoU']):.4f}")
    print(f"Binary debris IoU (1+2): {debris_extra['binary_debris_IoU']:.4f}")
    print(f"Plastic recall: {debris_extra.get('plastic_recall', 0.0):.4f}")
    print("=" * 60)

    rows_list.append({"Clasa": "mIoU foreground (1-3)", "IoU": debris_extra["mIoU_foreground"],
                      "Precision": np.nan, "Recall": np.nan, "F1-Score": np.nan})
    rows_list.append({"Clasa": "MARIDA MD IoU (1+2)", "IoU": debris_extra.get(
        "marida_md_IoU", debris_extra["binary_debris_IoU"]),
                      "Precision": np.nan, "Recall": np.nan, "F1-Score": np.nan})
    rows_list.append({"Clasa": "Binary debris IoU (1+2)", "IoU": debris_extra["binary_debris_IoU"],
                      "Precision": np.nan, "Recall": np.nan, "F1-Score": np.nan})
    rows_list.append({"Clasa": "Plastic recall", "IoU": np.nan,
                      "Precision": np.nan, "Recall": debris_extra.get("plastic_recall", 0.0),
                      "F1-Score": np.nan})

    safe_name = model_key.lower().replace("-", "_")
    df_raport = pd.DataFrame(rows_list)
    df_raport.to_csv(os.path.join(results_dir, f"raport_metrici_{safe_name}.csv"), index=False)
    plot_metrics_bars(toate_metricile, CLASE_NUME, results_dir, model_name)
    try:
        thesis_paths = plot_eval_metric_summary(debris_extra, toate_metricile, results_dir, model_name)
        if thesis_paths:
            print(f"Thesis metric plots: {', '.join(os.path.basename(p) for p in thesis_paths)}")
    except Exception as exc:
        print(f"[warning] plot_eval_metric_summary failed: {exc}")

    save_metrics_json(
        {"debris_metrics": debris_extra, "per_class": toate_metricile, "eval_config": eval_config},
        os.path.join(results_dir, "eval_metrics_full.json"),
    )

    fig_cm, ax_cm = plt.subplots(figsize=(8, 7), facecolor="white")
    cm_normalized = conf_matrix.astype("float") / (conf_matrix.sum(axis=1)[:, np.newaxis] + 1e-8)
    sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=CLASE_NUME, yticklabels=CLASE_NUME,
                linewidths=0.5, linecolor="white",
                cbar_kws={"shrink": 0.8, "label": "Proportion"}, ax=ax_cm)
    ax_cm.set_title(f"Confusion Matrix — {model_name}", fontsize=13, fontweight="bold", pad=12)
    ax_cm.set_ylabel("True Class", fontsize=11)
    ax_cm.set_xlabel("Predicted Class", fontsize=11)
    ax_cm.tick_params(axis="both", labelsize=10)
    fig_cm.tight_layout()
    fig_cm.savefig(os.path.join(results_dir, f"matrice_confuzie_{safe_name}.png"),
                   dpi=300, facecolor="white", edgecolor="none")
    plt.close(fig_cm)
    timing["save_reports_sec"] = sec_since(t_step)
    timing["total_sec"] = sec_since(t_run)
    timing = format_timing_dict(timing)

    print_timing_block(f"Evaluare {model_name}", timing)

    timing_path = os.path.join(results_dir, "eval_timing.json")
    with open(timing_path, "w", encoding="utf-8") as f:
        json.dump(timing, f, indent=2)

    return {
        "model_key": model_key,
        "model_name": model_name,
        "metrics_per_class": toate_metricile,
        "debris_metrics": debris_extra,
        "eval_config": eval_config,
        "results_dir": results_dir,
        "report_csv": os.path.join(results_dir, f"raport_metrici_{safe_name}.csv"),
        "duration_sec": timing["total_sec"],
        "duration_fmt": timing.get("total_fmt", fmt_duration(timing["total_sec"])),
        "timing": timing,
    }


if __name__ == "__main__":
    cale_marida = r"D:\TAID\Disertatie\MARIDA"
    model_key = "taunet_resnet50"
    evaluate_pipeline(
        cale_marida,
        os.path.join("saved_models", f"{model_key}_best.pth"),
        model_key=model_key,
        model_name="TAUNet-ResNet50",
        results_dir=os.path.join("results", model_key),
        tune_on_val=True,
    )
