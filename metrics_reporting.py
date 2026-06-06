"""Training/eval metric plots and tabular exports for thesis reporting."""

from __future__ import annotations

import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from experiment_timing import fmt_duration

# discussion.pdf conservative targets for thesis comparison
BENCHMARK_TARGETS = {
    "mIoU_foreground": (0.35, 0.42),
    "binary_debris_IoU": (0.25, 0.35),
    "plastic_IoU": (0.15, 0.25),
    "marida_md_IoU": (0.45, 0.60),
}


def save_metrics_json(metrics: dict, path: str) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    return path


def plot_training_dashboard(history_csv: str, save_dir: str, title: str) -> list[str]:
    """Multi-panel training curves: loss, mIoU, MD IoU, plastic, timing."""
    if not os.path.isfile(history_csv):
        return []
    df = pd.read_csv(history_csv)
    if df.empty:
        return []

    os.makedirs(save_dir, exist_ok=True)
    epochs = np.arange(1, len(df) + 1)
    paths = []

    metric_cols = [
        ("val_miou_fg", "Val mIoU (fg)", "#3CB44B"),
        ("val_md_iou", "Val MD IoU (1+2)", "#4363D8"),
        ("val_plastic_iou", "Val Plastic IoU", "#E6194B"),
        ("val_binary_debris_iou", "Val Binary Debris IoU", "#F58231"),
        ("val_plastic_recall", "Val Plastic Recall", "#911EB4"),
        ("val_checkpoint_score", "Checkpoint Score", "#42D4F4"),
        ("val_md_plastic_ckpt_score", "MD+Plastic Ckpt Score", "#BCBD22"),
    ]
    present = [(c, lbl, col) for c, lbl, col in metric_cols if c in df.columns and df[c].notna().any()]
    n_metric = max(1, len(present))
    ncols = min(3, n_metric)
    nrows = int(np.ceil(n_metric / ncols))

    fig, axes = plt.subplots(2 + nrows, 1, figsize=(11, 4 + 3 * nrows), facecolor="white")
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])

    ax = axes[0]
    ax.plot(epochs, df["train_loss"], label="Train", color="#2176AE", linewidth=1.6)
    if "val_loss" in df.columns:
        ax.plot(epochs, df["val_loss"], label="Val", color="#E84855", linewidth=1.6)
    ax.set_title(f"{title} — Loss", fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.35)

    ax_lr = axes[1]
    if "lr" in df.columns:
        ax_lr.plot(epochs, df["lr"], color="#F4A261", linewidth=1.6)
    ax_lr.set_title("Learning Rate", fontweight="bold")
    ax_lr.set_xlabel("Epoch")
    ax_lr.grid(True, linestyle="--", alpha=0.35)

    for i, (col, lbl, color) in enumerate(present):
        ax_m = axes[2 + i]
        vals = df[col].values
        ax_m.plot(epochs, vals, color=color, linewidth=1.6)
        if len(vals):
            best_i = int(np.nanargmax(vals))
            ax_m.scatter([epochs[best_i]], [vals[best_i]], color=color, s=35, zorder=5)
            ax_m.annotate(f"best={vals[best_i]:.4f} ep{epochs[best_i]}",
                          xy=(epochs[best_i], vals[best_i]), fontsize=8,
                          xytext=(5, 5), textcoords="offset points")
        ax_m.set_title(lbl, fontweight="bold")
        ax_m.set_xlabel("Epoch")
        ax_m.grid(True, linestyle="--", alpha=0.35)

    fig.suptitle(f"Training Dashboard — {title}", fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    p = os.path.join(save_dir, "dashboard_training.png")
    fig.savefig(p, dpi=300, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    paths.append(p)

    if "epoch_sec" in df.columns:
        fig_t, ax_t = plt.subplots(figsize=(10, 4), facecolor="white")
        ax_t.plot(epochs, df["epoch_sec"], label="Epoch total", color="#4363D8")
        if "train_sec" in df.columns:
            ax_t.plot(epochs, df["train_sec"], label="Train", color="#3CB44B", alpha=0.8)
        if "val_sec" in df.columns:
            ax_t.plot(epochs, df["val_sec"], label="Val", color="#E84855", alpha=0.8)
        ax_t.set_title(f"Per-Epoch Timing — {title}", fontweight="bold")
        ax_t.set_ylabel("Seconds")
        ax_t.set_xlabel("Epoch")
        ax_t.legend()
        ax_t.grid(True, linestyle="--", alpha=0.35)
        fig_t.tight_layout()
        p_t = os.path.join(save_dir, "dashboard_timing.png")
        fig_t.savefig(p_t, dpi=300, facecolor="white", bbox_inches="tight")
        plt.close(fig_t)
        paths.append(p_t)

        timing_rows = []
        for _, row in df.iterrows():
            timing_rows.append({
                "epoch": int(row.name) + 1 if isinstance(row.name, (int, np.integer)) else len(timing_rows) + 1,
                "epoch_sec": row.get("epoch_sec"),
                "epoch_fmt": fmt_duration(row.get("epoch_sec", 0)),
                "train_sec": row.get("train_sec"),
                "train_fmt": fmt_duration(row.get("train_sec", 0)),
                "val_sec": row.get("val_sec"),
                "val_fmt": fmt_duration(row.get("val_sec", 0)),
            })
        pd.DataFrame(timing_rows).to_csv(os.path.join(save_dir, "timing_per_epoch.csv"), index=False)

    df.to_csv(os.path.join(save_dir, "istoric_antrenare_full.csv"), index=False)
    return paths


def plot_benchmark_comparison(rows: list[dict], save_dir: str, filename: str = "benchmark_comparison.png") -> str | None:
    """Bar chart: model metrics vs discussion.pdf conservative targets."""
    if not rows:
        return None
    os.makedirs(save_dir, exist_ok=True)

    metrics = ["marida_md_IoU", "binary_debris_IoU", "plastic_IoU", "mIoU_foreground"]
    labels = ["MARIDA MD IoU", "Binary Debris IoU", "Plastic IoU", "mIoU fg"]
    models = [r.get("label", r.get("variant", "model")) for r in rows]

    x = np.arange(len(metrics))
    width = 0.8 / max(1, len(models))
    fig, ax = plt.subplots(figsize=(12, 5.5), facecolor="white")

    for mi, row in enumerate(rows):
        vals = [float(row.get(m, 0) or 0) for m in metrics]
        offset = (mi - (len(models) - 1) / 2) * width
        bars = ax.bar(x + offset, vals, width, label=models[mi])
        for bar, v in zip(bars, vals):
            if v > 0.005:
                ax.text(bar.get_x() + bar.get_width() / 2, v + 0.01, f"{v:.3f}",
                        ha="center", va="bottom", fontsize=7)

    for i, m in enumerate(metrics):
        if m in BENCHMARK_TARGETS:
            lo, hi = BENCHMARK_TARGETS[m]
            ax.hlines(lo, i - 0.4, i + 0.4, colors="#2a7e2a", linestyles="--", linewidth=1.2, alpha=0.7)
            ax.text(i + 0.42, lo, f"target {lo:.2f}", fontsize=7, color="#2a7e2a", va="center")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("IoU")
    ax.set_ylim(0, max(0.55, max(float(r.get(m, 0) or 0) for r in rows for m in metrics) * 1.25 + 0.05))
    ax.set_title("Model vs Thesis Targets (discussion.pdf conservative)", fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    fig.tight_layout()
    out = os.path.join(save_dir, filename)
    fig.savefig(out, dpi=300, facecolor="white", bbox_inches="tight")
    plt.close(fig)

    table = []
    for row in rows:
        entry = {k: row.get(k) for k in metrics}
        entry["label"] = row.get("label", row.get("variant"))
        entry["targets"] = {m: BENCHMARK_TARGETS.get(m) for m in metrics}
        table.append(entry)
    save_metrics_json(table, os.path.join(save_dir, "benchmark_comparison.json"))
    pd.DataFrame(table).to_csv(os.path.join(save_dir, "benchmark_comparison.csv"), index=False)
    return out


def plot_eval_metric_summary(debris_metrics: dict, per_class: dict, save_dir: str, model_name: str) -> list[str]:
    """Eval-time: thesis metrics bar + per-class IoU/F1 + precision-recall for plastic."""
    os.makedirs(save_dir, exist_ok=True)
    paths = []

    thesis_keys = [
        ("marida_md_IoU", "MARIDA MD IoU", "#4363D8"),
        ("binary_debris_IoU", "Binary Debris", "#3CB44B"),
        ("plastic_IoU", "Plastic", "#E6194B"),
        ("mIoU_foreground", "mIoU fg", "#F58231"),
    ]
    vals = [(k, lbl, debris_metrics.get(k, 0), col) for k, lbl, col in thesis_keys]
    fig, ax = plt.subplots(figsize=(9, 4.5), facecolor="white")
    bars = ax.bar([v[1] for v in vals], [v[2] for v in vals], color=[v[3] for v in vals])
    for bar, (_, _, v, _) in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.01, f"{v:.4f}", ha="center", fontsize=9)
    ax.set_ylim(0, max(0.5, max(v[2] for v in vals) * 1.3 + 0.05))
    ax.set_title(f"Thesis Metrics — {model_name}", fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    fig.tight_layout()
    p1 = os.path.join(save_dir, "grafic_thesis_metrics.png")
    fig.savefig(p1, dpi=300, facecolor="white")
    plt.close(fig)
    paths.append(p1)

    if per_class:
        fig2, ax2 = plt.subplots(figsize=(10, 5), facecolor="white")
        classes = sorted(per_class.keys())
        names = [f"Class {c}" for c in classes]
        ious = [per_class[c]["IoU"] for c in classes]
        recs = [per_class[c]["Recall"] for c in classes]
        precs = [per_class[c]["Precision"] for c in classes]
        x = np.arange(len(classes))
        w = 0.25
        ax2.bar(x - w, ious, w, label="IoU", color="#4363D8")
        ax2.bar(x, precs, w, label="Precision", color="#E6194B")
        ax2.bar(x + w, recs, w, label="Recall", color="#3CB44B")
        ax2.set_xticks(x)
        ax2.set_xticklabels(names)
        ax2.set_title(f"Per-Class IoU / P / R — {model_name}", fontweight="bold")
        ax2.legend()
        ax2.grid(axis="y", linestyle="--", alpha=0.35)
        fig2.tight_layout()
        p2 = os.path.join(save_dir, "grafic_per_class_pr_iou.png")
        fig2.savefig(p2, dpi=300, facecolor="white")
        plt.close(fig2)
        paths.append(p2)

    return paths
