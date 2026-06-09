"""
Automated MARIDA segmentation experiments — Disertatie_2 (14ch + pretrained backbone).

Default: taunet_resnet50 with md_outlier recipe (outlier-first binary MD + type refinement).
Thesis targets (discussion.pdf): MD IoU ~0.35–0.55, plastic IoU ~0.15–0.25, mIoU fg ~0.35–0.42.

Usage:
  python run_experiments.py --marida "D:\\TAID\\Disertatie\\MARIDA"
  python run_experiments.py --models taunet_resnet50 --recipe md_outlier --no-resume
  python run_experiments.py --recipe pretrained_strong  # legacy single-head ablation
  python run_experiments.py --skip-train --skip-eval  # only build comparison
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from evaluate import evaluate_pipeline, plot_training_curves, plot_training_curves_overlay
from metrics_reporting import plot_benchmark_comparison, plot_training_dashboard
from dataset import MARIDADataset, get_validation_augmentation, load_channel_stats, NUM_CHANNELS
from models import build_model
from segmentation_utils import _eval_batch
from experiment_timing import fmt_duration, print_timing_block, sec_since
from train import run_training, RECIPE_PRESETS, resolve_two_head
from visual_reporting import save_zoom_comparison

EXPERIMENT_SPECS = {
    "taunet_resnet50": {
        "display_name": "TAUNet-ResNet50",
        "two_head": False,
        "batch_size": 4,
    },
    "taunet": {
        "display_name": "TAUNet",
        "two_head": False,
        "batch_size": 4,
    },
    "resunext": {
        "display_name": "ResUNext",
        "two_head": False,
        "batch_size": 4,
    },
    "unet_resnet50": {
        "display_name": "UNet-ResNet50",
        "two_head": False,
        "batch_size": 4,
    },
}

DEFAULT_MODEL_ORDER = ["taunet_resnet50", "taunet", "resunext"]


def _paths(model_key: str, root: str, variant: str = "no_aug") -> dict:
    return {
        "model_key": model_key,
        "variant": variant,
        "run_dir": os.path.join(root, "results", model_key, variant),
        "checkpoint": os.path.join(root, "saved_models", variant, f"{model_key}_best.pth"),
        "eval_config": os.path.join(root, "results", model_key, variant, "eval_config.json"),
        "train_meta": os.path.join(root, "results", model_key, variant, "train_summary.json"),
    }


def _safe_record_step(record_fn, step_name, duration_sec, **extra):
    """Log pipeline timing without failing train/eval if logging kwargs clash."""
    try:
        record_fn(step_name, duration_sec, **extra)
    except Exception as exc:
        print(f"[warning] timing log failed for {step_name}: {exc}")


def _banner(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def _variant_already_done(root: str, model_key: str, variant: str) -> bool:
    summary_path = os.path.join(root, "results", model_key, variant, "train_summary.json")
    if not os.path.isfile(summary_path):
        return False
    try:
        with open(summary_path, encoding="utf-8") as f:
            summary = json.load(f)
        return not summary.get("interrupted", False) and "error" not in summary
    except Exception:
        return False


def _load_train_two_head(root: str, model_key: str, variant: str) -> bool | None:
    path = os.path.join(root, "results", model_key, variant, "train_summary.json")
    if not os.path.isfile(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            summary = json.load(f)
        if "two_head" in summary:
            return bool(summary["two_head"])
    except Exception:
        pass
    return None


def _detect_two_head(state_dict):
    return any(k.startswith("debris_head") or k.startswith("type_head") for k in state_dict)


def train_experiment_variant(
    marida_path: str, model_key: str, root: str, *,
    variant: str, epochs: int, patience: int,
    lr: float | None, debris_boost: float | None,
    use_deep_sup: bool | None, use_ohem: bool | None,
    use_amp: bool, num_workers: int, pin_memory: bool,
    cooldown_between_epochs: float = 0.0, resume: bool = True,
    max_steps_per_epoch=None, recipe: str = "pretrained_strong",
    two_head_override: bool | None = None,
    ssl4eo_strict: bool | None = None,
    init_checkpoint: str | None = None,
) -> dict:
    spec = EXPERIMENT_SPECS[model_key]
    run_dir = os.path.join(root, "results", model_key, variant)

    if resume and _variant_already_done(root, model_key, variant):
        print(f"[{model_key}/{variant}] deja antrenat (train_summary.json există). Skip.")
        with open(os.path.join(run_dir, "train_summary.json"), encoding="utf-8") as f:
            return json.load(f)

    t0 = time.time()
    two_head = resolve_two_head(model_key, recipe, two_head_override)
    if model_key in ("taunet", "taunet_resnet50"):
        mode = "multi-task (seg + debris + type)" if two_head else "single-head (4-class seg only)"
        print(f"[{model_key}/{variant}] TAUNet two_head={two_head} → {mode}")

    train_result = None
    fallback_steps = [
        {"batch_size": spec["batch_size"], "use_deep_sup": use_deep_sup, "use_ohem": use_ohem},
        {"batch_size": 2, "use_deep_sup": use_deep_sup, "use_ohem": use_ohem},
        {"batch_size": 1, "use_deep_sup": use_deep_sup, "use_ohem": use_ohem},
        {"batch_size": 1, "use_deep_sup": False, "use_ohem": False},
    ]
    last_exc = None
    for step in fallback_steps:
        try:
            print(f"[{model_key}/{variant}] trying batch={step['batch_size']} "
                  f"deep_sup={step['use_deep_sup']} ohem={step['use_ohem']} amp={use_amp} "
                  f"recipe={recipe} two_head={two_head}")
            train_result = run_training(
                marida_path, model_key,
                epochs=epochs, batch_size=step["batch_size"],
                lr=lr, patience=patience, two_head=two_head,
                use_deep_sup=step["use_deep_sup"], use_ohem=step["use_ohem"],
                debris_boost=debris_boost,
                saved_models_dir=os.path.join(root, "saved_models", variant),
                run_dir=run_dir,
                augmentation_enabled=(variant == "aug"),
                use_amp=use_amp, num_workers=num_workers, pin_memory=pin_memory,
                cooldown_between_epochs=cooldown_between_epochs,
                resume=resume, max_steps_per_epoch=max_steps_per_epoch,
                recipe=recipe,
                ssl4eo_strict=ssl4eo_strict,
                init_checkpoint=init_checkpoint,
            )
            train_result["effective_batch_size"] = step["batch_size"]
            break
        except KeyboardInterrupt:
            raise
        except RuntimeError as exc:
            last_exc = exc
            if "out of memory" not in str(exc).lower():
                raise
            print(f"[OOM] {model_key}/{variant}: {exc}")
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

    if train_result is None:
        raise RuntimeError(f"OOM fallback failed for {model_key}/{variant}: {last_exc}")

    train_result["display_name"] = spec["display_name"]
    train_result["variant"] = variant
    train_result["wall_duration_sec"] = round(time.time() - t0, 1)
    if "duration_sec" not in train_result:
        train_result["duration_sec"] = train_result["wall_duration_sec"]
    train_result["wall_duration_fmt"] = fmt_duration(train_result["wall_duration_sec"])

    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "train_summary.json"), "w", encoding="utf-8") as f:
        json.dump(train_result, f, indent=2)
    if train_result.get("timing"):
        with open(os.path.join(run_dir, "train_timing.json"), "w", encoding="utf-8") as f:
            json.dump(train_result["timing"], f, indent=2)
    return train_result


def evaluate_variant(marida_path, model_key, root, *, variant, tune_on_val,
                     force_retune, eval_workers, fast_tune=True, recipe="md_outlier"):
    t0 = time.time()
    spec = EXPERIMENT_SPECS[model_key]
    ckpt = os.path.join(root, "saved_models", variant, f"{model_key}_best.pth")
    results_dir = os.path.join(root, "results", model_key, variant)
    eval_cfg = os.path.join(results_dir, "eval_config.json")
    if force_retune and os.path.isfile(eval_cfg):
        os.remove(eval_cfg)

    two_head_hint = _load_train_two_head(root, model_key, variant)
    if two_head_hint is None and os.path.isfile(ckpt):
        import torch
        try:
            state = torch.load(ckpt, map_location="cpu", weights_only=True)
        except TypeError:
            state = torch.load(ckpt, map_location="cpu")
        two_head_hint = _detect_two_head(state)

    print(f"[{model_key}/{variant}] eval two_head="
          f"{two_head_hint if two_head_hint is not None else 'auto-detect from checkpoint'}")

    from train import RECIPE_PRESETS
    preset = RECIPE_PRESETS.get(recipe, {})
    tune_mode = preset.get("eval_tune_mode", "miou")
    default_eval = None
    if tune_mode == "md_plastic":
        default_eval = {
            "decode_mode": "two_head",
            "debris_threshold": preset.get("eval_debris_threshold", 0.40),
            "type_confidence_threshold": 0.0,
            "require_seg_fg": False,
            "tta_scales": list(preset.get("tta_scales_eval", [1.0])),
            "use_crf": preset.get("eval_use_crf", False),
            "min_component_size": preset.get("eval_min_component_size", 0),
        }

    result = evaluate_pipeline(
        marida_path, ckpt, model_key=model_key,
        model_name=f"{spec['display_name']} ({variant})",
        results_dir=results_dir, eval_config_path=eval_cfg,
        tune_on_val=tune_on_val, two_head=two_head_hint,
        num_workers=eval_workers, fast_tune=fast_tune,
        tune_mode=tune_mode, default_eval_config=default_eval,
        tune_md_weight=float(preset.get("checkpoint_md_iou_weight", 0.5)),
        tune_plastic_weight=float(preset.get("checkpoint_plastic_iou_weight", 0.5)),
        tune_plastic_precision_weight=float(
            preset.get("checkpoint_plastic_precision_weight", 0.0)
        ),
        tune_prefer_threshold=preset.get("eval_prefer_threshold"),
        tune_threshold_score_tol=float(preset.get("eval_threshold_score_tol", 0.01)),
    )
    result["wall_duration_sec"] = round(time.time() - t0, 1)
    if "duration_sec" not in result:
        result["duration_sec"] = result["wall_duration_sec"]
    result["variant"] = variant
    return result


def build_visual_comparisons(marida_path, model_key, root, *, max_samples=8):
    t0 = time.perf_counter()
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    no_path = os.path.join(root, "saved_models", "no_aug", f"{model_key}_best.pth")
    aug_path = os.path.join(root, "saved_models", "aug", f"{model_key}_best.pth")
    state_no = torch.load(no_path, map_location=device)
    state_aug = torch.load(aug_path, map_location=device)
    th_no = _detect_two_head(state_no)
    th_aug = _detect_two_head(state_aug)
    ds_no = any(k.startswith("aux_head") for k in state_no)
    ds_aug = any(k.startswith("aux_head") for k in state_aug)
    ssag_no = any(k.startswith("ssag") for k in state_no)
    ssag_aug = any(k.startswith("ssag") for k in state_aug)

    model_no = build_model(model_key, in_channels=NUM_CHANNELS, two_head=th_no,
                           deep_supervision=ds_no, use_ssag=ssag_no).to(device)
    model_aug = build_model(model_key, in_channels=NUM_CHANNELS, two_head=th_aug,
                            deep_supervision=ds_aug, use_ssag=ssag_aug).to(device)
    model_no.load_state_dict(state_no)
    model_aug.load_state_dict(state_aug)
    model_no.eval()
    model_aug.eval()

    def _load_cfg(variant):
        p = os.path.join(root, "results", model_key, variant, "eval_config.json")
        if os.path.isfile(p):
            with open(p, encoding="utf-8") as f:
                return json.load(f)
        return {"tta_scales": [1.0], "use_crf": False, "min_component_size": 8, "debris_threshold": 0.5}

    cfg_no = _load_cfg("no_aug")
    cfg_aug = _load_cfg("aug")
    ds = MARIDADataset(marida_path, split="test", transform=get_validation_augmentation())
    out_dir = os.path.join(root, "results", model_key, "comparisons")
    os.makedirs(out_dir, exist_ok=True)

    ch_stats = load_channel_stats(marida_path)
    ch_mean = np.array(ch_stats["mean"], dtype=np.float64)
    ch_std = np.array(ch_stats["std"], dtype=np.float64)

    for idx in range(min(max_samples, len(ds))):
        image, mask = ds[idx]
        image_b = image.unsqueeze(0).to(device)
        pred_no = _eval_batch(model_no, image_b, None, device, cfg_no, th_no)
        pred_aug = _eval_batch(model_aug, image_b, None, device, cfg_aug, th_aug)
        save_zoom_comparison(
            image, mask.numpy(), pred_no, pred_aug,
            os.path.join(out_dir, f"zoom_compare_{idx}.png"),
            model_name=EXPERIMENT_SPECS[model_key]["display_name"],
            channel_mean=ch_mean, channel_std=ch_std,
        )

    duration = sec_since(t0)
    print(f"[timing] visual comparisons {model_key}: {fmt_duration(duration)}")
    return {"wall_duration_sec": duration, "samples": min(max_samples, len(ds))}


def _load_train_summary(root: str, model_key: str, variant: str) -> dict | None:
    path = _paths(model_key, root, variant)["train_meta"]
    if not os.path.isfile(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def build_comparison_row(model_key, root, train_info, eval_info):
    spec = EXPERIMENT_SPECS[model_key]
    per_class_raw = eval_info["metrics_per_class"]
    per_class = {int(k): v for k, v in per_class_raw.items()}
    debris = eval_info["debris_metrics"]

    variant = (
        (train_info or {}).get("variant")
        or eval_info.get("variant")
        or os.path.basename(eval_info.get("results_dir", ""))
        or "no_aug"
    )
    if (not train_info or "error" in train_info) and variant:
        train_info = _load_train_summary(root, model_key, variant) or train_info
    paths = _paths(model_key, root, variant)

    if train_info and "two_head" in train_info:
        two_head_used = bool(train_info["two_head"])
    else:
        ckpt = (train_info or {}).get("checkpoint_path") or paths["checkpoint"]
        two_head_used = False
        if ckpt and os.path.isfile(ckpt):
            import torch
            try:
                state = torch.load(ckpt, map_location="cpu", weights_only=True)
            except TypeError:
                state = torch.load(ckpt, map_location="cpu")
            two_head_used = _detect_two_head(state)
    label = f"{spec['display_name']} ({variant})" if variant else spec["display_name"]
    row = {
        "model_key": model_key,
        "display_name": spec["display_name"],
        "variant": variant,
        "label": label,
        "two_head": two_head_used,
        "checkpoint": (train_info or {}).get("checkpoint_path", paths["checkpoint"]),
        "results_dir": eval_info["results_dir"],
        "mIoU_foreground": debris["mIoU_foreground"],
        "binary_debris_IoU": debris["binary_debris_IoU"],
        "marida_md_IoU": debris.get("marida_md_IoU", debris["binary_debris_IoU"]),
        "plastic_IoU": debris.get("plastic_IoU", per_class.get(1, {}).get("IoU", 0.0)),
        "plastic_recall": debris.get("plastic_recall", per_class.get(1, {}).get("Recall", 0.0)),
    }
    for c in range(4):
        row[f"IoU_class_{c}"] = per_class[c]["IoU"]
        row[f"F1_class_{c}"] = per_class[c]["F1-Score"]
        row[f"Precision_class_{c}"] = per_class[c]["Precision"]
        row[f"Recall_class_{c}"] = per_class[c]["Recall"]
    if train_info:
        row["best_val_miou_fg"] = train_info.get("best_val_miou_fg")
        row["train_epochs_ran"] = train_info.get("epochs_ran")
        row["train_duration_sec"] = train_info.get("duration_sec")
        row["train_duration_fmt"] = train_info.get("duration_fmt")
        if train_info.get("timing"):
            row["train_avg_epoch_sec"] = train_info["timing"].get("avg_epoch_sec")
            row["train_avg_epoch_fmt"] = train_info["timing"].get("avg_epoch_fmt")
    row["eval_duration_sec"] = eval_info.get("duration_sec")
    row["eval_duration_fmt"] = eval_info.get("duration_fmt")
    if eval_info.get("timing"):
        row["eval_tune_sec"] = eval_info["timing"].get("tune_on_val_sec")
        row["eval_tune_fmt"] = eval_info["timing"].get("tune_on_val_fmt")
        row["eval_test_sec"] = eval_info["timing"].get("test_inference_sec")
        row["eval_test_fmt"] = eval_info["timing"].get("test_inference_fmt")
    return row


def save_comparison_report(rows, root):
    os.makedirs(os.path.join(root, "results"), exist_ok=True)
    df = pd.DataFrame(rows)
    csv_path = os.path.join(root, "results", "raport_comparativ_modele.csv")
    df.to_csv(csv_path, index=False)
    json_path = os.path.join(root, "results", "raport_comparativ_modele.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

    names = [f"{r['display_name']}\n({r.get('variant', '')})" for r in rows]
    miou = [r["mIoU_foreground"] for r in rows]
    debris_iou = [r["binary_debris_IoU"] for r in rows]
    x = range(len(names))
    w = 0.32

    fig, ax = plt.subplots(figsize=(10, 5.5), facecolor="white")
    b1 = ax.bar([i - w / 2 for i in x], miou, width=w, label="mIoU foreground (1-3)",
                color="#4363D8", edgecolor="white")
    b2 = ax.bar([i + w / 2 for i in x], debris_iou, width=w, label="Binary debris IoU (1+2)",
                color="#3CB44B", edgecolor="white")
    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            if h > 0.005:
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01, f"{h:.3f}",
                        ha="center", va="bottom", fontsize=8, color="#444444")
    ax.set_xticks(list(x))
    ax.set_xticklabels(names, fontsize=9)
    ax.set_ylim(0, min(1.0, max(max(miou), max(debris_iou)) * 1.3 + 0.05))
    ax.set_ylabel("IoU", fontsize=11)
    ax.set_title("Model Comparison — MARIDA Test Set", fontsize=13, fontweight="bold", pad=12)
    ax.legend(fontsize=9, frameon=True, fancybox=True, edgecolor="#cccccc")
    ax.grid(axis="y", linestyle="--", alpha=0.4, color="#cccccc")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    chart_path = os.path.join(root, "results", "grafic_comparativ_miou.png")
    fig.savefig(chart_path, dpi=300, facecolor="white", edgecolor="none")
    plt.close(fig)

    try:
        bench_path = plot_benchmark_comparison(rows, os.path.join(root, "results"))
        if bench_path:
            print(f"Benchmark comparison: {bench_path}")
    except Exception as exc:
        print(f"[warning] plot_benchmark_comparison failed: {exc}")

    _banner("RAPORT COMPARATIV")
    cols = [c for c in [
        "display_name", "variant", "marida_md_IoU", "plastic_IoU", "plastic_recall",
        "mIoU_foreground", "binary_debris_IoU", "IoU_class_1", "IoU_class_2", "IoU_class_3",
    ] if c in df.columns]
    print(df[cols].to_string(index=False))
    print(f"\nSalvat: {csv_path}")
    return csv_path


def parse_args():
    p = argparse.ArgumentParser(description="Experimente MARIDA — Disertatie_2")
    p.add_argument("--marida", type=str, default=r"D:\TAID\Disertatie\MARIDA")
    p.add_argument("--root", type=str, default=".", help="Project root")
    p.add_argument("--models", type=str, default=",".join(DEFAULT_MODEL_ORDER))
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--patience", type=int, default=35)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--debris-boost", type=float, default=None)
    p.add_argument("--recipe", type=str, default="md_outlier",
                   choices=list(RECIPE_PRESETS.keys()))
    two_head_grp = p.add_mutually_exclusive_group()
    two_head_grp.add_argument("--two-head", action="store_true")
    two_head_grp.add_argument("--no-two-head", action="store_true")
    p.add_argument("--skip-train", action="store_true")
    p.add_argument("--skip-eval", action="store_true")
    p.add_argument("--no-tune", action="store_true")
    p.add_argument("--retune", action="store_true")
    p.add_argument("--no-deep-sup", action="store_true")
    p.add_argument("--no-ohem", action="store_true")
    p.add_argument("--fast", action="store_true")
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--num-workers", type=int, default=6)
    p.add_argument("--eval-workers", type=int, default=0)
    p.add_argument("--pin-memory", action="store_true")
    p.add_argument("--cooldown", type=float, default=0.0)
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("--full-tune", action="store_true")
    p.add_argument("--smoke", action="store_true")
    p.add_argument(
        "--allow-imagenet-backbone",
        action="store_true",
        help="Allow ImageNet fallback if SSL4EO-S12 cannot load (default: abort)",
    )
    p.add_argument(
        "--init-checkpoint",
        type=str,
        default=None,
        help="Load model weights before training (overrides recipe init_weights_path)",
    )
    return p.parse_args()


def main():
    args = parse_args()
    root = os.path.abspath(args.root)
    marida = os.path.abspath(args.marida)

    if not os.path.isdir(marida):
        print(f"EROARE: folderul MARIDA nu există: {marida}")
        sys.exit(1)

    model_keys = [m.strip().lower() for m in args.models.split(",") if m.strip()]
    for key in model_keys:
        if key not in EXPERIMENT_SPECS:
            print(f"Model necunoscut: {key}. Valide: {list(EXPERIMENT_SPECS)}")
            sys.exit(1)

    recipe_defaults = RECIPE_PRESETS.get(args.recipe, {})
    default_epochs = recipe_defaults.get("default_epochs", 150)
    default_patience = recipe_defaults.get("default_patience", 35)

    if args.fast:
        epochs, patience = 3, 2
    else:
        epochs = args.epochs if args.epochs != 150 else default_epochs
        patience = args.patience if args.patience != 35 else default_patience

    tune_on_val = not args.no_tune
    use_amp = not args.no_amp
    max_steps_per_epoch = 1 if args.smoke else None

    os.makedirs(os.path.join(root, "saved_models", "no_aug"), exist_ok=True)
    os.makedirs(os.path.join(root, "saved_models", "aug"), exist_ok=True)
    os.makedirs(os.path.join(root, "results"), exist_ok=True)

    if args.two_head:
        two_head_override = True
    elif args.no_two_head:
        two_head_override = False
    else:
        two_head_override = None

    manifest = {
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "project": "Disertatie_2",
        "marida_path": marida,
        "project_root": root,
        "models": model_keys,
        "recipe": args.recipe,
        "in_channels": NUM_CHANNELS,
        "epochs": epochs,
        "patience": patience,
    }
    manifest_path = os.path.join(root, "results", "experiment_manifest.json")

    _banner(f"DISERTATIE_2 EXPERIMENTS | {len(model_keys)} models | 14ch | MARIDA={marida}")
    print(f"Project root: {root}")
    print(f"Recipe: {args.recipe} | Input channels: {NUM_CHANNELS}")
    preset = RECIPE_PRESETS[args.recipe]
    print(
        f"[config] {args.recipe} expects: mode={preset.get('training_mode', 'standard')} "
        f"two_head={preset.get('two_head_default')} "
        f"lr={preset.get('lr')} freeze={preset.get('freeze_encoder_epochs')} "
        f"plastic_boost={preset.get('plastic_boost')} lovasz={preset.get('lovasz_weight')} "
        f"crop_mining={preset.get('crop_mining', False)} "
        f"variants={preset.get('variants', ['no_aug', 'aug'])}"
    )
    if preset.get("training_mode") == "md_outlier":
        print("[config] Outlier-first MARIDA MD pipeline — binary debris head primary, no_aug only")
    elif preset.get("two_head_default") and not args.no_two_head:
        print("[config] Two-head training ENABLED (ablation — use pretrained_strong for single-head)")
    else:
        print("[config] Single-head 4-class seg (history4 recipe)")

    do_resume = not args.no_resume
    recipe_variants = tuple(preset.get("variants", ("no_aug", "aug")))
    train_results = {v: {} for v in recipe_variants}
    eval_results = {v: {} for v in recipe_variants}
    pipeline_timing = {"steps": [], "by_model": {}}
    t_pipeline = time.perf_counter()

    def _record_step(step_name, duration_sec, **extra):
        entry = {
            "step": step_name,
            "duration_sec": round(float(duration_sec), 2),
            "duration_fmt": fmt_duration(duration_sec),
            **extra,
        }
        pipeline_timing["steps"].append(entry)
        print(f"[timing] {step_name}: {fmt_duration(duration_sec)}")

    def _save_manifest(status="in_progress"):
        manifest["status"] = status
        manifest["updated_at"] = datetime.now().isoformat(timespec="seconds")
        manifest["pipeline_timing"] = pipeline_timing
        manifest["pipeline_elapsed_sec"] = sec_since(t_pipeline)
        try:
            with open(manifest_path, "w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=2)
        except Exception:
            pass

    _train_kwargs = dict(
        epochs=epochs, patience=patience, lr=args.lr,
        debris_boost=args.debris_boost,
        use_deep_sup=False if args.no_deep_sup else None,
        use_ohem=False if args.no_ohem else None,
        use_amp=use_amp, num_workers=args.num_workers,
        pin_memory=args.pin_memory, cooldown_between_epochs=args.cooldown,
        resume=do_resume, max_steps_per_epoch=max_steps_per_epoch,
        recipe=args.recipe, two_head_override=two_head_override,
        ssl4eo_strict=False if args.allow_imagenet_backbone else None,
        init_checkpoint=args.init_checkpoint,
    )

    user_stopped = False
    try:
        for model_key in model_keys:
            spec = EXPERIMENT_SPECS[model_key]
            _banner(f"MODEL: {spec['display_name']} ({model_key})")

            if not args.skip_train:
                for variant in recipe_variants:
                    try:
                        t_step = time.perf_counter()
                        result = train_experiment_variant(
                            marida, model_key, root, variant=variant, **_train_kwargs)
                        train_results[variant][model_key] = result
                        _safe_record_step(
                            _record_step,
                            f"train/{model_key}/{variant}",
                            sec_since(t_step),
                            model=model_key,
                            variant=variant,
                            wall_duration_sec=result.get("duration_sec"),
                        )
                        pipeline_timing.setdefault("by_model", {}).setdefault(model_key, {})[variant] = {
                            "train_sec": result.get("duration_sec"),
                            "train_timing": result.get("timing"),
                        }
                        _save_manifest()
                        hist_csv = result.get("history_path")
                        if hist_csv and os.path.isfile(hist_csv):
                            curve_dir = os.path.join(root, "results", model_key, variant)
                            try:
                                t_plot = time.perf_counter()
                                plot_training_curves(hist_csv, curve_dir,
                                                    f"{spec['display_name']} ({variant})")
                                plot_training_dashboard(
                                    hist_csv, curve_dir, f"{spec['display_name']} ({variant})",
                                )
                                _safe_record_step(
                                    _record_step,
                                    f"plot_curves/{model_key}/{variant}",
                                    sec_since(t_plot),
                                    model=model_key,
                                    variant=variant,
                                )
                            except Exception as exc:
                                print(f"[warning] training curves plot failed: {exc}")
                        if result.get("interrupted"):
                            user_stopped = True
                            break
                        if args.cooldown > 0:
                            time.sleep(args.cooldown)
                    except KeyboardInterrupt:
                        user_stopped = True
                        break
                    except Exception as exc:
                        import traceback
                        print(f"EROARE antrenare {model_key}/{variant}: {exc}")
                        traceback.print_exc()
                        train_results[variant][model_key] = {"error": str(exc)}
                if user_stopped:
                    break

            if not args.skip_eval and not user_stopped:
                for variant in recipe_variants:
                    ckpt = os.path.join(root, "saved_models", variant, f"{model_key}_best.pth")
                    if not os.path.isfile(ckpt):
                        print(f"[{model_key}/{variant}] checkpoint lipsă. Evaluare omisă.")
                        continue
                    try:
                        t_step = time.perf_counter()
                        from train import RECIPE_PRESETS as _RECIPE_PRESETS
                        _preset = _RECIPE_PRESETS.get(args.recipe, {})
                        if args.full_tune:
                            fast_tune = False
                        elif "eval_fast_tune" in _preset:
                            fast_tune = bool(_preset["eval_fast_tune"])
                        else:
                            fast_tune = not args.full_tune
                        if args.recipe in (
                            "strong", "strong_two_head",
                            "pretrained_strong", "pretrained_strong_two_head",
                        ) and not args.full_tune and "eval_fast_tune" not in _preset:
                            fast_tune = False
                        eval_results[variant][model_key] = evaluate_variant(
                            marida, model_key, root, variant=variant,
                            tune_on_val=tune_on_val, force_retune=args.retune,
                            eval_workers=args.eval_workers, fast_tune=fast_tune,
                            recipe=args.recipe,
                        )
                        ev = eval_results[variant][model_key]
                        _safe_record_step(
                            _record_step,
                            f"eval/{model_key}/{variant}",
                            sec_since(t_step),
                            model=model_key,
                            variant=variant,
                            wall_duration_sec=ev.get("duration_sec"),
                        )
                        pipeline_timing.setdefault("by_model", {}).setdefault(model_key, {}).setdefault(variant, {})
                        pipeline_timing["by_model"][model_key][variant]["eval_sec"] = ev.get("duration_sec")
                        pipeline_timing["by_model"][model_key][variant]["eval_timing"] = ev.get("timing")
                        _save_manifest()
                    except KeyboardInterrupt:
                        user_stopped = True
                        break
                    except Exception as exc:
                        print(f"EROARE evaluare {model_key}/{variant}: {exc}")
                if user_stopped:
                    break
                try:
                    no_ckpt = os.path.join(root, "saved_models", "no_aug", f"{model_key}_best.pth")
                    aug_ckpt = os.path.join(root, "saved_models", "aug", f"{model_key}_best.pth")
                    if os.path.isfile(no_ckpt) and os.path.isfile(aug_ckpt):
                        t_step = time.perf_counter()
                        vis_timing = build_visual_comparisons(marida, model_key, root, max_samples=8)
                        _safe_record_step(
                            _record_step,
                            f"visual_compare/{model_key}",
                            sec_since(t_step),
                            model=model_key,
                            samples=(vis_timing or {}).get("samples"),
                            wall_duration_sec=(vis_timing or {}).get("wall_duration_sec"),
                        )
                except Exception as exc:
                    print(f"EROARE comparații vizuale {model_key}: {exc}")

    except KeyboardInterrupt:
        user_stopped = True

    comparison_rows = []
    for variant in recipe_variants:
        for mk in model_keys:
            ev = eval_results[variant].get(mk)
            if not ev or "error" in ev:
                continue
            tr = train_results[variant].get(mk)
            if isinstance(tr, dict) and "error" in tr:
                tr = None
            row = build_comparison_row(mk, root, tr, ev)
            row["variant"] = variant
            comparison_rows.append(row)

    if comparison_rows:
        t_step = time.perf_counter()
        save_comparison_report(comparison_rows, root)
        _record_step("comparison_report", sec_since(t_step))

    overlay_pairs = []
    for variant in recipe_variants:
        for mk in model_keys:
            hist_csv = os.path.join(root, "results", mk, variant, "istoric_antrenare.csv")
            if os.path.isfile(hist_csv):
                overlay_pairs.append((f"{EXPERIMENT_SPECS[mk]['display_name']} ({variant})", hist_csv))
    if overlay_pairs:
        try:
            t_step = time.perf_counter()
            plot_training_curves_overlay(overlay_pairs, os.path.join(root, "results"))
            _record_step("plot_curves_overlay", sec_since(t_step))
        except Exception:
            pass

    final_status = "interrupted_by_user" if user_stopped else "finished"
    manifest["finished_at"] = datetime.now().isoformat(timespec="seconds")
    pipeline_timing["total_sec"] = sec_since(t_pipeline)
    pipeline_timing["total_fmt"] = fmt_duration(pipeline_timing["total_sec"])
    _save_manifest(final_status)

    timing_report_path = os.path.join(root, "results", "experiment_timing.json")
    with open(timing_report_path, "w", encoding="utf-8") as f:
        json.dump(pipeline_timing, f, indent=2)

    print(f"\nManifest experiment: {manifest_path}")
    print(f"Raport timing: {timing_report_path}")
    print_timing_block("Pipeline experiment (total)", {"total_sec": pipeline_timing["total_sec"]})
    if pipeline_timing["steps"]:
        print("\n[timing] Pași:")
        for entry in pipeline_timing["steps"]:
            extra = ""
            if entry.get("duration_sec") is not None and entry["step"].startswith(("train/", "eval/")):
                inner = entry.get("duration_sec")
                if inner is not None:
                    extra = f" (intern: {fmt_duration(inner)})"
            print(f"  {entry['step']}: {fmt_duration(entry['duration_sec'])}{extra}")

    if user_stopped:
        _banner("OPRIT DE UTILIZATOR")
    else:
        _banner("FINALIZAT")


if __name__ == "__main__":
    main()
