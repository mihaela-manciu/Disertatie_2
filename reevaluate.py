"""
Re-evaluate trained checkpoint(s) with thesis-aligned decode tuning.

Fixes the broken default eval path (mIoU-only tune → thr=0.6 → plastic IoU=0).
Primary thesis row: **test @ training thr=0.40** (not val-tuned thr=0.30).

Defaults to saved_models/no_aug/taunet_resnet50_best.pth and reads train_summary.json.

Usage (latest run — history26 / md_outlier_refine):
  python reevaluate.py --marida "D:\\TAID\\Disertatie\\MARIDA" --root "D:\\TAID\\Disertatie\\Disertatie_2" --num-workers 4 --pin-memory --all-checkpoints

Single checkpoint:
  python reevaluate.py --marida MARIDA --root . --num-workers 4 --pin-memory

Optional:
  python reevaluate.py --checkpoint saved_models/no_aug/taunet_resnet50_best_md.pth --training-thr 0.40
  python reevaluate.py --all-checkpoints --compare-baseline results_3/taunet_resnet50/no_aug
"""

from __future__ import annotations

import argparse
import json
import os
import time

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

import pandas as pd
import torch
from torch.utils.data import DataLoader

from dataset import MARIDADataset, get_validation_augmentation, NUM_CHANNELS
from experiment_timing import fmt_duration, sec_since
from models import build_model
from segmentation_utils import (
    CLASE_NUME,
    evaluate_loader_with_config,
    load_eval_config,
    training_baseline_eval_config,
    tune_thresholds_md_plastic,
)
from training_stability import md_plastic_checkpoint_score

BENCHMARK_TARGETS = {
    "marida_md_IoU": (0.35, 0.55),
    "plastic_IoU": (0.15, 0.25),
    "mIoU_foreground": (0.35, 0.42),
}


def _load_model(checkpoint_path, model_key, device):
    try:
        state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(checkpoint_path, map_location=device)
    two_head = any(k.startswith("debris_head") or k.startswith("type_head") for k in state)
    has_deep_sup = any(k.startswith("aux_head") for k in state)
    has_ssag = any(k.startswith("ssag") for k in state)
    model = build_model(
        model_key,
        in_channels=NUM_CHANNELS,
        two_head=two_head,
        deep_supervision=has_deep_sup,
        use_ssag=has_ssag,
    ).to(device)
    model.load_state_dict(state)
    model.eval()
    return model, two_head


def _metrics_row(split: str, label: str, cfg: dict, metrics: dict) -> dict:
    per = metrics.get("per_class", {})
    return {
        "split": split,
        "label": label,
        "debris_threshold": cfg.get("debris_threshold"),
        "use_crf": cfg.get("use_crf"),
        "min_component_size": cfg.get("min_component_size"),
        "tta_scales": cfg.get("tta_scales"),
        "mIoU_foreground": metrics["mIoU_foreground"],
        "marida_md_IoU": metrics["marida_md_IoU"],
        "plastic_IoU": metrics["plastic_IoU"],
        "plastic_precision": per.get(1, {}).get("Precision", 0.0),
        "plastic_recall": metrics.get("plastic_recall", per.get(1, {}).get("Recall", 0.0)),
        "natural_IoU": per.get(2, {}).get("IoU", 0.0),
        "ships_IoU": per.get(3, {}).get("IoU", 0.0),
        "md_plastic_score": md_plastic_checkpoint_score(metrics),
        "config_json": json.dumps(cfg, sort_keys=True),
    }


def _print_metrics_block(title: str, metrics: dict, cfg: dict):
    per = metrics.get("per_class", {})
    print(f"\n{'=' * 64}\n{title}\n{'=' * 64}")
    print(f"Config: {json.dumps(cfg, indent=2)}")
    for c in range(1, 4):
        m = per[c]
        print(
            f"  {CLASE_NUME[c]}: IoU={m['IoU']:.4f}  P={m['Precision']:.3f}  "
            f"R={m['Recall']:.3f}"
        )
    print(f"  MARIDA MD IoU (1+2): {metrics['marida_md_IoU']:.4f}")
    print(f"  Plastic IoU:         {metrics['plastic_IoU']:.4f}")
    print(f"  mIoU foreground:     {metrics['mIoU_foreground']:.4f}")
    print(f"  MD+plastic score:    {md_plastic_checkpoint_score(metrics):.4f}")


CHECKPOINT_SPECS = (
    ("best", "_best.pth", "best_val_md_plastic_epoch"),
    ("best_plastic", "_best_plastic.pth", "best_plastic_only_epoch"),
    ("best_md", "_best_md.pth", "best_md_only_epoch"),
)


def _discover_checkpoints(
    project_root: str,
    model_key: str,
    variant: str,
    train_summary: dict | None,
    explicit_checkpoint: str | None,
) -> list[tuple[str, str, int | None]]:
    """Return [(label, abs_path, train_epoch_or_none), ...]."""
    if explicit_checkpoint is not None:
        path = os.path.abspath(explicit_checkpoint)
        return [("custom", path, None)]

    models_dir = os.path.join(project_root, "saved_models", variant)
    found: list[tuple[str, str, int | None]] = []
    for label, suffix, epoch_key in CHECKPOINT_SPECS:
        path = os.path.join(models_dir, f"{model_key}{suffix}")
        if not os.path.isfile(path):
            continue
        epoch = None
        if train_summary is not None:
            epoch = train_summary.get(epoch_key)
        found.append((label, path, epoch))
    if not found:
        default = os.path.join(models_dir, f"{model_key}_best.pth")
        if os.path.isfile(default):
            found.append(("best", default, train_summary.get("best_val_md_plastic_epoch") if train_summary else None))
    return found


def _load_baseline_reeval(baseline_results_dir: str) -> dict | None:
    csv_path = os.path.join(baseline_results_dir, "reeval_metrics.csv")
    if not os.path.isfile(csv_path):
        return None
    df = pd.read_csv(csv_path)
    baseline = {}
    for _, row in df.iterrows():
        key = f"{row['split']}_{row['label']}"
        baseline[key] = row.to_dict()
    return baseline


def _print_checkpoint_comparison(
    reports: list[dict],
    baseline: dict | None,
):
    print(f"\n{'=' * 72}")
    print("CHECKPOINT COMPARISON — test @ training baseline thr=0.40")
    print(f"{'=' * 72}")
    header = f"{'checkpoint':<14} {'epoch':>5} {'plastic':>8} {'MD':>8} {'mIoU_fg':>8} {'pl_P':>7} {'pl_R':>7}"
    print(header)
    print("-" * len(header))
    if baseline and "test_training_baseline_thr040" in baseline:
        b = baseline["test_training_baseline_thr040"]
        print(
            f"{'results_3':<14} {'87':>5} "
            f"{b['plastic_IoU']:>8.4f} {b['marida_md_IoU']:>8.4f} "
            f"{b['mIoU_foreground']:>8.4f} {b['plastic_precision']:>7.3f} "
            f"{b['plastic_recall']:>7.3f}"
        )
    for rep in reports:
        row = rep["results"]["test_training_baseline"]
        ep = rep.get("train_epoch")
        ep_str = str(ep) if ep is not None else "?"
        print(
            f"{rep['checkpoint_label']:<14} {ep_str:>5} "
            f"{row['plastic_IoU']:>8.4f} {row['marida_md_IoU']:>8.4f} "
            f"{row['mIoU_foreground']:>8.4f} {row['plastic_precision']:>7.3f} "
            f"{row['plastic_recall']:>7.3f}"
        )


def _load_train_summary(results_dir: str) -> dict | None:
    path = os.path.join(results_dir, "train_summary.json")
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _benchmark_gap(value: float, key: str) -> str:
    lo, hi = BENCHMARK_TARGETS[key]
    if value >= lo:
        return f"within target ({lo:.2f}–{hi:.2f})"
    pct = 100.0 * value / lo if lo > 0 else 0.0
    return f"{pct:.0f}% of lower bound ({lo:.2f})"


def reevaluate(
    marida_path: str,
    project_root: str,
    *,
    model_key: str = "taunet_resnet50",
    variant: str = "no_aug",
    checkpoint_path: str | None = None,
    checkpoint_label: str = "best",
    train_epoch: int | None = None,
    results_dir: str | None = None,
    num_workers: int = 0,
    pin_memory: bool = False,
    fast_tune: bool = True,
    training_thr: float = 0.40,
    save_outputs: bool = True,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        print(
            "[reeval] WARNING: CUDA not available — eval on CPU is very slow "
            "(~2h+ for --all-checkpoints). Run on the 3080 Ti machine."
        )
    if checkpoint_path is None:
        checkpoint_path = os.path.join(
            project_root, "saved_models", variant, f"{model_key}_best.pth",
        )
    if results_dir is None:
        results_dir = os.path.join(project_root, "results", model_key, variant)
    os.makedirs(results_dir, exist_ok=True)

    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    train_summary = _load_train_summary(results_dir)
    t0 = time.perf_counter()
    print(f"[reeval] checkpoint: {checkpoint_path} ({checkpoint_label})")
    print(f"[reeval] results dir:  {results_dir}")
    print(f"[reeval] device:       {device}")
    print(f"[reeval] training thr: {training_thr} (matches md_outlier val decode)")
    if train_epoch is not None:
        print(f"[reeval] training epoch for this ckpt: {train_epoch}")
    if train_summary:
        print(
            f"[reeval] train_summary: recipe={train_summary.get('recipe', '?')} "
            f"best_ep={train_summary.get('best_val_md_plastic_epoch')} "
            f"val_plastic={train_summary.get('best_val_plastic_iou', 0):.4f} "
            f"val_md={train_summary.get('best_val_md_iou', 0):.4f}"
        )
    else:
        print("[reeval] train_summary.json not found — proceeding with checkpoint only")

    model, two_head = _load_model(checkpoint_path, model_key, device)
    loader_kw = dict(batch_size=1, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(
        MARIDADataset(marida_path, split="val", transform=get_validation_augmentation()),
        **loader_kw,
    )
    test_loader = DataLoader(
        MARIDADataset(marida_path, split="test", transform=get_validation_augmentation()),
        **loader_kw,
    )

    baseline_cfg = training_baseline_eval_config(
        two_head=two_head, debris_threshold=training_thr,
    )
    baseline_label = f"training_baseline_thr{training_thr:.2f}".replace(".", "")
    print(f"\n[reeval] --- Training-matched baseline (thr={training_thr}, no CRF) ---")
    val_baseline = evaluate_loader_with_config(
        model, val_loader, device, baseline_cfg, two_head, progress_desc="val baseline",
    )
    test_baseline = evaluate_loader_with_config(
        model, test_loader, device, baseline_cfg, two_head, progress_desc="test baseline",
    )
    _print_metrics_block("VAL — training baseline", val_baseline, baseline_cfg)
    _print_metrics_block("TEST — training baseline", test_baseline, baseline_cfg)

    best_row, leaderboard = tune_thresholds_md_plastic(
        model, val_loader, device, use_two_head=two_head, fast=fast_tune,
        preferred_threshold=training_thr,
        preferred_threshold_tol=0.01,
    )
    tuned_cfg = {k: best_row[k] for k in best_row if not k.startswith("val_")}
    val_tuned = evaluate_loader_with_config(
        model, val_loader, device, tuned_cfg, two_head, progress_desc="val tuned",
    )
    print("\n[reeval] --- Best config by MD+plastic score on VAL ---")
    _print_metrics_block("VAL — tuned (best)", val_tuned, tuned_cfg)

    test_tuned = evaluate_loader_with_config(
        model, test_loader, device, tuned_cfg, two_head, progress_desc="test tuned",
    )
    _print_metrics_block("TEST — tuned (apply val-best config)", test_tuned, tuned_cfg)

    old_eval_cfg = load_eval_config(os.path.join(results_dir, "eval_config.json"))
    old_test_metrics = None
    if old_eval_cfg is not None:
        print("\n[reeval] --- Previous pipeline eval config (for comparison) ---")
        old_test_metrics = evaluate_loader_with_config(
            model, test_loader, device, old_eval_cfg, two_head,
            progress_desc="test old eval_config",
        )
        _print_metrics_block("TEST — old eval_config.json", old_test_metrics, old_eval_cfg)

    rows = [
        _metrics_row("val", baseline_label, baseline_cfg, val_baseline),
        _metrics_row("test", baseline_label, baseline_cfg, test_baseline),
        _metrics_row("val", "tuned_md_plastic", tuned_cfg, val_tuned),
        _metrics_row("test", "tuned_md_plastic", tuned_cfg, test_tuned),
    ]
    if old_test_metrics is not None and old_eval_cfg is not None:
        rows.append(_metrics_row("test", "old_pipeline_eval", old_eval_cfg, old_test_metrics))

    df = pd.DataFrame(rows)
    if save_outputs:
        suffix = "" if checkpoint_label == "best" else f"_{checkpoint_label}"
        csv_path = os.path.join(results_dir, f"reeval_metrics{suffix}.csv")
        df.to_csv(csv_path, index=False)

        leaderboard_sorted = sorted(
            leaderboard, key=lambda r: r["val_md_plastic_score"], reverse=True,
        )[:15]
        lb_path = os.path.join(results_dir, f"reeval_tune_leaderboard{suffix}.json")
        with open(lb_path, "w", encoding="utf-8") as f:
            json.dump(leaderboard_sorted, f, indent=2)

    report = {
        "checkpoint": os.path.abspath(checkpoint_path),
        "checkpoint_label": checkpoint_label,
        "train_epoch": train_epoch,
        "model_key": model_key,
        "variant": variant,
        "two_head": two_head,
        "fast_tune": fast_tune,
        "training_thr": training_thr,
        "train_summary": train_summary,
        "benchmark_targets": BENCHMARK_TARGETS,
        "configs": {
            "training_baseline": baseline_cfg,
            "tuned_best": tuned_cfg,
            "old_pipeline_eval": old_eval_cfg,
        },
        "results": {
            "val_training_baseline": _metrics_row("val", "training_baseline", baseline_cfg, val_baseline),
            "test_training_baseline": _metrics_row("test", "training_baseline", baseline_cfg, test_baseline),
            "val_tuned": _metrics_row("val", "tuned", tuned_cfg, val_tuned),
            "test_tuned": _metrics_row("test", "tuned", tuned_cfg, test_tuned),
        },
        "benchmark_assessment": {
            "test_tuned": {
                "plastic_IoU": _benchmark_gap(test_tuned["plastic_IoU"], "plastic_IoU"),
                "marida_md_IoU": _benchmark_gap(test_tuned["marida_md_IoU"], "marida_md_IoU"),
                "mIoU_foreground": _benchmark_gap(test_tuned["mIoU_foreground"], "mIoU_foreground"),
            },
            "test_training_baseline": {
                "plastic_IoU": _benchmark_gap(test_baseline["plastic_IoU"], "plastic_IoU"),
                "marida_md_IoU": _benchmark_gap(test_baseline["marida_md_IoU"], "marida_md_IoU"),
                "mIoU_foreground": _benchmark_gap(test_baseline["mIoU_foreground"], "mIoU_foreground"),
            },
        },
        "duration_sec": sec_since(t0),
        "duration_fmt": fmt_duration(sec_since(t0)),
        "recommendation": (
            "Thesis table: use test_training_baseline @ thr=0.40. "
            "Do not use val-tuned thr=0.30 on test. Ignore old_pipeline_eval."
        ),
    }
    if old_test_metrics is not None:
        report["results"]["test_old_pipeline"] = _metrics_row(
            "test", "old_pipeline", old_eval_cfg, old_test_metrics,
        )

    if save_outputs:
        suffix = "" if checkpoint_label == "best" else f"_{checkpoint_label}"
        report_path = os.path.join(results_dir, f"reeval_report{suffix}.json")
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        if checkpoint_label == "best":
            tuned_cfg_path = os.path.join(results_dir, "reeval_config_best.json")
            with open(tuned_cfg_path, "w", encoding="utf-8") as f:
                json.dump(tuned_cfg, f, indent=2)

            baseline_cfg_path = os.path.join(results_dir, "reeval_config_baseline.json")
            with open(baseline_cfg_path, "w", encoding="utf-8") as f:
                json.dump(baseline_cfg, f, indent=2)

        print(f"\n[reeval] Saved: {report_path}")
        print(f"[reeval] Saved: {csv_path}")
        if checkpoint_label == "best":
            print(f"[reeval] Saved: {tuned_cfg_path}")
        print(f"[reeval] Duration: {report['duration_fmt']}")
        print("\n[reeval] Benchmark gap (TEST @ thr=0.40 baseline):")
        for k, v in report["benchmark_assessment"]["test_training_baseline"].items():
            print(f"  {k}: {v}")

    return report


def reevaluate_all(
    marida_path: str,
    project_root: str,
    *,
    model_key: str = "taunet_resnet50",
    variant: str = "no_aug",
    results_dir: str | None = None,
    compare_baseline_dir: str | None = None,
    num_workers: int = 0,
    pin_memory: bool = False,
    fast_tune: bool = True,
    training_thr: float = 0.40,
):
    if results_dir is None:
        results_dir = os.path.join(project_root, "results", model_key, variant)
    train_summary = _load_train_summary(results_dir)
    checkpoints = _discover_checkpoints(
        project_root, model_key, variant, train_summary, None,
    )
    if not checkpoints:
        raise FileNotFoundError(
            f"No checkpoints under saved_models/{variant}/ for {model_key}",
        )

    baseline = None
    if compare_baseline_dir:
        baseline_path = compare_baseline_dir
        if not os.path.isabs(baseline_path):
            baseline_path = os.path.join(project_root, compare_baseline_dir)
        baseline = _load_baseline_reeval(baseline_path)
        if baseline:
            print(f"[reeval] Loaded baseline from {baseline_path}")

    print(f"[reeval] Evaluating {len(checkpoints)} checkpoint(s): "
          f"{', '.join(l for l, _, _ in checkpoints)}")

    reports = []
    all_rows = []
    for label, ckpt_path, epoch in checkpoints:
        print(f"\n{'#' * 72}\n# CHECKPOINT: {label} (epoch {epoch})\n{'#' * 72}")
        rep = reevaluate(
            marida_path,
            project_root,
            model_key=model_key,
            variant=variant,
            checkpoint_path=ckpt_path,
            checkpoint_label=label,
            train_epoch=epoch,
            results_dir=results_dir,
            num_workers=num_workers,
            pin_memory=pin_memory,
            fast_tune=fast_tune,
            training_thr=training_thr,
            save_outputs=True,
        )
        reports.append(rep)
        for key, row in rep["results"].items():
            if key.startswith("test_"):
                all_rows.append({
                    "checkpoint": label,
                    "train_epoch": epoch,
                    "eval_split": key,
                    **{k: row[k] for k in (
                        "plastic_IoU", "marida_md_IoU", "mIoU_foreground",
                        "plastic_precision", "plastic_recall", "md_plastic_score",
                        "debris_threshold",
                    )},
                })

    _print_checkpoint_comparison(reports, baseline)

    summary_path = os.path.join(results_dir, "reeval_metrics_all_checkpoints.csv")
    pd.DataFrame(all_rows).to_csv(summary_path, index=False)

    combined = {
        "checkpoints": reports,
        "baseline_results_dir": compare_baseline_dir,
        "summary_csv": summary_path,
        "best_for_plastic": max(
            reports,
            key=lambda r: r["results"]["test_training_baseline"]["plastic_IoU"],
        )["checkpoint_label"],
        "best_for_md": max(
            reports,
            key=lambda r: r["results"]["test_training_baseline"]["marida_md_IoU"],
        )["checkpoint_label"],
    }
    combined_path = os.path.join(results_dir, "reeval_report_combined.json")
    with open(combined_path, "w", encoding="utf-8") as f:
        json.dump(combined, f, indent=2, default=str)

    print(f"\n[reeval] Saved combined: {combined_path}")
    print(f"[reeval] Saved summary:  {summary_path}")
    print(
        f"[reeval] Best test plastic @ thr={training_thr}: "
        f"{combined['best_for_plastic']}"
    )
    print(
        f"[reeval] Best test MD @ thr={training_thr}: "
        f"{combined['best_for_md']}"
    )
    return combined


def main():
    parser = argparse.ArgumentParser(description="Re-evaluate checkpoint with MD+plastic tuning")
    parser.add_argument("--marida", required=True, help="MARIDA dataset root")
    parser.add_argument("--root", default=".", help="Project root (Disertatie_2)")
    parser.add_argument("--model", default="taunet_resnet50", dest="model_key")
    parser.add_argument("--variant", default="no_aug")
    parser.add_argument("--checkpoint", default=None, help="Path to *_best.pth")
    parser.add_argument("--results-dir", default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--pin-memory", action="store_true", help="DataLoader pin_memory")
    parser.add_argument(
        "--training-thr", type=float, default=0.40,
        help="Debris threshold matching training val decode (history24: 0.40)",
    )
    parser.add_argument(
        "--full-tune", action="store_true",
        help="Larger grid: TTA scales + CRF options (slower)",
    )
    parser.add_argument(
        "--all-checkpoints", action="store_true",
        help="Reeval _best, _best_plastic, _best_md sidecars from train_summary",
    )
    parser.add_argument(
        "--compare-baseline", default="results_3/taunet_resnet50/no_aug",
        help="Results dir with baseline reeval_metrics.csv (default: results_3)",
    )
    parser.add_argument(
        "--no-baseline-compare", action="store_true",
        help="Skip loading results_3 baseline comparison table",
    )
    args = parser.parse_args()

    project_root = os.path.abspath(args.root)
    results_dir = args.results_dir
    if results_dir is None and args.checkpoint is None:
        results_dir = os.path.join(
            project_root, "results", args.model_key.lower(), args.variant,
        )
    marida_path = os.path.abspath(args.marida)
    common_kw = dict(
        model_key=args.model_key.lower(),
        variant=args.variant,
        results_dir=results_dir,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        fast_tune=not args.full_tune,
        training_thr=args.training_thr,
    )
    if args.all_checkpoints and args.checkpoint is not None:
        parser.error("Use either --all-checkpoints or --checkpoint, not both.")
    if args.all_checkpoints:
        reevaluate_all(
            marida_path,
            project_root,
            compare_baseline_dir=None if args.no_baseline_compare else args.compare_baseline,
            **common_kw,
        )
    else:
        reevaluate(
            marida_path,
            project_root,
            checkpoint_path=args.checkpoint,
            **common_kw,
        )


if __name__ == "__main__":
    main()
