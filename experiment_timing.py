"""Shared timing helpers for training and experiment pipelines."""

from __future__ import annotations

import time


def sec_since(t0: float) -> float:
    return round(time.perf_counter() - t0, 2)


def fmt_duration(sec: float | None) -> str:
    """Always show raw seconds plus human-readable m/s or h/m/s."""
    if sec is None:
        return "—"
    sec = float(sec)
    if sec < 60:
        return f"{sec:.1f}s"
    m = int(sec // 60)
    s = int(round(sec % 60))
    if sec < 3600:
        return f"{sec:.1f}s ({m}m {s}s)"
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = int(round(sec % 60))
    return f"{sec:.1f}s ({h}h {m}m {s}s)"


def format_timing_dict(steps: dict) -> dict:
    """Attach human-readable *_fmt fields for every *_sec value."""
    out = dict(steps)
    for key, val in list(steps.items()):
        if key.endswith("_sec") and isinstance(val, (int, float)):
            out[f"{key[:-4]}_fmt"] = fmt_duration(val)
    if "total_sec" in steps and isinstance(steps["total_sec"], (int, float)):
        out["total_fmt"] = fmt_duration(steps["total_sec"])
    return out


def print_timing_block(title: str, steps: dict, *, total_key: str = "total_sec") -> None:
    print(f"\n[timing] {title}")
    total = steps.get(total_key)
    for key, val in steps.items():
        if key == total_key or key.endswith("_fmt"):
            continue
        if not key.endswith("_sec"):
            continue
        label = key.replace("_sec", "").replace("_", " ")
        print(f"  {label}: {fmt_duration(val)}")
    if total is not None:
        print(f"  total: {fmt_duration(total)}")
