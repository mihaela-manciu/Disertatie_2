"""Shared timing helpers for training and experiment pipelines."""

from __future__ import annotations

import time


def sec_since(t0: float) -> float:
    return round(time.perf_counter() - t0, 2)


def fmt_duration(sec: float | None) -> str:
    if sec is None:
        return "—"
    sec = float(sec)
    if sec < 60:
        return f"{sec:.1f}s"
    if sec < 3600:
        m = int(sec // 60)
        s = int(sec % 60)
        return f"{m}m {s}s"
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    return f"{h}h {m}m"


def print_timing_block(title: str, steps: dict, *, total_key: str = "total_sec") -> None:
    print(f"\n[timing] {title}")
    total = steps.get(total_key)
    for key, val in steps.items():
        if key == total_key:
            continue
        label = key.replace("_sec", "").replace("_", " ")
        print(f"  {label}: {fmt_duration(val)}")
    if total is not None:
        print(f"  total: {fmt_duration(total)}")
