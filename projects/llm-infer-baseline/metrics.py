"""Utility helpers for timing inference steps."""
from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Dict


@contextmanager
def timed(name: str, stats: Dict[str, float]):
    """Context manager to time blocks and store duration under `name`."""
    start = time.perf_counter()
    yield
    stats[name] = time.perf_counter() - start


def log_stats(stats: Dict[str, float]) -> str:
    """Return a printable string for collected timing stats."""
    lines = ["=== STATS (seconds) ==="]
    for key, value in stats.items():
        lines.append(f"{key:20s}: {value:.4f}")
    return "\n".join(lines)
