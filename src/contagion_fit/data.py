"""Load the unified cascade table and compute complementary CDFs.

The table ``data/processed/cascades_unified.csv`` is produced by
``scripts/preprocess.py`` and has columns: platform, cascade_id, label, size.
This module never simulates anything; it only reads observed sizes and turns
them into the log-log CCDF representation used everywhere downstream.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

from contagion_fit.config import CASCADES_CSV


def load_cascades(
    platform: str | None = None,
    label: str | None = None,
    *,
    path: Path = CASCADES_CSV,
    min_size: int = 1,
) -> np.ndarray:
    """Return observed cascade sizes as a 1-D array.

    Parameters
    ----------
    platform, label
        Optional filters. ``None`` means "do not filter on this column".
    min_size
        Drop cascades smaller than this (default keeps everything >= 1).
    """
    sizes: list[int] = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if platform is not None and row["platform"] != platform:
                continue
            if label is not None and row["label"] != label:
                continue
            size = int(row["size"])
            if size >= min_size:
                sizes.append(size)
    return np.asarray(sizes, dtype=np.int64)


def available_groups(*, path: Path = CASCADES_CSV) -> dict[str, list[str]]:
    """Map each platform to the sorted list of labels present for it."""
    groups: dict[str, set[str]] = defaultdict(set)
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            groups[row["platform"]].add(row["label"])
    return {p: sorted(labels) for p, labels in sorted(groups.items())}


def ccdf(sizes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Empirical complementary CDF: ``P(X >= x)``.

    Returns ``(x, p)`` where ``x`` are the sorted unique sizes and ``p[i]`` is
    the fraction of observations ``>= x[i]``. Designed for log-log plotting, so
    sizes of zero are not expected (cascades have size >= 1).
    """
    sizes = np.asarray(sizes, dtype=np.float64)
    if sizes.size == 0:
        return np.empty(0), np.empty(0)
    x = np.sort(np.unique(sizes))
    # For each unique x, count observations >= x.
    sorted_sizes = np.sort(sizes)
    # searchsorted on the left gives the count strictly less than x.
    counts_lt = np.searchsorted(sorted_sizes, x, side="left")
    p = 1.0 - counts_lt / sizes.size
    return x, p


def summary(sizes: np.ndarray) -> dict[str, float]:
    """Compact descriptive statistics for a size sample."""
    if sizes.size == 0:
        return {"n": 0, "median": float("nan"), "mean": float("nan"), "max": float("nan")}
    return {
        "n": int(sizes.size),
        "median": float(np.median(sizes)),
        "mean": float(np.mean(sizes)),
        "max": int(np.max(sizes)),
    }
