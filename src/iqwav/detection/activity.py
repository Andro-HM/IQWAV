"""Windowed time-domain activity hints for IQ captures."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

__all__ = ["ActivityDetection", "ActivityRegion", "detect_activity"]


@dataclass(frozen=True)
class ActivityRegion:
    start_sample: int
    end_sample: int
    average_power: float
    peak_power: float
    start_time_s: float | None
    end_time_s: float | None
    duration_s: float | None


@dataclass(frozen=True)
class ActivityDetection:
    regions: tuple[ActivityRegion, ...]
    baseline_power: float
    threshold_power: float
    window_size: int
    relative_threshold_db: float


def _validate(samples: np.ndarray, window_size: int, relative_threshold_db: float, merge_gap_samples: int, background_percentile: float) -> np.ndarray:
    x = np.asarray(samples)
    if x.ndim != 1 or x.size == 0 or not np.all(np.isfinite(x)):
        raise ValueError("samples must be a non-empty one-dimensional finite array.")
    if isinstance(window_size, bool) or not isinstance(window_size, (int, np.integer)) or window_size < 1:
        raise ValueError("window_size must be an integer >= 1.")
    if not math.isfinite(relative_threshold_db):
        raise ValueError("relative_threshold_db must be finite.")
    if isinstance(merge_gap_samples, bool) or not isinstance(merge_gap_samples, (int, np.integer)) or merge_gap_samples < 0:
        raise ValueError("merge_gap_samples must be an integer >= 0.")
    if not math.isfinite(background_percentile) or not 0.0 < background_percentile <= 100.0:
        raise ValueError("background_percentile must lie in (0, 100].")
    return x


def detect_activity(samples: np.ndarray, *, fs: float | None = None, window_size: int = 256, relative_threshold_db: float = 6.0, merge_gap_samples: int = 0, background_percentile: float = 20.0) -> ActivityDetection:
    """Return power-thresholded time regions, using a baseline-power estimate.

    The baseline comes from lower-power windows and can be contaminated by a
    high-occupancy or continuous signal. In particular, constant-power input
    can produce no regions; callers must not use this result to suppress a
    spectral survey.
    """
    x = _validate(samples, window_size, relative_threshold_db, merge_gap_samples, background_percentile)
    if fs is not None and (not math.isfinite(fs) or fs <= 0):
        raise ValueError("fs must be positive and finite when supplied.")
    powers = np.abs(x) ** 2
    bounds = [(start, min(start + window_size, x.size)) for start in range(0, x.size, window_size)]
    local = np.array([float(np.mean(powers[start:end])) for start, end in bounds])
    cutoff = float(np.percentile(local, background_percentile))
    baseline = float(np.mean(local[local <= cutoff]))
    threshold = baseline * 10.0 ** (relative_threshold_db / 10.0)
    active = local > threshold
    runs: list[tuple[int, int]] = []
    start = None
    for index, flag in enumerate(active):
        if flag and start is None:
            start = index
        elif not flag and start is not None:
            runs.append((start, index)); start = None
    if start is not None:
        runs.append((start, len(bounds)))
    merged: list[tuple[int, int]] = []
    for first, last in runs:
        sample_run = (bounds[first][0], bounds[last - 1][1])
        if merged and sample_run[0] - merged[-1][1] <= merge_gap_samples:
            merged[-1] = (merged[-1][0], sample_run[1])
        else:
            merged.append(sample_run)
    regions = []
    for start, end in merged:
        start_t = start / fs if fs is not None else None
        end_t = end / fs if fs is not None else None
        regions.append(ActivityRegion(start, end, float(np.mean(powers[start:end])), float(np.max(powers[start:end])), start_t, end_t, (end - start) / fs if fs is not None else None))
    return ActivityDetection(tuple(regions), baseline, threshold, int(window_size), float(relative_threshold_db))
