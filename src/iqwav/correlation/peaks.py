"""Correlation peak-detection utilities."""

import math
from numbers import Integral, Real

import numpy as np
import numpy.typing as npt
from scipy.signal import find_peaks

from .cross_correlation import _validate_signal

__all__ = ["find_correlation_peaks"]


def find_correlation_peaks(
    correlation: np.ndarray,
    lags: np.ndarray,
    *,
    min_height: float | None = None,
    min_distance: int = 1,
    prominence: float | None = None,
    use_magnitude: bool = True,
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64], np.ndarray]:
    """Find local extrema that are correlation peaks.

    ``correlation`` and ``lags`` must be one-dimensional, equally sized,
    finite arrays. A returned peak index indexes both inputs, so peak
    lags follow the caller's explicit lag convention. By default local
    maxima are found in ``abs(correlation)``, which is appropriate for
    complex correlation and for negative real matches. Set
    ``use_magnitude=False`` to detect only positive maxima of real-valued
    correlation. ``min_height`` and ``prominence`` apply to the selected
    real-valued search sequence, while ``min_distance`` is a minimum
    separation in array indices, as defined by SciPy.

    Args:
        correlation: 1-D real or complex finite correlation values.
        lags: 1-D finite integer lag values aligned with ``correlation``.
        min_height: Optional nonnegative finite minimum peak height.
        min_distance: Positive integer minimum separation between peaks.
        prominence: Optional nonnegative finite minimum prominence.
        use_magnitude: Whether to search magnitude instead of signed
            values.

    Returns:
        Peak indices, corresponding lags, and the original (possibly
        complex) correlation values at the peaks.

    Raises:
        ValueError: If arrays or peak-selection arguments are invalid,
            or signed peak detection is requested for complex
            correlation.
    """
    correlation = _validate_signal(correlation, "correlation")
    lags = np.asarray(lags)
    if lags.ndim != 1:
        raise ValueError(f"lags must be one-dimensional, got shape {lags.shape}.")
    if lags.size != correlation.size:
        raise ValueError("lags must have the same length as correlation.")
    if not np.issubdtype(lags.dtype, np.integer):
        raise ValueError("lags must contain integer values.")
    if not isinstance(min_distance, Integral) or isinstance(min_distance, bool):
        raise ValueError("min_distance must be a positive integer.")
    if min_distance < 1:
        raise ValueError("min_distance must be a positive integer.")
    for name, value in (("min_height", min_height), ("prominence", prominence)):
        if value is not None and (
            not isinstance(value, Real)
            or isinstance(value, bool)
            or not math.isfinite(value)
            or value < 0.0
        ):
            raise ValueError(f"{name} must be a nonnegative finite value or None.")
    if not isinstance(use_magnitude, bool):
        raise ValueError("use_magnitude must be a bool.")
    if not use_magnitude and np.iscomplexobj(correlation):
        raise ValueError("signed peak detection requires real-valued correlation.")

    search_values = np.abs(correlation) if use_magnitude else correlation
    indices, _ = find_peaks(
        search_values,
        height=min_height,
        distance=int(min_distance),
        prominence=prominence,
    )
    return indices.astype(np.int64), lags[indices].astype(np.int64), correlation[indices]
