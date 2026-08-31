"""Autocorrelation primitives."""

import numpy as np
import numpy.typing as npt

__all__ = ["autocorrelation", "normalized_autocorrelation"]


def _validate_samples(samples: np.ndarray) -> np.ndarray:
    """Validate 1-D finite real or complex numeric samples."""
    samples = np.asarray(samples)
    if samples.ndim != 1:
        raise ValueError(
            f"samples must be one-dimensional, got shape {samples.shape}."
        )
    if samples.size == 0:
        raise ValueError("samples must contain at least one value.")
    if samples.dtype.kind not in "fiuc":
        raise ValueError(
            f"samples must be real or complex numeric data, "
            f"got dtype {samples.dtype}."
        )
    if not np.all(np.isfinite(samples)):
        raise ValueError("samples must contain only finite values.")
    return samples


def _validate_max_lag(max_lag: int | None, n: int) -> int:
    """Validate and resolve the maximum lag."""
    if max_lag is None:
        return n - 1
    if isinstance(max_lag, bool) or not isinstance(max_lag, (int, np.integer)):
        raise ValueError(f"max_lag must be an integer or None, got {max_lag!r}.")
    if not 0 <= max_lag < n:
        raise ValueError(
            f"max_lag must satisfy 0 <= max_lag < {n}, got {max_lag!r}."
        )
    return int(max_lag)


def autocorrelation(
    samples: np.ndarray,
    max_lag: int | None = None,
) -> npt.NDArray[np.float64] | npt.NDArray[np.complex128]:
    """Compute the non-negative-lag autocorrelation of a 1-D signal.

    For each lag ``k`` this computes

    ``R[k] = (1 / (N - k)) * sum(x[n + k] * conj(x[n]))``

    using overlap normalization ``1/(N - k)``, so large lags are not
    reduced merely because fewer samples overlap. No mean removal,
    detrending, or FFT acceleration is applied, and the output index
    equals the lag directly.

    Args:
        samples: 1-D real or complex sample array. Must be non-empty,
            numeric, and contain only finite values.
        max_lag: Highest lag to compute. If None, ``N - 1``. Otherwise
            must be a non-bool integer satisfying ``0 <= max_lag < N``.

    Returns:
        A 1-D array of length ``max_lag + 1`` where entry ``k`` is the
        autocorrelation at lag ``k``. Real input yields float64, complex
        input yields complex128.

    Raises:
        ValueError: If ``samples`` or ``max_lag`` is invalid.
    """
    samples = _validate_samples(samples)
    n = samples.shape[0]
    max_lag = _validate_max_lag(max_lag, n)
    if np.iscomplexobj(samples):
        result = np.empty(max_lag + 1, dtype=np.complex128)
    else:
        result = np.empty(max_lag + 1, dtype=np.float64)
    for k in range(max_lag + 1):
        result[k] = np.vdot(samples[: n - k], samples[k:]) / (n - k)
    return result


def normalized_autocorrelation(
    samples: np.ndarray,
    max_lag: int | None = None,
) -> npt.NDArray[np.float64] | npt.NDArray[np.complex128]:
    """Compute the autocorrelation normalized by its zero-lag value.

    Returns ``autocorrelation(samples, max_lag) / R[0]`` so that the
    zero-lag entry equals 1. The input conventions of
    :func:`autocorrelation` apply unchanged.

    Args:
        samples: 1-D real or complex sample array as in
            :func:`autocorrelation`. Must have non-zero energy.
        max_lag: Highest lag to compute, as in :func:`autocorrelation`.

    Returns:
        A 1-D array of length ``max_lag + 1`` with the normalized
        autocorrelation; entry 0 equals 1. Real input yields float64,
        complex input yields complex128.

    Raises:
        ValueError: If ``samples`` or ``max_lag`` is invalid, or the
            input has zero energy (``R[0] == 0``).
    """
    raw = autocorrelation(samples, max_lag=max_lag)
    r_zero = raw[0]
    if r_zero == 0:
        raise ValueError(
            "Cannot normalize autocorrelation: input has zero energy "
            "(R[0] == 0)."
        )
    return raw / r_zero
