"""Cross-correlation utilities."""

import numpy as np
import numpy.typing as npt

__all__ = ["cross_correlation", "normalized_cross_correlation"]


def _validate_signal(samples: np.ndarray, name: str) -> np.ndarray:
    """Return a validated, one-dimensional finite numeric signal."""
    samples = np.asarray(samples)
    if samples.ndim != 1:
        raise ValueError(
            f"{name} must be one-dimensional, got shape {samples.shape}."
        )
    if samples.size == 0:
        raise ValueError(f"{name} must contain at least one value.")
    if samples.dtype.kind not in "fiuc":
        raise ValueError(
            f"{name} must contain numeric values, got dtype {samples.dtype}."
        )
    if not np.all(np.isfinite(samples)):
        raise ValueError(f"{name} must contain only finite values.")
    return samples


def _correlation_dtype(first: np.ndarray, second: np.ndarray) -> npt.DTypeLike:
    """Use at least double precision and preserve complex correlation."""
    return np.result_type(
        first.dtype,
        second.dtype,
        np.complex128
        if (np.iscomplexobj(first) or np.iscomplexobj(second))
        else np.float64,
    )


def cross_correlation(
    first: np.ndarray, second: np.ndarray
) -> tuple[npt.NDArray[np.int64], np.ndarray]:
    """Compute the full discrete cross-correlation of two finite signals.

    The returned values implement the convention

    ``r_xy[lag] = sum_n x[n + lag] * conj(y[n])``,

    where ``x`` is ``first``, ``y`` is ``second``, and terms whose indices
    fall outside either finite input are omitted. The returned lag array
    runs from ``-(len(second) - 1)`` through ``len(first) - 1`` and is
    aligned with the correlation values. Under this convention, if
    ``first`` is a delayed copy of ``second`` by ``d`` samples, their
    correlation peak occurs at lag ``+d``.

    Args:
        first: 1-D real or complex finite signal ``x``.
        second: 1-D real or complex finite signal ``y``.

    Returns:
        Integer lags and full correlation values. Real inputs return
        float64 values; either complex input returns complex128 values.

    Raises:
        ValueError: If either input is empty, non-numeric, non-finite, or
            not one-dimensional.
    """
    first = _validate_signal(first, "first")
    second = _validate_signal(second, "second")
    dtype = _correlation_dtype(first, second)
    first = first.astype(dtype, copy=False)
    second = second.astype(dtype, copy=False)
    lags = np.arange(-(second.size - 1), first.size, dtype=np.int64)
    return lags, np.correlate(first, second, "full")


def _overlap_energies(
    first: np.ndarray, second: np.ndarray, lags: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return the energies of the paired sample regions at every lag."""
    first_energy = np.concatenate(([0.0], np.cumsum(np.abs(first) ** 2)))
    second_energy = np.concatenate(([0.0], np.cumsum(np.abs(second) ** 2)))
    first_start = np.maximum(lags, 0)
    second_start = np.maximum(-lags, 0)
    overlap = np.minimum(first.size - first_start, second.size - second_start)
    first_overlap = (
        first_energy[first_start + overlap] - first_energy[first_start]
    )
    second_overlap = (
        second_energy[second_start + overlap] - second_energy[second_start]
    )
    return first_overlap, second_overlap


def normalized_cross_correlation(
    first: np.ndarray, second: np.ndarray
) -> tuple[npt.NDArray[np.int64], np.ndarray]:
    """Compute overlap-energy-normalized full cross-correlation.

    The unnormalized convention is
    ``r_xy[lag] = sum_n x[n + lag] * conj(y[n])``. Each output is
    normalized by the energy in exactly the samples participating at that
    lag, so the normalization uses exact-overlap energies rather than one
    global energy:

    ``rho_xy[lag] = r_xy[lag] / sqrt(E_x(lag) * E_y(lag))``.

    Every defined value therefore has magnitude at most one, and equal
    overlapping segments have magnitude one. If either *entire input* has
    zero energy, normalization is undefined and raises ``ValueError``.
    With nonzero whole-input energy, an overlap can still have zero
    energy only when it contains zero-valued samples; such undefined lag
    values are returned as zero rather than NaN. Lag ordering and the
    delayed-copy ``+d`` interpretation are the same as
    :func:`cross_correlation`.

    Args:
        first: 1-D real or complex finite signal ``x``.
        second: 1-D real or complex finite signal ``y``.

    Returns:
        Integer lags and normalized full correlation values. Real inputs
        return float64 values; either complex input returns complex128
        values.

    Raises:
        ValueError: If an input is invalid or either input has zero
            energy.
    """
    first = _validate_signal(first, "first")
    second = _validate_signal(second, "second")
    dtype = _correlation_dtype(first, second)
    first = first.astype(dtype, copy=False)
    second = second.astype(dtype, copy=False)
    if float(np.sum(np.abs(first) ** 2)) == 0.0:
        raise ValueError("first must have nonzero energy for normalization.")
    if float(np.sum(np.abs(second) ** 2)) == 0.0:
        raise ValueError("second must have nonzero energy for normalization.")

    lags, correlation = cross_correlation(first, second)
    first_overlap, second_overlap = _overlap_energies(first, second, lags)
    denominator = np.sqrt(first_overlap * second_overlap)
    normalized = np.zeros_like(correlation)
    np.divide(correlation, denominator, out=normalized, where=denominator > 0.0)
    return lags, normalized
