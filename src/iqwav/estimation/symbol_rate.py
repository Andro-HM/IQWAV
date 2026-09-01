"""Blind symbol-rate estimation baseline for rectangular-pulse PSK."""

from dataclasses import dataclass

import numpy as np

from ..correlation import normalized_autocorrelation
from .occupied_band import _validate_real_scalar, _validate_samples

__all__ = ["SymbolRateEstimate", "estimate_symbol_rate"]


@dataclass(frozen=True)
class SymbolRateEstimate:
    """Blind symbol-rate estimate for a rectangular-pulse PSK-like signal.

    ``samples_per_symbol`` is the selected integer symbol spacing in
    samples, ``symbol_rate_hz`` is ``fs / samples_per_symbol``, and
    ``score`` is the normalized transition-energy autocorrelation value
    at the selected lag.
    """

    symbol_rate_hz: float
    samples_per_symbol: int
    score: float


def _validate_lag_int(value: object, name: str, minimum: int) -> None:
    """Validate a non-bool integer option with a minimum."""
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, np.integer))
        or value < minimum
    ):
        raise ValueError(f"{name} must be an integer >= {minimum}, got {value!r}.")


def estimate_symbol_rate(
    samples: np.ndarray,
    fs: float,
    *,
    min_sps: int = 2,
    max_sps: int = 64,
    min_score: float = 0.10,
) -> SymbolRateEstimate:
    """Estimate the symbol rate of a rectangular-pulse PSK-like IQ signal.

    Baseline algorithm: compute the adjacent-sample transition energy
    ``e[n] = |x[n+1] - x[n]|^2``, remove its mean, and feed the centered
    transition energy into the production
    :func:`iqwav.correlation.normalized_autocorrelation`. A rectangular
    symbol grid produces recurring autocorrelation peaks at multiples of
    the samples-per-symbol value ``SPS``, so the estimator scans candidate
    lags ``min_sps .. max_sps`` in increasing order and returns the
    SMALLEST local maximum whose normalized autocorrelation value is at
    least ``min_score``. The smallest qualifying peak is preferred over
    the globally strongest peak because a harmonic such as ``2*SPS`` or
    ``3*SPS`` can score as high as the fundamental.

    For ideal BPSK/QPSK the symbol magnitude is approximately constant,
    so ``|x[n]|`` alone carries little symbol-boundary information; the
    adjacent-difference transition energy exposes the boundaries instead.
    Equal consecutive symbols drop their boundary impulse, but long
    random sequences still preserve the periodic boundary grid.

    This is deliberately a bounded baseline, not general blind baud
    estimation. It assumes approximately BPSK/QPSK-like symbols with a
    rectangular (sample-and-hold) pulse shape, integer samples per
    symbol, enough random symbol transitions, a known sample rate, a
    reasonably stationary signal, and moderate SNR. It does NOT yet
    support RRC or other pulse shaping, fractional samples per symbol,
    severe CFO (which introduces within-symbol sample differences that
    are not corrected here), timing drift, or arbitrary modulations, and
    it performs no timing synchronization, matched filtering, or carrier
    recovery. Strong noise can hide the transition periodicity by
    dominating the transition energy.

    Args:
        samples: 1-D real or complex finite samples with at least 2
            values and at least ``max_sps + 2`` samples.
        fs: Positive finite real sampling rate in Hz. Not inferred.
        min_sps: Smallest candidate samples-per-symbol lag. Must be a
            non-bool integer >= 2.
        max_sps: Largest candidate samples-per-symbol lag. Must be a
            non-bool integer >= ``min_sps`` and <= ``len(samples) - 2``.
        min_score: Minimum normalized transition-autocorrelation value
            for an accepted candidate. Must satisfy
            ``0 < min_score <= 1``.

    Returns:
        The :class:`SymbolRateEstimate` with the selected lag, its
        symbol rate ``fs / samples_per_symbol``, and its score.

    Raises:
        ValueError: If any argument is invalid, the waveform has no
            usable transitions (e.g. a constant waveform), or no
            candidate lag reaches ``min_score``.
    """
    samples = _validate_samples(samples)
    fs = _validate_real_scalar(fs, "fs")
    if fs <= 0:
        raise ValueError(f"fs must be positive and finite, got {fs!r}.")
    _validate_lag_int(min_sps, "min_sps", 2)
    _validate_lag_int(max_sps, "max_sps", min_sps)
    if max_sps > samples.shape[0] - 2:
        raise ValueError(
            f"max_sps must be <= len(samples) - 2 = {samples.shape[0] - 2} "
            f"for the requested lag range, got {max_sps!r}."
        )
    score_min = _validate_real_scalar(min_score, "min_score")
    if not 0.0 < score_min <= 1.0:
        raise ValueError(
            f"min_score must satisfy 0 < min_score <= 1, got {min_score!r}."
        )

    transitions = np.abs(samples[1:] - samples[:-1]) ** 2
    centered = transitions - np.mean(transitions)
    if np.all(centered == 0.0):
        raise ValueError(
            "Transition energy has no variance; the waveform appears "
            "constant with no usable symbol transitions."
        )
    correlation = normalized_autocorrelation(centered, max_lag=max_sps)
    for lag in range(min_sps, max_sps + 1):
        value = float(correlation[lag])
        if value < score_min:
            continue
        is_local_peak = value >= correlation[lag - 1] and (
            lag == max_sps or value >= correlation[lag + 1]
        )
        if is_local_peak:
            return SymbolRateEstimate(
                symbol_rate_hz=fs / lag,
                samples_per_symbol=int(lag),
                score=value,
            )
    raise ValueError(
        f"No candidate lag in [{min_sps}, {max_sps}] reached a normalized "
        f"transition-autocorrelation score of at least {score_min!r}; no "
        f"reliable symbol-rate estimate is available under this baseline "
        f"method."
    )
