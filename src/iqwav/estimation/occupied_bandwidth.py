"""Cumulative-power occupied-bandwidth measurement."""

import math
from dataclasses import dataclass

import numpy as np

from ..dsp import magnitude_spectrum
from .occupied_band import _validate_real_scalar

__all__ = ["OccupiedBandwidthEstimate", "estimate_occupied_bandwidth"]

_MIN_SAMPLES = 4
# Relative-to-signal threshold below which the AC (non-DC) content of a
# signal is treated as numerically zero, matching the convention used by
# iqwav.estimation.spectral_peak.
_CONSTANT_SIGNAL_RTOL = 1e-12
# Relative slack so that a target derived from summing the same array is
# always reachable despite floating-point round-off, in particular for
# power_fraction == 1.0.
_TARGET_RTOL = 1e-9


@dataclass(frozen=True)
class OccupiedBandwidthEstimate:
    """Result of a cumulative-power occupied-bandwidth measurement.

    Attributes:
        lower_hz: Lower edge, in Hz, of the narrowest contiguous interval
            of FFT bins whose accumulated power reaches the requested
            fraction of the total measured spectral power. For complex
            input a Nyquist-wrapping interval is reported with
            ``lower_hz > upper_hz``.
        upper_hz: Upper edge, in Hz, of that interval.
        center_hz: Midpoint of the interval. For a wrapping interval this
            is the circular midpoint canonicalized to the signed baseband
            range.
        bandwidth_hz: Width of the selected interval: the circular width
            (``bins * fs / N``) for complex input, and
            ``upper_hz - lower_hz`` for real input whose edges are
            clamped to ``[0, fs/2]``.
        requested_power_fraction: The user-requested power fraction.
        achieved_power_fraction: The actual selected-bin power divided by
            the total measured spectral power.
        wraps_nyquist: Whether the selected complex-input interval wraps
            across the Nyquist boundary. Always False for real input.
    """

    lower_hz: float
    upper_hz: float
    center_hz: float
    bandwidth_hz: float
    requested_power_fraction: float
    achieved_power_fraction: float
    wraps_nyquist: bool


def _validate_samples_for_bandwidth(samples: np.ndarray) -> np.ndarray:
    """Validate 1-D finite numeric samples with at least 4 values."""
    samples = np.asarray(samples)
    if samples.ndim != 1:
        raise ValueError(
            f"samples must be one-dimensional, got shape {samples.shape}."
        )
    if samples.dtype.kind not in "fiuc":
        raise ValueError(
            f"samples must be real or complex numeric data, "
            f"got dtype {samples.dtype}."
        )
    if samples.shape[0] < _MIN_SAMPLES:
        raise ValueError(
            f"samples must contain at least {_MIN_SAMPLES} values to "
            f"estimate an occupied bandwidth, got {samples.shape[0]}."
        )
    if not np.all(np.isfinite(samples)):
        raise ValueError("samples must contain only finite values.")
    return samples


def _reject_degenerate_signal(samples: np.ndarray) -> None:
    """Raise ValueError for zero-energy or constant DC-only signals."""
    energy = float(np.sum(np.abs(samples) ** 2))
    if energy == 0.0:
        raise ValueError(
            "samples have zero energy; an occupied bandwidth cannot be "
            "measured for a zero signal."
        )
    ac = samples - np.mean(samples)
    ac_level = float(np.max(np.abs(ac))) if ac.size else 0.0
    scale = max(1.0, float(np.max(np.abs(samples))))
    if ac_level <= _CONSTANT_SIGNAL_RTOL * scale:
        raise ValueError(
            "samples are constant (or all zero) and contain no oscillating "
            "component; a constant DC-only signal has no meaningful "
            "occupied bandwidth under the cumulative-power definition."
        )


def _smallest_window(power: np.ndarray, threshold: float) -> tuple[int, int]:
    """Return inclusive (left, right) bounds of the smallest contiguous window.

    Finds the shortest contiguous window of the non-negative ``power``
    array whose sum is at least ``threshold``, using the exact
    two-pointer minimum-subarray technique. Ties are broken by the
    leftmost window encountered during the scan. ``threshold`` must be
    reachable within the array.
    """
    n = power.shape[0]
    best_length = None
    best = (0, n - 1)
    running = 0.0
    left = 0
    for right in range(n):
        running += float(power[right])
        while left <= right and running >= threshold:
            length = right - left + 1
            if best_length is None or length < best_length:
                best_length = length
                best = (left, right)
            running -= float(power[left])
            left += 1
    return best


def _canonical_frequency(value: float, fs: float) -> float:
    """Canonicalize a frequency to the signed baseband range."""
    return ((value + fs / 2.0) % fs) - fs / 2.0


def estimate_occupied_bandwidth(
    samples: np.ndarray,
    fs: float,
    *,
    power_fraction: float = 0.99,
) -> OccupiedBandwidthEstimate:
    """Measure the occupied bandwidth as a cumulative-power FFT-bin interval.

    Definition:
        The occupied bandwidth is the NARROWEST contiguous frequency
        interval, expressed in whole FFT bins, whose accumulated bin power
        ``abs(FFT[k])**2`` contains at least ``power_fraction`` of the
        total measured spectral power in the analyzed block. The spectrum
        is computed via :func:`iqwav.dsp.magnitude_spectrum` with bin
        spacing ``fs / N``. This is a cumulative measured-power metric:
        it does NOT estimate or subtract a noise floor, and noise,
        interference, DC, and any other spectral energy all count. A
        99% result means 99% of the TOTAL MEASURED FFT POWER in this
        block, not necessarily 99% of signal-only power.

    Complex input (circular search):
        For complex IQ the discrete frequency axis is circular: the bins
        immediately below ``+fs/2`` and immediately above ``-fs/2`` are
        adjacent across the Nyquist boundary. The search therefore finds
        the shortest CYCLIC contiguous run of bins reaching the requested
        power (bins sorted to ascending frequency, the power array
        doubled, and an exact two-pointer minimum-window search with
        windows of at most N bins, allowed to cross the end/start
        boundary).

        A winning interval that does not cross Nyquist is reported with
        ``wraps_nyquist=False`` and ``lower_hz < upper_hz``. One that
        crosses Nyquist is reported with ``wraps_nyquist=True``,
        ``lower_hz > upper_hz``, interpreted as
        ``[lower_hz, +fs/2) U [-fs/2, upper_hz)``; ``center_hz`` is the
        circular midpoint canonicalized to the signed baseband range. If
        the full band is required (all N bins), the result is the full
        interval ``[-fs/2, +fs/2]`` with ``wraps_nyquist=False``.

    Real input (conjugate folding):
        The FFT of a real signal is conjugate-symmetric, so ``+f`` and
        ``-f`` bins are two views of one physical component. Real input
        is folded onto the non-negative physical-frequency axis (DC and,
        for even N, Nyquist counted once; other bins summed in pairs),
        the search runs on that axis only, and the returned edges are
        clamped to ``[0, fs/2]`` so ``wraps_nyquist`` is always False.
        Folding preserves the total power, so ``power_fraction`` means
        the same fraction of total power for real and complex input.

    Bin-edge convention:
        Each bin of width ``df = fs / N`` centered at ``f`` is treated as
        occupying ``[f - df/2, f + df/2)``. Reported edges are bin EDGES,
        not centers; for real input the physical edges are clamped to
        ``0 <= f <= fs/2``.

    This IS: a cumulative-power occupied-bandwidth measurement, FFT/bin
    based, known-fs. This is NOT: HM's noise-floor occupied-band detector
    (:func:`iqwav.estimation.detect_occupied_bands`), signal-presence
    detection, noise subtraction, SNR estimation, CFO estimation, carrier
    estimation, modulation recognition, or a standardized regulatory OBW
    unless the caller's definition matches this exact cumulative-power
    method.

    Args:
        samples: 1-D real or complex finite numeric samples with at least
            4 values, not constant and not zero-energy.
        fs: Positive finite real sampling rate in Hz. Not inferred.
        power_fraction: Fraction of total measured spectral power that
            the returned interval must contain. Must be a finite real
            scalar in ``(0, 1]``.

    Returns:
        The :class:`OccupiedBandwidthEstimate`. ``achieved_power_fraction``
        is at least ``requested_power_fraction`` subject only to
        floating-point tolerance.

    Raises:
        ValueError: If any argument is invalid, or ``samples`` is
            zero-energy or constant (including all-zero).
    """
    samples = _validate_samples_for_bandwidth(samples)
    fs = _validate_real_scalar(fs, "fs")
    if fs <= 0:
        raise ValueError(f"fs must be positive and finite, got {fs!r}.")
    fraction = _validate_real_scalar(power_fraction, "power_fraction")
    if not 0.0 < fraction <= 1.0:
        raise ValueError(
            f"power_fraction must satisfy 0 < power_fraction <= 1, "
            f"got {power_fraction!r}."
        )
    _reject_degenerate_signal(samples)

    n = samples.shape[0]
    resolution = fs / n
    freqs, magnitude = magnitude_spectrum(samples, fs=fs, fftshift=False)
    power = magnitude.astype(np.float64) ** 2
    total = float(np.sum(power))
    if total <= 0.0:
        raise ValueError(
            "samples have zero measured spectral power; an occupied "
            "bandwidth cannot be measured."
        )
    threshold = fraction * total - _TARGET_RTOL * max(1.0, total)

    if np.iscomplexobj(samples):
        order = np.argsort(freqs, kind="stable")
        freqs_sorted = freqs[order]
        power_sorted = power[order]
        doubled = np.concatenate((power_sorted, power_sorted))
        left, right = _smallest_window(doubled, threshold)
        length = right - left + 1
        if length >= n:
            window_sum = float(np.sum(power_sorted))
            return OccupiedBandwidthEstimate(
                lower_hz=-fs / 2.0,
                upper_hz=fs / 2.0,
                center_hz=0.0,
                bandwidth_hz=float(n * resolution),
                requested_power_fraction=float(fraction),
                achieved_power_fraction=window_sum / total,
                wraps_nyquist=False,
            )
        start_bin = left
        end_bin = right % n
        lower_hz = float(freqs_sorted[start_bin]) - resolution / 2.0
        upper_hz = float(freqs_sorted[end_bin]) + resolution / 2.0
        wraps = right >= n or lower_hz < -fs / 2.0
        if lower_hz < -fs / 2.0:
            lower_hz += fs
        bandwidth_hz = float(length * resolution)
        if wraps:
            center_hz = _canonical_frequency(
                lower_hz + bandwidth_hz / 2.0, fs
            )
        else:
            center_hz = (lower_hz + upper_hz) / 2.0
        window_sum = float(np.sum(doubled[left : right + 1]))
        return OccupiedBandwidthEstimate(
            lower_hz=lower_hz,
            upper_hz=upper_hz,
            center_hz=center_hz,
            bandwidth_hz=bandwidth_hz,
            requested_power_fraction=float(fraction),
            achieved_power_fraction=window_sum / total,
            wraps_nyquist=wraps,
        )

    half = n // 2
    folded = power[: half + 1].copy()
    for k in range(1, (n + 1) // 2):
        folded[k] += power[n - k]
    folded_freqs = freqs[: half + 1].copy()
    if n % 2 == 0:
        # numpy.fft.fftfreq assigns the Nyquist bin (index n // 2, for
        # even n) the negative value -fs/2; since Nyquist's sign is
        # inherently ambiguous, use +fs/2 to keep this axis strictly
        # ascending and non-negative.
        folded_freqs[half] = abs(folded_freqs[half])
    left, right = _smallest_window(folded, threshold)
    lower_hz = max(0.0, float(folded_freqs[left]) - resolution / 2.0)
    upper_hz = min(fs / 2.0, float(folded_freqs[right]) + resolution / 2.0)
    window_sum = float(np.sum(folded[left : right + 1]))
    return OccupiedBandwidthEstimate(
        lower_hz=lower_hz,
        upper_hz=upper_hz,
        center_hz=(lower_hz + upper_hz) / 2.0,
        bandwidth_hz=upper_hz - lower_hz,
        requested_power_fraction=float(fraction),
        achieved_power_fraction=window_sum / total,
        wraps_nyquist=False,
    )
