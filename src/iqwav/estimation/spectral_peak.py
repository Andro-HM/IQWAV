"""Dominant spectral peak frequency estimation."""

import math
from dataclasses import dataclass

import numpy as np

from ..dsp import magnitude_spectrum
from .occupied_band import _validate_real_scalar

__all__ = ["PeakFrequencyEstimate", "estimate_peak_frequency"]

_MIN_SAMPLES = 4
# Relative-to-signal threshold below which the AC (non-DC) content of a
# signal is treated as numerically zero, i.e. a constant/zero signal with
# no oscillating component to localize a frequency for.
_CONSTANT_SIGNAL_RTOL = 1e-12


@dataclass(frozen=True)
class PeakFrequencyEstimate:
    """Dominant spectral component estimate.

    Attributes:
        frequency_hz: The final frequency estimate in Hz. Equal to
            ``bin_frequency_hz`` when ``refined`` is False, or to the
            sub-bin-refined frequency when ``refined`` is True.
        bin_frequency_hz: The center frequency, in Hz, of the raw FFT bin
            with the largest magnitude, with no sub-bin refinement.
        resolution_hz: The raw FFT bin spacing ``fs / N`` in Hz. This is
            the width of one FFT bin, not an error bound.
        bin_index: Index of the selected bin into NumPy's standard
            (unshifted) FFT ordering, matching ``numpy.fft.fftfreq``.
        refined: Whether ``frequency_hz`` includes sub-bin parabolic
            interpolation (True) or is the raw bin center (False).
    """

    frequency_hz: float
    bin_frequency_hz: float
    resolution_hz: float
    bin_index: int
    refined: bool


def _validate_samples_for_peak(samples: np.ndarray) -> np.ndarray:
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
            f"estimate a spectral peak, got {samples.shape[0]}."
        )
    if not np.all(np.isfinite(samples)):
        raise ValueError("samples must contain only finite values.")
    return samples


def _reject_constant_signal(samples: np.ndarray) -> None:
    """Raise ValueError if samples have no oscillating (AC) content."""
    ac = samples - np.mean(samples)
    ac_level = float(np.max(np.abs(ac))) if ac.size else 0.0
    scale = max(1.0, float(np.max(np.abs(samples))))
    if ac_level <= _CONSTANT_SIGNAL_RTOL * scale:
        raise ValueError(
            "samples are constant (or all zero) and contain no oscillating "
            "component; a dominant frequency cannot be estimated for a "
            "signal with no spectral content away from DC."
        )


def _wrap_bin_to_freq(bin_pos: float, n: int, fs: float) -> float:
    """Map a possibly fractional FFT bin index to a signed frequency.

    Generalizes ``numpy.fft.fftfreq`` to non-integer bin positions by
    wrapping into the half-open interval ``(-n/2, n/2]`` of bins before
    scaling by ``fs / n``.
    """
    wrapped = ((bin_pos + n / 2.0) % n) - n / 2.0
    return wrapped * fs / n


def _parabolic_log_refine(magnitude: np.ndarray, peak_index: int) -> float:
    """Return a sub-bin offset via parabolic interpolation of log-magnitude.

    Fits a parabola through the log-magnitude of the peak bin and its two
    circular (FFT-wrapped) neighbors and returns the vertex offset in
    bins, bounded to ``[-0.5, 0.5]``. This is the standard three-point
    log-magnitude interpolator for FFT peaks and assumes the true
    spectral peak is well-approximated locally by a parabola in
    log-magnitude, which holds for an isolated sinusoid observed through
    a rectangular window. Degenerate triples (e.g. a flat top or an
    exact zero neighbor) produce no offset.
    """
    n = magnitude.size
    left = magnitude[(peak_index - 1) % n]
    center = magnitude[peak_index]
    right = magnitude[(peak_index + 1) % n]
    eps = np.finfo(np.float64).tiny
    y_left = math.log(max(left, eps))
    y_center = math.log(max(center, eps))
    y_right = math.log(max(right, eps))
    denom = y_left - 2.0 * y_center + y_right
    if denom == 0.0:
        return 0.0
    delta = 0.5 * (y_left - y_right) / denom
    return float(np.clip(delta, -0.5, 0.5))


def estimate_peak_frequency(
    samples: np.ndarray,
    fs: float,
    *,
    refine: bool = True,
) -> PeakFrequencyEstimate:
    """Estimate the frequency of the dominant spectral component.

    Locates the largest-magnitude bin of the FFT-based two-sided
    spectrum (via :func:`iqwav.dsp.magnitude_spectrum`) and reports its
    frequency, optionally refined to sub-bin precision by three-point
    parabolic interpolation of the local log-magnitude with circular FFT
    neighbors.

    This is a DOMINANT SPECTRAL COMPONENT estimator. It is NOT blind
    carrier-frequency estimation, occupied-band center estimation, CFO
    estimation, CFO correction, signal detection, bandwidth estimation,
    or SNR/noise estimation. For a wideband modulated signal,
    ``frequency_hz`` is simply the strongest spectral component and must
    not be described as the signal center or carrier unless the signal
    model justifies that interpretation.

    Real vs. complex input:
        For complex-valued ``samples`` (typical baseband IQ), the full
        two-sided spectrum is searched and the returned frequency is
        signed: a tone at ``+f`` estimates to approximately ``+f``, and
        a tone at ``-f`` to approximately ``-f``.

        For real-valued ``samples`` the spectrum is conjugate-symmetric,
        so the search is restricted to the non-negative half
        (``0`` to ``fs/2`` inclusive) and the returned frequency is
        always non-negative; sign is not a meaningful concept for a
        real-valued tone.

    DC handling:
        A perfectly constant (or all-zero) waveform is rejected because
        it has no oscillating frequency to localize. The input is never
        mean-removed: if a non-constant waveform genuinely has DC as its
        strongest FFT component, ``0 Hz`` is returned as the dominant
        spectral component.

    Resolution and refinement:
        The raw FFT bin spacing is ``fs / N`` and is returned as
        ``resolution_hz``. With ``refine=True`` (default) the estimate
        is sharpened by the bounded sub-bin interpolation;
        ``bin_frequency_hz`` always reports the unrefined bin center.
        With ``refine=False``, ``frequency_hz`` equals
        ``bin_frequency_hz`` exactly.

    Args:
        samples: 1-D real or complex finite numeric samples with at
            least 4 values, not constant.
        fs: Positive finite real sampling rate in Hz. Not inferred.
        refine: If True (default), apply sub-bin parabolic
            log-magnitude interpolation. If False, return the raw FFT
            bin center. Must be a bool.

    Returns:
        The :class:`PeakFrequencyEstimate` for the dominant spectral
        component.

    Raises:
        ValueError: If ``samples``, ``fs``, or ``refine`` is invalid, or
            ``samples`` is constant (including all-zero).
    """
    samples = _validate_samples_for_peak(samples)
    fs = _validate_real_scalar(fs, "fs")
    if fs <= 0:
        raise ValueError(f"fs must be positive and finite, got {fs!r}.")
    if not isinstance(refine, bool):
        raise ValueError(f"refine must be a bool, got {refine!r}.")
    _reject_constant_signal(samples)

    n = samples.shape[0]
    freqs, magnitude = magnitude_spectrum(samples, fs=fs, fftshift=False)

    if np.iscomplexobj(samples):
        search_magnitude = magnitude
    else:
        search_magnitude = magnitude[: n // 2 + 1]

    peak_index = int(np.argmax(search_magnitude))
    bin_frequency = float(freqs[peak_index])
    if not np.iscomplexobj(samples):
        # numpy.fft.fftfreq assigns the Nyquist bin (index n // 2, for
        # even n) a negative frequency; since Nyquist's sign is inherently
        # ambiguous, take the magnitude to keep the documented "always
        # non-negative for real input" contract exact at that one bin.
        bin_frequency = abs(bin_frequency)
    resolution = fs / n

    if not refine:
        return PeakFrequencyEstimate(
            frequency_hz=bin_frequency,
            bin_frequency_hz=bin_frequency,
            resolution_hz=resolution,
            bin_index=peak_index,
            refined=False,
        )

    delta_bins = _parabolic_log_refine(magnitude, peak_index)
    refined_frequency = _wrap_bin_to_freq(peak_index + delta_bins, n, fs)
    if not np.iscomplexobj(samples) and refined_frequency < 0.0:
        refined_frequency = -refined_frequency

    return PeakFrequencyEstimate(
        frequency_hz=refined_frequency,
        bin_frequency_hz=bin_frequency,
        resolution_hz=resolution,
        bin_index=peak_index,
        refined=True,
    )
