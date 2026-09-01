"""Coarse frequency-offset estimation for oversampled complex PSK-like signals."""

from dataclasses import dataclass

import numpy as np

from ..correlation import autocorrelation
from .occupied_band import _validate_real_scalar

__all__ = ["FrequencyOffsetEstimate", "estimate_frequency_offset"]


@dataclass(frozen=True)
class FrequencyOffsetEstimate:
    """Coarse frequency-offset estimate for a complex PSK-like signal.

    ``frequency_offset_hz`` is the estimated constant carrier-frequency
    offset, ``phase_increment_rad`` is the underlying per-sample phase
    increment, and ``coherence`` is the lag-1 correlation magnitude
    reliability measure.
    """

    frequency_offset_hz: float
    phase_increment_rad: float
    coherence: float


def estimate_frequency_offset(
    samples: np.ndarray,
    fs: float,
    *,
    min_coherence: float = 0.05,
) -> FrequencyOffsetEstimate:
    """Estimate a coarse constant frequency offset of a complex IQ signal.

    For a stationary frequency offset ``x[n] = s[n] * exp(j*2*pi*df*n/fs)``
    on an oversampled rectangular PSK-like waveform, most adjacent samples
    lie inside the same symbol, so their product ``x[n+1] * conj(x[n])``
    carries the phase increment ``exp(j * 2*pi*df/fs)``. Using the
    production autocorrelation convention
    ``R[k] = (1/(N-k)) * sum(x[n+k] * conj(x[n]))`` with ``max_lag=1``:

    ``phase_increment_rad = angle(R[1])``
    ``frequency_offset_hz  = fs * phase_increment_rad / (2*pi)``

    A positive injected offset therefore produces a positive estimate;
    the estimator never conjugates or reverses the result. Symbol
    boundaries contribute random products that average out over long
    random sequences, reducing coherence but not biasing the phase.

    Reliability measure::

        coherence = abs(R[1]) / real(R[0])

    with near 1 meaning highly coherent adjacent samples. If
    ``coherence < min_coherence`` the estimate is rejected with
    ``ValueError``.

    This is deliberately a coarse baseline, not carrier synchronization.
    It assumes complex oversampled BPSK/QPSK-like symbols with a
    rectangular pulse shape (more than one sample per symbol), enough
    random symbols, a known sample rate, a stationary constant offset,
    and moderate SNR. No correction is performed. Because the estimate
    uses the principal angle of ``R[1]`` (in ``[-pi, +pi]``), it is
    inherently unambiguous only over approximately ``-fs/2 <= df <
    fs/2``, with no unwrapping or ambiguity resolution; estimation also
    becomes less meaningful near Nyquist. Severe noise can make R[1]
    unreliable, and signals with 1 sample per symbol are generally
    unsuitable because adjacent symbols are effectively uncorrelated.
    Pulse-shaped or arbitrary modulations are not yet validated.

    Args:
        samples: 1-D complex finite samples with at least 2 values and
            non-zero energy. Real-only input is rejected because the
            offset sign/phase information is not preserved in the same
            way.
        fs: Positive finite real sampling rate in Hz. Not inferred.
        min_coherence: Minimum accepted lag-1 coherence. Must be a
            finite real scalar in ``[0, 1]``; 0 is allowed for
            diagnostic use.

    Returns:
        The :class:`FrequencyOffsetEstimate`.

    Raises:
        ValueError: If any argument is invalid, the input has zero
            energy, or ``coherence < min_coherence``.
    """
    samples = np.asarray(samples)
    if samples.ndim != 1:
        raise ValueError(
            f"samples must be one-dimensional, got shape {samples.shape}."
        )
    if samples.dtype.kind != "c":
        raise ValueError(
            f"samples must be complex-valued IQ data, got dtype {samples.dtype}."
        )
    if samples.shape[0] < 2:
        raise ValueError(
            f"samples must contain at least 2 samples, got {samples.shape[0]}."
        )
    if not np.all(np.isfinite(samples)):
        raise ValueError("samples must contain only finite values.")
    fs = _validate_real_scalar(fs, "fs")
    if fs <= 0:
        raise ValueError(f"fs must be positive and finite, got {fs!r}.")
    coherence_min = _validate_real_scalar(min_coherence, "min_coherence")
    if not 0.0 <= coherence_min <= 1.0:
        raise ValueError(
            f"min_coherence must satisfy 0 <= min_coherence <= 1, "
            f"got {min_coherence!r}."
        )

    correlation = autocorrelation(samples, max_lag=1)
    r_zero = float(np.real(correlation[0]))
    if r_zero == 0.0:
        raise ValueError(
            "samples have zero energy; cannot estimate a frequency offset."
        )
    r_one = complex(correlation[1])
    phase_increment_rad = float(np.angle(r_one))
    coherence = float(np.abs(r_one) / r_zero)
    if coherence < coherence_min:
        raise ValueError(
            f"Lag-1 coherence {coherence:.4f} is below min_coherence "
            f"{coherence_min!r}; no reliable coarse frequency-offset "
            f"estimate is available."
        )
    return FrequencyOffsetEstimate(
        frequency_offset_hz=float(fs * phase_increment_rad / (2.0 * np.pi)),
        phase_increment_rad=phase_increment_rad,
        coherence=coherence,
    )
