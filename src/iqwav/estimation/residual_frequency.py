"""Residual carrier-frequency refinement for IQWAV rectangular BPSK/QPSK."""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize_scalar

from .occupied_band import _validate_real_scalar

__all__ = [
    "ResidualFrequencyOffsetEstimate",
    "estimate_residual_frequency_offset",
]


_M_TH_POWER = {
    "bpsk": 2,
    "qpsk": 4,
}

# Frozen validated algorithm constants. Do not retune per record.
_FFT_PAD = 8
_REFINE_BINS = 8
_BRENT_XATOL_HZ = 1e-6


@dataclass(frozen=True)
class ResidualFrequencyOffsetEstimate:
    """Residual constant-CFO estimate for rectangular IQWAV BPSK/QPSK.

    ``residual_frequency_hz`` is the estimated leftover carrier-frequency
    offset of the supplied block, wrapped into the modulation's canonical
    M-th-power range. ``phase_increment_rad`` is the corresponding
    per-sample residual-carrier increment of the original samples,
    ``2*pi*residual_frequency_hz/fs``.
    """

    residual_frequency_hz: float
    phase_increment_rad: float


def _wrap_half_open(value: float, half_period: float) -> float:
    """Wrap ``value`` into ``[-half_period, +half_period)``."""
    period = 2.0 * half_period
    return (value + half_period) % period - half_period


def _canonical_limits(fs: float, order: int) -> tuple[float, float]:
    half = fs / (2.0 * order)
    return -half, half


def estimate_residual_frequency_offset(
    samples: np.ndarray,
    fs: float,
    modulation: str,
) -> ResidualFrequencyOffsetEstimate:
    """Estimate a constant residual CFO of rectangular IQWAV BPSK/QPSK.

    Intended use is leftover-frequency refinement after coarse lag-1
    correction::

        coarse = estimate_frequency_offset(x, fs)
        y      = correct_frequency_offset(x, fs, coarse.frequency_offset_hz)
        fine   = estimate_residual_frequency_offset(y, fs, modulation)
        z      = correct_frequency_offset(y, fs, fine.residual_frequency_hz)

    This is one-shot residual-frequency refinement, not carrier tracking,
    not a Costas loop, and not a PLL. It estimates one constant leftover
    frequency for the whole supplied block.

    For a constant residual CFO and constant phase

        x[n] = s[n] * exp(j*2*pi*df*n/fs) * exp(j*phi)

    every ideal IQWAV constellation point satisfies a fixed ``s**M``
    relation (BPSK: ``s**2 = +1``; QPSK: ``s**4 = -1``), so the M-th
    power of a globally scaled copy is a tone at ``M*df``. The estimator
    maximises whole-block coherent tone power of

        z = (x / mean(|x|)) ** M

    An 8x zero-padded FFT locates the global peak inside the canonical
    residual range; Brent then maximises the same coherent-power
    objective in a local window around that peak. There is no deadband
    and no lag-1 statistic.

        P(df) = |sum z[n] * exp(-j*2*pi*M*df*n/fs)|^2
        df_hat in canonical range

    ``c_M`` constellation-reference compensation is not used: it is a
    constant and does not change the tone frequency.

    Canonical unambiguous ranges:

        BPSK (M=2): [-fs/4, +fs/4)
        QPSK (M=4): [-fs/8, +fs/8)

    Offsets outside that interval alias with period ``fs/M``::

        df_true = df_hat + k * fs / M

    That range is narrower than the coarse lag-1 estimator's
    ``(-fs/2, fs/2]``. This function does not replace
    ``estimate_frequency_offset``.

    No lag-1 coherence is returned. The previous experimental lag-1
    ``coherence`` / ``min_coherence`` API was specific to a rejected
    estimator and is not preserved under a different statistic.

    Scope and assumptions:
        - Known IQWAV modulation family, exactly ``"bpsk"`` or
          ``"qpsk"``. Arbitrary M-PSK is deliberately not advertised.
        - Currently validated only on rectangular integer-SPS IQWAV
          BPSK/QPSK waveforms and on ideal symbol-rate samples of those
          same constellations. This does NOT imply support for
          pulse-shaped, RRC, fractional-SPS, or arbitrary oversampled
          PSK.
        - The leftover frequency is assumed constant over the block.
        - This is NOT coarse acquisition, NOT carrier tracking, NOT a
          Costas loop, NOT a PLL, NOT static phase estimation, and NOT
          timing recovery.
        - Validated noisy operation is the 20 dB AWGN rectangular
          BPSK/QPSK campaign in LOGS; that is not a universal SNR claim.

    Args:
        samples: 1-D complex finite samples with at least 2 values and
            non-zero energy. Real-only input is rejected because the
            offset sign/phase information is not preserved.
        fs: Positive finite real sampling rate in Hz. Not inferred.
        modulation: Supported modulation family, exactly ``"bpsk"`` or
            ``"qpsk"``.

    Returns:
        The :class:`ResidualFrequencyOffsetEstimate`.

    Raises:
        ValueError: If any argument is invalid or the input has zero
            energy.
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
    if modulation not in _M_TH_POWER:
        raise ValueError(
            f"modulation must be one of {tuple(_M_TH_POWER)}, "
            f"got {modulation!r}."
        )

    order = _M_TH_POWER[modulation]
    scale = float(np.mean(np.abs(samples)))
    if scale == 0.0:
        raise ValueError(
            "samples have zero energy; cannot estimate a residual "
            "frequency offset."
        )
    powered = (samples / scale) ** order
    n_time = np.arange(powered.shape[0], dtype=np.float64)
    nfft = _FFT_PAD * powered.shape[0]
    spectrum = np.fft.fft(powered, n=nfft)
    freqs_z = np.fft.fftfreq(nfft, d=1.0 / fs)
    residual_axis = freqs_z / order
    search_lo, search_hi = _canonical_limits(fs, order)
    mask = (residual_axis >= search_lo) & (residual_axis < search_hi)
    if not np.any(mask):
        raise ValueError(
            "no FFT bins inside the canonical residual-frequency range."
        )
    peak_local = int(np.argmax(np.abs(spectrum[mask]) ** 2))
    df0 = float(residual_axis[mask][peak_local])
    df_bin_hz = (fs / nfft) / order
    half_width = _REFINE_BINS * df_bin_hz
    lo = max(search_lo, df0 - half_width)
    hi = min(search_hi, df0 + half_width)
    if hi <= lo:
        hi = min(search_hi, lo + df_bin_hz)

    def coherent_power(df: float) -> float:
        phasor = np.exp(-1j * 2.0 * np.pi * (order * df) * n_time / fs)
        return float(np.abs(np.dot(powered, phasor)) ** 2)

    result = minimize_scalar(
        lambda df: -coherent_power(df),
        bounds=(lo, hi),
        method="bounded",
        options={"xatol": _BRENT_XATOL_HZ},
    )
    residual_frequency_hz = _wrap_half_open(float(result.x), fs / (2.0 * order))
    phase_increment_rad = 2.0 * np.pi * residual_frequency_hz / fs
    return ResidualFrequencyOffsetEstimate(
        residual_frequency_hz=float(residual_frequency_hz),
        phase_increment_rad=float(phase_increment_rad),
    )
