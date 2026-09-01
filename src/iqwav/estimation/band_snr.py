"""Blind in-band SNR estimation for a detected occupied band."""

from dataclasses import dataclass

import numpy as np

from ..dsp import welch_psd
from .occupied_band import (
    OccupiedBand,
    _validate_real_scalar,
    _validate_samples,
)

__all__ = ["SNREstimate", "estimate_band_snr"]


@dataclass(frozen=True)
class SNREstimate:
    """Blind in-band SNR estimate for one occupied band.

    ``snr_db`` is the ratio of estimated in-band signal power to
    estimated in-band noise power. It is not Eb/N0, Es/N0, a
    carrier-to-noise ratio, a modulation-quality measure, or an
    absolute receiver noise figure.
    """

    snr_db: float
    signal_power: float
    noise_power: float
    total_inband_power: float
    noise_psd: float
    noise_psd_db: float


def estimate_band_snr(
    samples: np.ndarray,
    fs: float,
    band: OccupiedBand,
    *,
    nperseg: int | None = None,
) -> SNREstimate:
    """Estimate the in-band signal-to-noise power ratio of an occupied band.

    This is a spectral/in-band estimate: the ratio of TOTAL estimated
    signal power inside the band to the estimated noise power falling
    inside the same band. It does not estimate Eb/N0, Es/N0, a
    carrier-to-noise ratio, modulation quality, or BER.

    Algorithm: computes the two-sided centered Welch PSD, selects the
    bins whose centers lie within ``band.lower_hz <= f <= band.upper_hz``,
    estimates the background noise PSD as the median of the finite,
    positive LINEAR PSD values outside the band, and integrates with the
    Welch bin spacing ``df``::

        total_inband_power = sum(inband_psd) * df
        noise_power       = noise_psd * n_inband_bins * df
        signal_power      = max(total_inband_power - noise_power, 0.0)
        snr_db            = 10*log10(signal_power / noise_power)

    All power arithmetic stays in linear units; only the final results
    are converted to dB. ``snr_db`` is ``-inf`` when the estimated signal
    power is zero. The median is taken in linear units directly, never by
    mediating dB values.

    Baseline assumptions: the background noise is approximately
    broadband and locally stationary, and fewer than roughly half of the
    usable out-of-band bins contain strong signals or interference.
    Other occupied channels inside the out-of-band region bias this
    baseline in crowded spectra.

    Args:
        samples: 1-D real or complex finite samples with at least 2
            values.
        fs: Positive finite real sampling rate in Hz. Not inferred.
        band: The target :class:`OccupiedBand`, e.g. from
            :func:`iqwav.estimation.detect_occupied_bands`. Must satisfy
            ``lower_hz < upper_hz`` and contain at least one Welch bin.
        nperseg: Welch segment length, with the same semantics as
            :func:`iqwav.dsp.welch_psd`.

    Returns:
        The :class:`SNREstimate` with all powers in linear units and
        ``snr_db`` / ``noise_psd_db`` in dB. Powers are finite and
        non-negative; ``snr_db`` may be ``-inf``.

    Raises:
        ValueError: If ``samples``, ``fs``, or ``nperseg`` is invalid;
            ``band`` is not an :class:`OccupiedBand` or has
            ``lower_hz >= upper_hz``; no Welch bin falls inside the band;
            or no usable out-of-band PSD bins remain for noise
            estimation.
    """
    samples = _validate_samples(samples)
    fs = _validate_real_scalar(fs, "fs")
    if fs <= 0:
        raise ValueError(f"fs must be positive and finite, got {fs!r}.")
    if not isinstance(band, OccupiedBand):
        raise ValueError(
            f"band must be an OccupiedBand instance, got {type(band).__name__}."
        )
    if not band.lower_hz < band.upper_hz:
        raise ValueError(
            f"band.lower_hz must be less than band.upper_hz, "
            f"got [{band.lower_hz!r}, {band.upper_hz!r}]."
        )
    if nperseg is not None and (
        not isinstance(nperseg, (int, np.integer)) or nperseg < 1
    ):
        raise ValueError(f"nperseg must be a positive integer, got {nperseg!r}.")

    freqs, psd = welch_psd(samples, fs, nperseg=nperseg)
    df = float(freqs[1] - freqs[0])
    inband = (freqs >= band.lower_hz) & (freqs <= band.upper_hz)
    n_inband = int(np.count_nonzero(inband))
    if n_inband == 0:
        raise ValueError(
            f"No Welch frequency bins fall inside the band "
            f"[{band.lower_hz!r}, {band.upper_hz!r}]; usable frequency "
            f"range is [{freqs[0]!r}, {freqs[-1]!r}]."
        )
    outband_psd = psd[~inband]
    usable = outband_psd[np.isfinite(outband_psd) & (outband_psd > 0.0)]
    if usable.size == 0:
        raise ValueError(
            "No usable out-of-band PSD bins remain for noise estimation."
        )
    noise_psd = float(np.median(usable))
    total_inband_power = float(np.sum(psd[inband]) * df)
    noise_power = float(noise_psd * n_inband * df)
    signal_power = max(total_inband_power - noise_power, 0.0)
    if signal_power == 0.0:
        snr_db = float("-inf")
    else:
        snr_db = float(10.0 * np.log10(signal_power / noise_power))
    return SNREstimate(
        snr_db=snr_db,
        signal_power=signal_power,
        noise_power=noise_power,
        total_inband_power=total_inband_power,
        noise_psd=noise_psd,
        noise_psd_db=float(10.0 * np.log10(noise_psd)),
    )
