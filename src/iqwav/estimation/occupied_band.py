"""Blind spectral occupied-band detection."""

import math
from dataclasses import dataclass

import numpy as np

from ..dsp import welch_psd

__all__ = ["OccupiedBand", "detect_occupied_bands"]


@dataclass(frozen=True)
class OccupiedBand:
    """A contiguous spectral region rising above the noise floor.

    All frequencies are relative to the IQ/baseband center of the
    analyzed capture; they are not absolute RF frequencies.
    """

    lower_hz: float
    upper_hz: float
    center_hz: float
    bandwidth_hz: float
    peak_hz: float
    peak_db: float
    peak_above_noise_db: float


def _validate_samples(samples: np.ndarray) -> np.ndarray:
    """Validate 1-D finite real or complex numeric samples."""
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
    if samples.shape[0] < 2:
        raise ValueError(
            f"samples must contain at least 2 samples for Welch PSD "
            f"estimation, got {samples.shape[0]}."
        )
    if not np.all(np.isfinite(samples)):
        raise ValueError("samples must contain only finite values.")
    return samples


def _validate_real_scalar(value: object, name: str) -> float:
    """Validate a finite real scalar number and return it as a float."""
    if isinstance(value, bool) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise ValueError(f"{name} must be a real scalar number, got {value!r}.")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value!r}.")
    return value


def _contiguous_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Return (start, end) index pairs of True runs, end exclusive."""
    runs = []
    start = None
    for index, flag in enumerate(mask):
        if flag and start is None:
            start = index
        elif not flag and start is not None:
            runs.append((start, index))
            start = None
    if start is not None:
        runs.append((start, mask.size))
    return runs


def detect_occupied_bands(
    samples: np.ndarray,
    fs: float,
    *,
    nperseg: int | None = None,
    threshold_db: float = 6.0,
    min_bins: int = 3,
) -> tuple[list[OccupiedBand], float]:
    """Detect spectral bands rising significantly above the noise floor.

    Baseline algorithm: computes the Welch PSD (density scaling,
    two-sided, centered frequency ordering), converts it to dB, estimates
    the noise floor as the MEDIAN of the finite PSD values in dB, marks
    bins above ``noise_floor_db + threshold_db`` as occupied, groups
    contiguous occupied bins, and discards runs shorter than ``min_bins``.

    This is deliberately a simple explicit baseline, not a sophisticated
    detector. The median noise floor is only meaningful when less than
    roughly half of the analyzed spectrum is strongly occupied. No band
    merging, gap bridging, or SNR estimation is performed.

    Frequencies are relative to the IQ/baseband center: a band reported
    at +400 kHz means the energy sits around +400 kHz relative to the
    capture center, not at an absolute RF frequency. Welch's default
    constant detrending removes the per-segment mean, so a strong DC
    spike is largely suppressed by the existing PSD convention.

    Args:
        samples: 1-D real or complex finite samples with at least 2
            values.
        fs: Positive finite real sampling rate in Hz. Not inferred.
        nperseg: Welch segment length, with the same semantics as
            :func:`iqwav.dsp.welch_psd`.
        threshold_db: Occupied bins must exceed the noise floor by this
            many dB. Must be strictly positive.
        min_bins: Minimum number of contiguous occupied bins for a
            detected band. Must be an integer >= 1.

    Returns:
        A tuple ``(bands, noise_floor_db)``: the list of detected
        :class:`OccupiedBand` entries ordered by increasing frequency
        (possibly empty), and the estimated noise floor in dB.

    Raises:
        ValueError: If any argument is invalid, or the PSD contains no
            finite values (zero-energy or constant input) so no noise
            floor can be estimated.
    """
    samples = _validate_samples(samples)
    fs = _validate_real_scalar(fs, "fs")
    if fs <= 0:
        raise ValueError(f"fs must be positive and finite, got {fs!r}.")
    threshold = _validate_real_scalar(threshold_db, "threshold_db")
    if threshold <= 0:
        raise ValueError(
            f"threshold_db must be strictly positive, got {threshold_db!r}."
        )
    if (
        isinstance(min_bins, bool)
        or not isinstance(min_bins, (int, np.integer))
        or min_bins < 1
    ):
        raise ValueError(f"min_bins must be an integer >= 1, got {min_bins!r}.")
    if nperseg is not None and (
        not isinstance(nperseg, (int, np.integer)) or nperseg < 1
    ):
        raise ValueError(f"nperseg must be a positive integer, got {nperseg!r}.")

    freqs, psd = welch_psd(samples, fs, nperseg=nperseg)
    with np.errstate(divide="ignore"):
        psd_db = 10.0 * np.log10(psd)
    finite_db = psd_db[np.isfinite(psd_db)]
    if finite_db.size == 0:
        raise ValueError(
            "PSD contains no finite values (zero-energy or constant "
            "input); cannot estimate a noise floor."
        )
    noise_floor_db = float(np.median(finite_db))
    mask = psd_db > noise_floor_db + threshold
    df = float(freqs[1] - freqs[0])

    bands = []
    for start, end in _contiguous_runs(mask):
        if end - start < min_bins:
            continue
        peak_index = start + int(np.argmax(psd_db[start:end]))
        peak_db = float(psd_db[peak_index])
        lower_hz = float(freqs[start] - df / 2.0)
        upper_hz = float(freqs[end - 1] + df / 2.0)
        bands.append(
            OccupiedBand(
                lower_hz=lower_hz,
                upper_hz=upper_hz,
                center_hz=(lower_hz + upper_hz) / 2.0,
                bandwidth_hz=upper_hz - lower_hz,
                peak_hz=float(freqs[peak_index]),
                peak_db=peak_db,
                peak_above_noise_db=peak_db - noise_floor_db,
            )
        )
    return bands, noise_floor_db
