"""Power spectral density estimation utilities."""

import math

import numpy as np
import numpy.typing as npt
from scipy import signal

__all__ = ["periodogram_psd", "welch_psd"]


def _validate_psd_args(samples: np.ndarray, fs: float) -> np.ndarray:
    """Validate PSD arguments and return the samples as an ndarray."""
    if not math.isfinite(fs) or fs <= 0:
        raise ValueError(f"fs must be positive and finite, got {fs!r}.")
    samples = np.asarray(samples)
    if samples.ndim != 1:
        raise ValueError(
            f"samples must be one-dimensional, got shape {samples.shape}."
        )
    if samples.size == 0:
        raise ValueError("samples must contain at least one value.")
    if not np.all(np.isfinite(samples)):
        raise ValueError("samples must contain only finite values.")
    return samples


def periodogram_psd(
    samples: np.ndarray,
    fs: float,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Estimate the power spectral density with a single periodogram.

    Wraps ``scipy.signal.periodogram`` with density scaling and a two-sided
    spectrum, then shifts to centered frequency ordering so the frequency
    axis runs from negative frequencies through 0 to positive frequencies.

    Args:
        samples: 1-D real or complex sample array. Must be non-empty with
            only finite values.
        fs: Sampling frequency in Hz. Must be positive and finite.

    Returns:
        A tuple ``(freqs, psd)``: the frequency axis in Hz and the linear
        (non-dB) PSD values, both float64 arrays of the same length as
        ``samples``.

    Raises:
        ValueError: If ``fs`` is not positive and finite or ``samples`` is
            not a non-empty 1-D array of finite values.
    """
    samples = _validate_psd_args(samples, fs)
    freqs, psd = signal.periodogram(samples, fs=fs, return_onesided=False)
    return np.fft.fftshift(freqs), np.fft.fftshift(psd)


def welch_psd(
    samples: np.ndarray,
    fs: float,
    nperseg: int | None = None,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Estimate the power spectral density with Welch's method.

    Wraps ``scipy.signal.welch`` with density scaling and a two-sided
    spectrum, then shifts to centered frequency ordering so the frequency
    axis runs from negative frequencies through 0 to positive frequencies.

    Args:
        samples: 1-D real or complex sample array. Must be non-empty with
            only finite values.
        fs: Sampling frequency in Hz. Must be positive and finite.
        nperseg: Segment length in samples. If None, the SciPy default is
            used. If provided, must be a positive integer.

    Returns:
        A tuple ``(freqs, psd)``: the frequency axis in Hz and the linear
        (non-dB) PSD values, both float64 arrays of length ``nperseg`` (or
        the SciPy default segment length).

    Raises:
        ValueError: If ``fs`` is not positive and finite, ``samples`` is not
            a non-empty 1-D array of finite values, or ``nperseg`` is
            provided and is not a positive integer.
    """
    samples = _validate_psd_args(samples, fs)
    if nperseg is not None and (
        not isinstance(nperseg, (int, np.integer)) or nperseg < 1
    ):
        raise ValueError(f"nperseg must be a positive integer, got {nperseg!r}.")
    freqs, psd = signal.welch(
        samples, fs=fs, nperseg=nperseg, return_onesided=False
    )
    return np.fft.fftshift(freqs), np.fft.fftshift(psd)
