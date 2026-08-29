"""Spectrogram (waterfall) computation utilities."""

import math

import numpy as np
import numpy.typing as npt
from scipy import signal

__all__ = ["spectrogram_data"]


def spectrogram_data(
    samples: np.ndarray,
    fs: float,
    nperseg: int = 256,
    noverlap: int | None = None,
) -> tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
]:
    """Compute a two-sided linear-power spectrogram of a 1-D signal.

    Wraps ``scipy.signal.spectrogram`` with a two-sided spectrum and shifts
    the frequency axis and the spectrogram rows so frequencies run from
    negative frequencies through 0 to positive frequencies. Power values
    stay linear (no dB conversion).

    Args:
        samples: 1-D real or complex sample array. Must be non-empty with
            only finite values.
        fs: Sampling frequency in Hz. Must be positive and finite.
        nperseg: Segment length in samples. Must be a positive integer.
        noverlap: Number of overlapping samples between segments. If None,
            the SciPy default is used. If provided, must be a non-negative
            integer strictly less than ``nperseg``.

    Returns:
        A tuple ``(time, freqs, power)``: the increasing time axis in
        seconds, the centered frequency axis in Hz, and the linear
        spectrogram power values with shape ``(len(freqs), len(time))``.

    Raises:
        ValueError: If any argument violates the constraints above.
    """
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
    if not isinstance(nperseg, (int, np.integer)) or nperseg < 1:
        raise ValueError(f"nperseg must be a positive integer, got {nperseg!r}.")
    if noverlap is not None and (
        not isinstance(noverlap, (int, np.integer))
        or noverlap < 0
        or noverlap >= nperseg
    ):
        raise ValueError(
            f"noverlap must be a non-negative integer strictly less than "
            f"nperseg={nperseg}, got {noverlap!r}."
        )
    freqs, time, power = signal.spectrogram(
        samples,
        fs=fs,
        nperseg=nperseg,
        noverlap=noverlap,
        return_onesided=False,
    )
    freqs = np.fft.fftshift(freqs)
    power = np.fft.fftshift(power, axes=0)
    return time, freqs, power
