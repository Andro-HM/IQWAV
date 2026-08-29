"""FFT-based spectrum analysis utilities."""

import math

import numpy as np
import numpy.typing as npt

__all__ = ["magnitude_spectrum"]


def magnitude_spectrum( samples: np.ndarray, fs: float, fftshift: bool = True,) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Compute the FFT-based magnitude spectrum of a 1-D signal.

    The magnitude is the raw ``abs(FFT)`` of ``samples``: no dB conversion,
    no normalization, no windowing, and no averaging.

    Args:
        samples: 1-D real or complex sample array with at least one value.
            All values must be finite.
        fs: Sampling frequency in Hz. Must be positive and finite.
        fftshift: If True, shift the zero-frequency component to the center
            so the frequency axis runs from ``-fs/2`` to ``+fs/2`` with
            negative frequencies on the left. If False, keep normal NumPy
            FFT ordering (``0 .. fs/2``, then ``-fs/2 .. 0``).

    Returns:
        A tuple ``(freqs, magnitude)`` where ``freqs`` is the frequency axis
        in Hz and ``magnitude`` is the raw FFT magnitude, both float64
        arrays of the same length as ``samples`` and in the same ordering.

    Raises:
        ValueError: If ``fs`` is not positive and finite, if ``samples`` is
            not one-dimensional, is empty, or contains non-finite values.
    """

    if not math.isfinite(fs) or fs <= 0:
        raise ValueError(f"fs must be positive and finite, got {fs!r}.")
    
    samples = np.asarray(samples)

    
    if samples.ndim != 1:
        raise ValueError( f"samples must be one-dimensional, got shape {samples.shape}.")
    if samples.size == 0:
        raise ValueError("samples must contain at least one value.")
    if not np.all(np.isfinite(samples)):
        raise ValueError("samples must contain only finite values.")

    
    freqs = np.fft.fftfreq(samples.shape[0], d=1.0 / fs)
    magnitude = np.abs(np.fft.fft(samples))
    if fftshift:
        freqs = np.fft.fftshift(freqs)
        magnitude = np.fft.fftshift(magnitude)
    return freqs, magnitude
