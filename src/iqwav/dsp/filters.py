"""FIR filter design and application utilities."""

import math

import numpy as np
import numpy.typing as npt
from scipy import signal

__all__ = [
    "apply_fir_filter",
    "design_bandpass_fir",
    "design_highpass_fir",
    "design_lowpass_fir",
]


def _validate_fs(fs: float) -> None:
    """Validate the sampling frequency."""
    if not math.isfinite(fs) or fs <= 0:
        raise ValueError(f"fs must be positive and finite, got {fs!r}.")


def _validate_numtaps(numtaps: int) -> None:
    """Validate the number of FIR taps."""
    if not isinstance(numtaps, (int, np.integer)) or numtaps < 2:
        raise ValueError(f"numtaps must be an integer >= 2, got {numtaps!r}.")


def _validate_cutoff(value: float, fs: float, name: str) -> None:
    """Validate a single cutoff frequency against the Nyquist limit."""
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value!r}.")
    if not 0 < value < fs / 2:
        raise ValueError(
            f"{name} must lie strictly inside (0, fs/2 = {fs / 2}), got {value!r}."
        )


def design_lowpass_fir(
    fs: float,
    cutoff: float,
    numtaps: int = 101,
) -> npt.NDArray[np.float64]:
    """Design a linear-phase low-pass FIR filter.

    Args:
        fs: Sampling frequency in Hz. Must be positive and finite.
        cutoff: Cutoff frequency in Hz. Must lie strictly inside
            ``(0, fs/2)``.
        numtaps: Filter length. Must be an integer >= 2.

    Returns:
        The float64 FIR coefficients of length ``numtaps``.

    Raises:
        ValueError: If any argument violates the constraints above.
    """
    _validate_fs(fs)
    _validate_numtaps(numtaps)
    _validate_cutoff(cutoff, fs, "cutoff")
    return signal.firwin(numtaps, cutoff, fs=fs)


def design_highpass_fir(
    fs: float,
    cutoff: float,
    numtaps: int = 101,
) -> npt.NDArray[np.float64]:
    """Design a linear-phase high-pass FIR filter.

    Args:
        fs: Sampling frequency in Hz. Must be positive and finite.
        cutoff: Cutoff frequency in Hz. Must lie strictly inside
            ``(0, fs/2)``.
        numtaps: Filter length. Must be an integer >= 2. Note that SciPy
            requires an odd length here, because an even-length filter
            cannot pass the Nyquist frequency.

    Returns:
        The float64 FIR coefficients of length ``numtaps``.

    Raises:
        ValueError: If any argument violates the constraints above.
    """
    _validate_fs(fs)
    _validate_numtaps(numtaps)
    _validate_cutoff(cutoff, fs, "cutoff")
    return signal.firwin(numtaps, cutoff, fs=fs, pass_zero=False)


def design_bandpass_fir(
    fs: float,
    lowcut: float,
    highcut: float,
    numtaps: int = 101,
) -> npt.NDArray[np.float64]:
    """Design a linear-phase band-pass FIR filter.

    Args:
        fs: Sampling frequency in Hz. Must be positive and finite.
        lowcut: Lower passband edge in Hz.
        highcut: Upper passband edge in Hz. Must satisfy
            ``0 < lowcut < highcut < fs/2``.
        numtaps: Filter length. Must be an integer >= 2.

    Returns:
        The float64 FIR coefficients of length ``numtaps``.

    Raises:
        ValueError: If any argument violates the constraints above.
    """
    _validate_fs(fs)
    _validate_numtaps(numtaps)
    for name, value in (("lowcut", lowcut), ("highcut", highcut)):
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite, got {value!r}.")
    if not 0 < lowcut < highcut < fs / 2:
        raise ValueError(
            f"must have 0 < lowcut < highcut < fs/2 = {fs / 2}, "
            f"got lowcut={lowcut!r}, highcut={highcut!r}."
        )
    return signal.firwin(numtaps, [lowcut, highcut], fs=fs, pass_zero=False)


def apply_fir_filter(
    samples: np.ndarray,
    taps: np.ndarray,
) -> npt.NDArray[np.float64] | npt.NDArray[np.complex128]:
    """Apply an FIR filter to a 1-D signal using a causal direct form.

    Args:
        samples: 1-D real or complex sample array. Must be non-empty with
            only finite values.
        taps: 1-D FIR coefficients. Must be non-empty with only finite
            values.

    Returns:
        The filtered samples, same length and ordering as the input. Real
        input stays real (float64); complex input stays complex
        (complex128).

    Raises:
        ValueError: If ``samples`` or ``taps`` is not a non-empty 1-D array
            of finite values.
    """
    samples = np.asarray(samples)
    if samples.ndim != 1:
        raise ValueError(
            f"samples must be one-dimensional, got shape {samples.shape}."
        )
    if samples.size == 0:
        raise ValueError("samples must contain at least one value.")
    if not np.all(np.isfinite(samples)):
        raise ValueError("samples must contain only finite values.")
    taps = np.asarray(taps)
    if taps.ndim != 1:
        raise ValueError(f"taps must be one-dimensional, got shape {taps.shape}.")
    if taps.size == 0:
        raise ValueError("taps must contain at least one value.")
    if not np.all(np.isfinite(taps)):
        raise ValueError("taps must contain only finite values.")
    return signal.lfilter(taps, 1.0, samples)
