"""IQ carrier impairment utilities."""

import math

import numpy as np
import numpy.typing as npt

__all__ = ["apply_frequency_offset", "apply_phase_offset"]


def _validate_iq_samples(samples: np.ndarray) -> npt.NDArray[np.complexfloating]:
    """Validate 1-D non-empty finite complex IQ samples."""
    samples = np.asarray(samples)
    if samples.ndim != 1:
        raise ValueError(
            f"samples must be one-dimensional, got shape {samples.shape}."
        )
    if samples.size == 0:
        raise ValueError("samples must contain at least one value.")
    if not np.all(np.isfinite(samples)):
        raise ValueError("samples must contain only finite values.")
    if not np.iscomplexobj(samples):
        raise ValueError("samples must be complex-valued IQ data.")
    return samples


def apply_frequency_offset(
    samples: np.ndarray,
    fs: float,
    freq_offset_hz: float,
) -> npt.NDArray[np.complex128]:
    """Apply a carrier frequency offset to complex IQ samples.

    Multiplies the samples by ``exp(j * 2*pi*freq_offset_hz*n/fs)``, which
    shifts the spectrum by ``freq_offset_hz``: a positive offset shifts the
    spectrum toward positive frequencies.

    Args:
        samples: 1-D complex IQ sample array. Must be non-empty with only
            finite values.
        fs: Sampling frequency in Hz. Must be positive and finite.
        freq_offset_hz: Frequency offset in Hz. Must be finite.

    Returns:
        The offset samples, same length, complex128.

    Raises:
        ValueError: If any argument violates the constraints above.
    """
    if not math.isfinite(fs) or fs <= 0:
        raise ValueError(f"fs must be positive and finite, got {fs!r}.")
    if not math.isfinite(freq_offset_hz):
        raise ValueError(
            f"freq_offset_hz must be finite, got {freq_offset_hz!r}."
        )
    samples = _validate_iq_samples(samples)
    n = np.arange(samples.shape[0], dtype=np.float64)
    shifted = samples * np.exp(1j * 2.0 * np.pi * freq_offset_hz * n / fs)
    return shifted.astype(np.complex128, copy=False)


def apply_phase_offset(
    samples: np.ndarray,
    phase_rad: float,
) -> npt.NDArray[np.complex128]:
    """Apply a constant carrier phase offset to complex IQ samples.

    Multiplies the samples by ``exp(j * phase_rad)``, which rotates every
    sample by ``phase_rad`` without changing magnitudes.

    Args:
        samples: 1-D complex IQ sample array. Must be non-empty with only
            finite values.
        phase_rad: Phase offset in radians. Must be finite.

    Returns:
        The rotated samples, same length, complex128.

    Raises:
        ValueError: If any argument violates the constraints above.
    """
    if not math.isfinite(phase_rad):
        raise ValueError(f"phase_rad must be finite, got {phase_rad!r}.")
    samples = _validate_iq_samples(samples)
    rotated = samples * np.exp(1j * phase_rad)
    return rotated.astype(np.complex128, copy=False)
