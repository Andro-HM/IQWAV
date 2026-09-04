"""Receiver-side carrier-frequency-offset correction."""

import math

import numpy as np
import numpy.typing as npt

__all__ = ["correct_frequency_offset"]


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


def correct_frequency_offset(
    samples: np.ndarray,
    fs: float,
    frequency_offset_hz: float,
) -> npt.NDArray[np.complex128]:
    """Correct a caller-supplied constant carrier frequency offset.

    Convention:
        If the received signal relates to the transmitted signal ``s[n]``
        by ``r[n] = s[n] * exp(+j * 2*pi*frequency_offset_hz*n / fs)``,
        this returns ``y[n] = r[n] * exp(-j * 2*pi*frequency_offset_hz*n / fs)``
        with ``n = 0, 1, ..., N-1`` in array order. A positive supplied
        CFO therefore applies a negative correcting rotation, and this
        function exactly inverts
        :func:`iqwav.dsp.apply_frequency_offset` when the true offset is
        supplied.

    Scope:
        This ONLY corrects a caller-supplied constant frequency offset.
        It is NOT CFO estimation, blind carrier recovery, phase-offset
        correction, a Costas loop, a PLL, carrier tracking, timing
        recovery, symbol-rate estimation, AMR, filtering, or resampling.

    Phase behavior:
        CFO correction removes the phase SLOPE with time. It does NOT
        necessarily remove a constant phase offset: for
        ``r[n] = s[n] * exp(j*phi0) * exp(j*2*pi*df*n/fs)`` a perfect
        correction yields ``s[n] * exp(j*phi0)``, so the constellation
        may become stationary yet remain rotated. Carrier-phase recovery
        is a separate future milestone.

    Aliasing:
        No restriction is imposed on the offset magnitude relative to
        the symbol rate. The correcting phasor is periodic in the offset
        with period ``fs``, so offsets outside the ``(-fs/2, fs/2]``
        principal range alias modulo the sampling frequency; the caller
        supplies the offset in the same convention it was measured.

    Args:
        samples: 1-D complex IQ sample array. Must be non-empty with
            only finite values. Never mutated.
        fs: Positive finite real sampling rate in Hz.
        frequency_offset_hz: Known frequency offset to remove, in Hz.
            May be positive, negative, or zero; must be a finite real
            scalar.

    Returns:
        A new complex128 array of corrected samples with unchanged
        length; magnitudes are preserved exactly.

    Raises:
        ValueError: If ``samples`` is not a non-empty 1-D finite complex
            array, or ``fs`` / ``frequency_offset_hz`` is not a finite
            real scalar of the required kind.
    """
    samples = np.asarray(samples)
    if samples.ndim != 1:
        raise ValueError(
            f"samples must be one-dimensional, got shape {samples.shape}."
        )
    if samples.size == 0:
        raise ValueError("samples must contain at least one value.")
    if not np.iscomplexobj(samples):
        raise ValueError("samples must be complex-valued IQ data.")
    if not np.all(np.isfinite(samples)):
        raise ValueError("samples must contain only finite values.")
    fs = _validate_real_scalar(fs, "fs")
    if fs <= 0:
        raise ValueError(f"fs must be positive and finite, got {fs!r}.")
    offset = _validate_real_scalar(frequency_offset_hz, "frequency_offset_hz")

    n = np.arange(samples.shape[0], dtype=np.float64)
    corrected = samples * np.exp(-1j * 2.0 * np.pi * offset * n / fs)
    return corrected.astype(np.complex128, copy=False)
