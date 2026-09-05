"""Receiver-side constant carrier-phase-offset correction."""

import math

import numpy as np
import numpy.typing as npt

__all__ = ["correct_phase_offset"]


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


def correct_phase_offset(
    samples: np.ndarray,
    phase_offset_rad: float,
) -> npt.NDArray[np.complex128]:
    """Correct a caller-supplied constant carrier phase offset.

    Convention:
        If the received signal relates to the transmitted signal ``s[n]``
        by ``r[n] = s[n] * exp(+j * phase_offset_rad)``, this returns
        ``y[n] = r[n] * exp(-j * phase_offset_rad)``. A positive supplied
        phase therefore applies a negative correcting rotation, and this
        function exactly inverts :func:`iqwav.dsp.apply_phase_offset`
        when the true phase is supplied.

    Scope:
        This ONLY removes a caller-supplied constant phase rotation. It
        is NOT phase estimation, blind carrier recovery, CFO correction,
        a Costas loop, a PLL, carrier tracking, timing recovery, AMR,
        filtering, or resampling.

    Phase ambiguity:
        Blind PSK phase recovery is rotationally ambiguous (modulo ``pi``
        for BPSK, modulo ``pi/2`` for QPSK), so correcting a blind
        estimate can still leave the constellation rotated onto itself;
        absolute bit labeling cannot be resolved from the phase estimate
        alone.

    Args:
        samples: 1-D complex IQ sample array. Must be non-empty with
            only finite values. Never mutated.
        phase_offset_rad: Known constant phase offset to remove, in
            radians. May be positive, negative, or zero; must be a
            finite real scalar. Zero-phase correction still returns a
            new array.

    Returns:
        A new complex128 array of corrected samples with unchanged
        length; magnitudes are preserved exactly.

    Raises:
        ValueError: If ``samples`` is not a non-empty 1-D finite complex
            array, or ``phase_offset_rad`` is not a finite real scalar
            (bools are rejected).
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
    phase = _validate_real_scalar(phase_offset_rad, "phase_offset_rad")

    corrected = samples * np.exp(-1j * phase)
    return corrected.astype(np.complex128, copy=False)
