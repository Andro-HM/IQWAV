"""Bounded complex-baseband analog modulation primitives.

These modulators convert a caller-supplied real message into complex
baseband IQ. They do not generate messages, inject CFO, AWGN, amplitude
scaling, or static carrier phase, and they are not classifiers.
"""

import math

import numpy as np
import numpy.typing as npt

__all__ = ["am_modulate", "fm_modulate", "pm_modulate"]


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


def _validate_message(message: np.ndarray) -> npt.NDArray[np.float64]:
    """Validate a real normalized analog message and return float64."""
    message = np.asarray(message)
    if message.ndim != 1:
        raise ValueError(
            f"message must be one-dimensional, got shape {message.shape}."
        )
    if message.size == 0:
        raise ValueError("message must contain at least one value.")
    if np.iscomplexobj(message):
        raise ValueError("message must be real-valued.")
    if not (
        np.issubdtype(message.dtype, np.floating)
        or np.issubdtype(message.dtype, np.integer)
    ):
        raise ValueError(
            f"message must be a real floating or integer array, "
            f"got dtype {message.dtype}."
        )
    message = np.asarray(message, dtype=np.float64)
    if not np.all(np.isfinite(message)):
        raise ValueError("message must contain only finite values.")
    if np.any(np.abs(message) > 1.0):
        raise ValueError("message samples must satisfy |m[n]| <= 1.")
    return message


def _validate_nonnegative_deviation(value: object, name: str) -> float:
    """Validate a finite peak deviation that is >= 0."""
    value = _validate_real_scalar(value, name)
    if value < 0.0:
        raise ValueError(f"{name} must be >= 0, got {value!r}.")
    return value


def am_modulate(
    message: np.ndarray,
    modulation_index: float,
) -> npt.NDArray[np.complex128]:
    """Conventional carrier-present AM (DSB-LC) at complex baseband.

    For a real normalized message ``m[n]`` and modulation index ``μ``:

    ``s[n] = 1 + μ m[n]``

    stored as complex baseband IQ with the real envelope on I and Q = 0.
    The carrier amplitude is 1. This is conventional AM with a transmitted
    carrier, not DSB-SC.

    The initial contract requires ``|m[n]| <= 1`` and ``0 <= μ <= 1``, so
    the envelope stays non-negative. Amplitude scaling, CFO, AWGN, and
    static carrier phase are not applied here.

    Args:
        message: 1-D real message ``m[n]``. Must be non-empty, finite, and
            satisfy ``|m[n]| <= 1``.
        modulation_index: Modulation index ``μ``. Must be a finite real
            scalar with ``0 <= μ <= 1``.

    Returns:
        A 1-D complex128 array of the same length as ``message``.

    Raises:
        ValueError: If ``message`` or ``modulation_index`` is invalid.
    """
    modulation_index = _validate_real_scalar(
        modulation_index, "modulation_index"
    )
    if not 0.0 <= modulation_index <= 1.0:
        raise ValueError(
            f"modulation_index must satisfy 0 <= μ <= 1, "
            f"got {modulation_index!r}."
        )
    message = _validate_message(message)
    envelope = 1.0 + modulation_index * message
    return np.asarray(envelope, dtype=np.complex128)


def fm_modulate(
    message: np.ndarray,
    fs: float,
    frequency_deviation_hz: float,
) -> npt.NDArray[np.complex128]:
    """Complex-baseband FM with supplied peak frequency deviation.

    Discrete-time contract, with ``φ[-1] = 0``:

    ``φ[n] - φ[n-1] = 2 π (Δf / Fs) m[n]``

    ``s[n] = exp(j φ[n])``

    Equivalently ``φ[n] = 2 π (Δf / Fs) cumsum(m)[n]``. Instantaneous
    frequency at sample ``n`` is ``Δf m[n]`` hertz. Unit magnitude;
    extra amplitude, CFO, AWGN, and static carrier phase are not applied.

    Existing :func:`iqwav.demod.fm_demodulate` returns wrapped adjacent
    phase increments ``angle(s[n+1] conj(s[n]))``. Under this contract,
    that output equals ``2 π (Δf / Fs) m[n+1]`` when the increment
    magnitude is below ``π``. It therefore recovers ``m[1:]``, not
    ``m[:-1]`` or the full ``m``, at that radians-per-sample scale.

    A constant message ``m[n] = c`` is a complex tone of frequency
    ``c Δf``, with initial sample ``s[0] = exp(j 2 π Δf c / Fs)`` rather
    than ``1``. That constant phase offset is a consequence of
    ``φ[-1] = 0`` and is not a static-phase impairment primitive.

    Args:
        message: 1-D real message ``m[n]``. Must be non-empty, finite, and
            satisfy ``|m[n]| <= 1``.
        fs: Sampling frequency in Hz. Must be positive and finite.
        frequency_deviation_hz: Peak frequency deviation ``Δf`` in Hz.
            Must be finite and satisfy ``0 <= Δf < fs / 2`` so that the
            instantaneous frequency stays strictly below Nyquist.

    Returns:
        A 1-D complex128 array of the same length as ``message``.

    Raises:
        ValueError: If ``message``, ``fs``, or ``frequency_deviation_hz``
            is invalid.
    """
    fs = _validate_real_scalar(fs, "fs")
    if fs <= 0.0:
        raise ValueError(f"fs must be positive and finite, got {fs!r}.")
    frequency_deviation_hz = _validate_nonnegative_deviation(
        frequency_deviation_hz, "frequency_deviation_hz"
    )
    nyquist = fs / 2.0
    if frequency_deviation_hz >= nyquist:
        raise ValueError(
            f"frequency_deviation_hz must be strictly below the Nyquist "
            f"limit fs/2 = {nyquist}, got {frequency_deviation_hz!r}."
        )
    message = _validate_message(message)
    phase = (
        2.0 * np.pi * frequency_deviation_hz / fs
    ) * np.cumsum(message)
    return np.exp(1j * phase)


def pm_modulate(
    message: np.ndarray,
    phase_deviation_rad: float,
) -> npt.NDArray[np.complex128]:
    """Complex-baseband PM with supplied peak phase deviation.

    ``s[n] = exp(j β m[n])``

    with ``φ[n] = β m[n]`` and no integration. Unit magnitude; extra
    amplitude, CFO, AWGN, and static carrier phase are not applied.
    Adjacent phase differences equal ``β (m[n] - m[n-1])`` and are not
    proportional to ``m[n]`` in general. That is the mathematical
    distinction from :func:`fm_modulate`, not an AMC identifiability
    claim.

    Args:
        message: 1-D real message ``m[n]``. Must be non-empty, finite, and
            satisfy ``|m[n]| <= 1``.
        phase_deviation_rad: Peak phase deviation ``β`` in radians. Must
            be a finite real scalar ``>= 0``. Values large enough to wrap
            ``(-π, π]`` are mathematically valid PM.

    Returns:
        A 1-D complex128 array of the same length as ``message``.

    Raises:
        ValueError: If ``message`` or ``phase_deviation_rad`` is invalid.
    """
    phase_deviation_rad = _validate_nonnegative_deviation(
        phase_deviation_rad, "phase_deviation_rad"
    )
    message = _validate_message(message)
    return np.exp(1j * phase_deviation_rad * message)
