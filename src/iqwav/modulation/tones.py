"""Synthetic tone generation with known ground-truth parameters.

Provides reusable NumPy generators for real-valued sinusoidal tones and
complex baseband IQ tones, intended as deterministic test signals for
developing and validating downstream IQWAV DSP components.
"""

import math

import numpy as np
import numpy.typing as npt

__all__ = ["generate_iq_tone", "generate_real_tone"]


def _validate_tone_args(
    fs: float,
    freq: float,
    duration: float,
    amplitude: float,
    phase: float,
    allow_nyquist: bool,
) -> int:
    """Validate tone-generation arguments and return the sample count."""
    for name, value in (
        ("fs", fs),
        ("freq", freq),
        ("duration", duration),
        ("amplitude", amplitude),
        ("phase", phase),
    ):
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite, got {value!r}.")
    if fs <= 0:
        raise ValueError(f"fs must be positive, got {fs!r}.")
    if duration <= 0:
        raise ValueError(f"duration must be positive, got {duration!r}.")
    if amplitude < 0:
        raise ValueError(f"amplitude must be non-negative, got {amplitude!r}.")
    nyquist = fs / 2
    if abs(freq) > nyquist:
        raise ValueError(
            f"|freq| must not exceed the Nyquist limit fs/2 = {nyquist}, got {freq!r}."
        )
    if abs(freq) == nyquist and not allow_nyquist:
        raise ValueError(
            f"|freq| must be strictly below the Nyquist limit fs/2 = {nyquist} "
            f"because the sign of freq is not resolvable at Nyquist, got {freq!r}."
        )
    n_total = fs * duration
    if not math.isfinite(n_total):
        raise ValueError(f"fs * duration is too large, got {n_total!r}.")
    n_samples = int(round(n_total))
    if n_samples < 1:
        raise ValueError(
            f"duration {duration!r} is too short to produce at least one "
            f"sample at fs = {fs!r}."
        )
    return n_samples


def generate_real_tone(
    fs: float,
    freq: float,
    duration: float,
    amplitude: float = 1.0,
    phase: float = 0.0,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Generate a real-valued sinusoidal tone.

    Produces ``N = round(fs * duration)`` samples of
    ``x(t) = A * cos(2*pi*f*t + phase)`` at times ``t = n / fs`` for
    ``n = 0 .. N-1``.

    Args:
        fs: Sampling frequency in Hz. Must be positive.
        freq: Tone frequency in Hz. May be negative, but ``abs(freq)`` must
            not exceed the Nyquist limit ``fs / 2``.
        duration: Tone duration in seconds. Must be positive and long enough
            for at least one sample at ``fs``.
        amplitude: Peak amplitude ``A``. Must be non-negative.
        phase: Initial phase in radians.

    Returns:
        A tuple ``(time, samples)`` where ``time`` is the float64 array of
        sample times in seconds and ``samples`` is the float64 real tone.

    Raises:
        ValueError: If any argument is non-finite or violates the
            constraints listed above.
    """
    n_samples = _validate_tone_args(
        fs, freq, duration, amplitude, phase, allow_nyquist=True
    )
    time = np.arange(n_samples, dtype=np.float64) / fs
    samples = amplitude * np.cos(2.0 * np.pi * freq * time + phase)
    return time, samples


def generate_iq_tone(
    fs: float,
    freq: float,
    duration: float,
    amplitude: float = 1.0,
    phase: float = 0.0,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.complex128]]:
    """Generate a complex baseband IQ tone.

    Produces ``N = round(fs * duration)`` samples of
    ``x(t) = A * exp(j*(2*pi*f*t + phase))`` at times ``t = n / fs`` for
    ``n = 0 .. N-1``. A negative ``freq`` generates the conjugate
    (lower-sideband) tone.

    Args:
        fs: Sampling frequency in Hz. Must be positive.
        freq: Tone frequency in Hz. May be negative, but ``abs(freq)`` must
            be strictly below the Nyquist limit ``fs / 2``; at Nyquist the
            sign of the frequency is not resolvable, so it is rejected.
        duration: Tone duration in seconds. Must be positive and long enough
            for at least one sample at ``fs``.
        amplitude: Envelope magnitude ``A``. Must be non-negative.
        phase: Initial phase in radians.

    Returns:
        A tuple ``(time, samples)`` where ``time`` is the float64 array of
        sample times in seconds and ``samples`` is the complex128 IQ tone.

    Raises:
        ValueError: If any argument is non-finite or violates the
            constraints listed above.
    """
    n_samples = _validate_tone_args(
        fs, freq, duration, amplitude, phase, allow_nyquist=False
    )
    time = np.arange(n_samples, dtype=np.float64) / fs
    samples = amplitude * np.exp(1j * (2.0 * np.pi * freq * time + phase))
    return time, samples
