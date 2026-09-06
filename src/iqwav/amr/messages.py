"""Harness-only analog message generators.

These helpers are not production modulators. They build real messages
that satisfy the analog modulator contract ``|m[n]| <= 1``. The same
families are used for AM, FM, and PM.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

__all__ = [
    "MESSAGE_FAMILIES",
    "generate_analog_message",
]

MESSAGE_FAMILIES = ("tone", "multi_tone", "bandlimited")
_N_MULTI_TONES = 3


def _peak_normalize(message: np.ndarray) -> npt.NDArray[np.float64]:
    """Scale a real message so its peak absolute value is 1, then clip."""
    peak = float(np.max(np.abs(message)))
    if peak == 0.0:
        return message
    scaled = message / peak
    return np.clip(scaled, -1.0, 1.0)


def _draw_freq_norm(
    rng: np.random.Generator,
    freq_norm_range: tuple[float, float],
) -> float:
    low, high = freq_norm_range
    return float(rng.uniform(low, high))


def generate_analog_message(
    n_samples: int,
    fs: float,
    family: str,
    rng: np.random.Generator,
    freq_norm_range: tuple[float, float],
) -> tuple[npt.NDArray[np.float64], tuple[float, ...]]:
    """Generate one normalized analog message.

    Args:
        n_samples: Message length. Must be >= 1.
        fs: Sampling frequency in Hz. Must be positive. Used only for
            bandlimited cutoff conversion; tone frequencies are drawn as
            fractions of ``fs``.
        family: One of ``tone``, ``multi_tone``, ``bandlimited``.
        rng: Payload/message RNG stream. Not shared with AWGN.
        freq_norm_range: Inclusive-open uniform range for ``f / fs``
            (tone frequencies or bandlimited cutoff). Must lie in
            ``(0, 0.5)``.

    Returns:
        ``(message, freq_norm)`` where ``message`` is float64 with
        ``|m[n]| <= 1`` and ``freq_norm`` is the drawn ``f/fs`` tuple.

    Raises:
        ValueError: If arguments are invalid.
    """
    if not isinstance(n_samples, (int, np.integer)) or isinstance(
        n_samples, bool
    ):
        raise ValueError(f"n_samples must be an integer >= 1, got {n_samples!r}.")
    if n_samples < 1:
        raise ValueError(f"n_samples must be an integer >= 1, got {n_samples!r}.")
    if family not in MESSAGE_FAMILIES:
        raise ValueError(
            f"family must be one of {MESSAGE_FAMILIES}, got {family!r}."
        )
    if not isinstance(rng, np.random.Generator):
        raise ValueError(
            f"rng must be a numpy.random.Generator, got {type(rng).__name__}."
        )
    low, high = freq_norm_range
    if not 0.0 < low <= high < 0.5:
        raise ValueError(
            f"freq_norm_range must satisfy 0 < low <= high < 0.5, "
            f"got {freq_norm_range!r}."
        )
    if not np.isfinite(fs) or fs <= 0.0:
        raise ValueError(f"fs must be positive and finite, got {fs!r}.")

    n = np.arange(n_samples, dtype=np.float64)
    if family == "tone":
        freq_norm = _draw_freq_norm(rng, freq_norm_range)
        phase = float(rng.uniform(0.0, 2.0 * np.pi))
        message = np.cos(2.0 * np.pi * freq_norm * n + phase)
        freqs = (freq_norm,)
    elif family == "multi_tone":
        freqs_list = []
        message = np.zeros(n_samples, dtype=np.float64)
        for _ in range(_N_MULTI_TONES):
            freq_norm = _draw_freq_norm(rng, freq_norm_range)
            phase = float(rng.uniform(0.0, 2.0 * np.pi))
            message += np.cos(2.0 * np.pi * freq_norm * n + phase)
            freqs_list.append(freq_norm)
        freqs = tuple(freqs_list)
    else:
        cutoff_norm = _draw_freq_norm(rng, freq_norm_range)
        white = rng.normal(0.0, 1.0, n_samples)
        spectrum = np.fft.rfft(white)
        freq_bins = np.fft.rfftfreq(n_samples, d=1.0 / fs)
        spectrum[freq_bins > cutoff_norm * fs] = 0.0
        message = np.fft.irfft(spectrum, n=n_samples)
        freqs = (cutoff_norm,)
    return _peak_normalize(np.asarray(message, dtype=np.float64)), freqs
