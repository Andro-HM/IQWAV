"""Symbol-to-sampled-waveform conversion using rectangular pulses."""

import numpy as np

from .digital import bpsk_modulate, qpsk_modulate

__all__ = ["bpsk_waveform", "qpsk_waveform", "symbols_to_samples"]


def _validate_samples_per_symbol(samples_per_symbol: int) -> None:
    """Validate the samples-per-symbol factor."""
    if not isinstance(samples_per_symbol, (int, np.integer)) or samples_per_symbol < 1:
        raise ValueError(
            f"samples_per_symbol must be an integer >= 1, "
            f"got {samples_per_symbol!r}."
        )


def symbols_to_samples(symbols: np.ndarray, samples_per_symbol: int) -> np.ndarray:
    """Repeat each symbol to form a rectangular-pulse sampled waveform.

    Args:
        symbols: 1-D real or complex symbol array. Must be non-empty with
            only finite values.
        samples_per_symbol: Repetition factor per symbol. Must be an
            integer >= 1.

    Returns:
        The sampled waveform of length ``len(symbols) *
        samples_per_symbol`` where each symbol is repeated
        ``samples_per_symbol`` times. Real input stays real, complex input
        stays complex.

    Raises:
        ValueError: If ``symbols`` is not a non-empty 1-D array of finite
            values, or ``samples_per_symbol`` is not an integer >= 1.
    """
    _validate_samples_per_symbol(samples_per_symbol)
    symbols = np.asarray(symbols)
    if symbols.ndim != 1:
        raise ValueError(
            f"symbols must be one-dimensional, got shape {symbols.shape}."
        )
    if symbols.size == 0:
        raise ValueError("symbols must contain at least one value.")
    if not np.all(np.isfinite(symbols)):
        raise ValueError("symbols must contain only finite values.")
    return np.repeat(symbols, samples_per_symbol)


def bpsk_waveform(
    bits: np.ndarray,
    samples_per_symbol: int,
) -> np.ndarray:
    """Convert bits to a rectangular-pulse BPSK sampled waveform.

    Maps ``bits`` with :func:`bpsk_modulate` and repeats each symbol
    ``samples_per_symbol`` times.

    Args:
        bits: 1-D array-like of bits validated as in
            :func:`bpsk_modulate`.
        samples_per_symbol: Repetition factor per symbol. Must be an
            integer >= 1.

    Returns:
        The complex128 sampled waveform of length
        ``len(bits) * samples_per_symbol``.

    Raises:
        ValueError: If ``bits`` or ``samples_per_symbol`` is invalid.
    """
    _validate_samples_per_symbol(samples_per_symbol)
    symbols = bpsk_modulate(bits)
    return symbols_to_samples(symbols, samples_per_symbol)


def qpsk_waveform(
    bits: np.ndarray,
    samples_per_symbol: int,
) -> np.ndarray:
    """Convert bit pairs to a rectangular-pulse QPSK sampled waveform.

    Maps ``bits`` with :func:`qpsk_modulate` and repeats each symbol
    ``samples_per_symbol`` times.

    Args:
        bits: 1-D array-like of bits validated as in
            :func:`qpsk_modulate`.
        samples_per_symbol: Repetition factor per symbol. Must be an
            integer >= 1.

    Returns:
        The complex128 sampled waveform of length
        ``(len(bits) // 2) * samples_per_symbol``.

    Raises:
        ValueError: If ``bits`` or ``samples_per_symbol`` is invalid.
    """
    _validate_samples_per_symbol(samples_per_symbol)
    symbols = qpsk_modulate(bits)
    return symbols_to_samples(symbols, samples_per_symbol)
