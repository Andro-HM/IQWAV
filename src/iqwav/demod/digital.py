"""Known-timing hard-decision digital demodulation."""

import numpy as np
import numpy.typing as npt

__all__ = ["bpsk_demodulate", "qpsk_demodulate"]


def _validate_demod_args(
    samples: np.ndarray,
    samples_per_symbol: int,
    require_complex: bool,
) -> np.ndarray:
    """Validate demodulator arguments and return samples as an ndarray."""
    if not isinstance(samples_per_symbol, (int, np.integer)) or samples_per_symbol < 1:
        raise ValueError(
            f"samples_per_symbol must be an integer >= 1, "
            f"got {samples_per_symbol!r}."
        )
    samples = np.asarray(samples)
    if samples.ndim != 1:
        raise ValueError(
            f"samples must be one-dimensional, got shape {samples.shape}."
        )
    if samples.size == 0:
        raise ValueError("samples must contain at least one value.")
    if not np.all(np.isfinite(samples)):
        raise ValueError("samples must contain only finite values.")
    if require_complex and not np.iscomplexobj(samples):
        raise ValueError("samples must be complex-valued IQ data.")
    if samples.size % samples_per_symbol != 0:
        raise ValueError(
            f"len(samples) = {samples.size} must be exactly divisible by "
            f"samples_per_symbol = {samples_per_symbol}."
        )
    return samples


def _block_averages(
    samples: np.ndarray,
    samples_per_symbol: int,
) -> npt.NDArray[np.complexfloating]:
    """Average consecutive samples_per_symbol-sized blocks."""
    blocks = samples.reshape(-1, samples_per_symbol)
    return blocks.mean(axis=1)


def bpsk_demodulate(
    samples: np.ndarray,
    samples_per_symbol: int,
) -> npt.NDArray[np.int64]:
    """Hard-decision BPSK demodulation with perfectly known symbol timing.

    Averages every consecutive ``samples_per_symbol`` block to one symbol
    value and maps ``real(symbol) >= 0`` to bit 0 and ``real(symbol) < 0``
    to bit 1, inverting :func:`iqwav.modulation.bpsk_modulate`.

    Args:
        samples: 1-D real or complex sampled waveform. Must be non-empty
            with only finite values, and its length must be exactly
            divisible by ``samples_per_symbol``.
        samples_per_symbol: Number of samples per symbol. Must be an
            integer >= 1.

    Returns:
        A 1-D int64 array with one recovered bit per symbol.

    Raises:
        ValueError: If any argument violates the constraints above.
    """
    samples = _validate_demod_args(samples, samples_per_symbol, require_complex=False)
    averages = _block_averages(samples, samples_per_symbol)
    return (np.real(averages) < 0).astype(np.int64)


def qpsk_demodulate(
    samples: np.ndarray,
    samples_per_symbol: int,
) -> npt.NDArray[np.int64]:
    """Hard-decision QPSK demodulation with perfectly known symbol timing.

    Averages every consecutive ``samples_per_symbol`` block to one complex
    symbol value and inverts the Gray mapping of
    :func:`iqwav.modulation.qpsk_modulate`: ``I >= 0, Q >= 0 -> 00``,
    ``I < 0, Q >= 0 -> 01``, ``I < 0, Q < 0 -> 11``, ``I >= 0, Q < 0 -> 10``,
    where zero parts count as non-negative.

    Args:
        samples: 1-D complex sampled waveform. Must be non-empty with only
            finite values, and its length must be exactly divisible by
            ``samples_per_symbol``.
        samples_per_symbol: Number of samples per symbol. Must be an
            integer >= 1.

    Returns:
        A 1-D int64 array with two recovered bits per symbol, ordered as
        bit pairs matching the modulator input.

    Raises:
        ValueError: If any argument violates the constraints above,
            including real-valued input.
    """
    samples = _validate_demod_args(samples, samples_per_symbol, require_complex=True)
    averages = _block_averages(samples, samples_per_symbol)
    first_bit = (np.imag(averages) < 0).astype(np.int64)
    second_bit = (np.real(averages) < 0).astype(np.int64)
    return np.column_stack((first_bit, second_bit)).reshape(-1)
