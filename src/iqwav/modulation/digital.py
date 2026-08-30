"""Symbol-level digital modulation utilities."""

import numpy as np
import numpy.typing as npt

__all__ = ["bpsk_modulate", "qpsk_modulate"]


def _validate_bits(bits: np.ndarray) -> npt.NDArray[np.integer]:
    """Validate a 1-D integer array of bits with values 0 and 1 only."""
    bits = np.asarray(bits)
    if bits.ndim != 1:
        raise ValueError(
            f"bits must be one-dimensional, got shape {bits.shape}."
        )
    if bits.size == 0:
        raise ValueError("bits must contain at least one value.")
    if not (np.issubdtype(bits.dtype, np.integer) or bits.dtype == np.bool_):
        raise ValueError(
            f"bits must be an integer or boolean array of 0s and 1s, "
            f"got dtype {bits.dtype}."
        )
    if not np.all((bits == 0) | (bits == 1)):
        raise ValueError("bits must contain only values 0 and 1.")
    return bits


def bpsk_modulate(bits: np.ndarray) -> npt.NDArray[np.complex128]:
    """Map bits to BPSK symbols.

    Maps bit 0 to ``+1 + 0j`` and bit 1 to ``-1 + 0j``.

    Args:
        bits: 1-D array-like of bits. Must be non-empty, integer or
            boolean typed, and contain only values 0 and 1.

    Returns:
        A 1-D complex128 array with one symbol per input bit.

    Raises:
        ValueError: If ``bits`` is not a non-empty 1-D integer array of
            0s and 1s.
    """
    bits = _validate_bits(bits)
    return np.where(bits == 0, 1.0 + 0.0j, -1.0 + 0.0j)


_QPSK_CONSTELLATION = np.array(
    [
        (1.0 + 1.0j) / np.sqrt(2.0),
        (-1.0 + 1.0j) / np.sqrt(2.0),
        (1.0 - 1.0j) / np.sqrt(2.0),
        (-1.0 - 1.0j) / np.sqrt(2.0),
    ],
    dtype=np.complex128,
)


def qpsk_modulate(bits: np.ndarray) -> npt.NDArray[np.complex128]:
    """Map bit pairs to QPSK symbols using Gray mapping.

    Maps bit pairs to unit-magnitude symbols as ``00 -> (+1+1j)/sqrt(2)``,
    ``01 -> (-1+1j)/sqrt(2)``, ``11 -> (-1-1j)/sqrt(2)``, and
    ``10 -> (+1-1j)/sqrt(2)``, where the first bit of a pair selects the
    quadrature sign and the second bit selects the in-phase sign.

    Args:
        bits: 1-D array-like of bits. Must be non-empty, integer or boolean
            typed, contain only values 0 and 1, and have an even length.

    Returns:
        A 1-D complex128 array with one symbol per two input bits.

    Raises:
        ValueError: If ``bits`` is invalid as for :func:`bpsk_modulate` or
            its length is not even.
    """
    bits = _validate_bits(bits)
    if bits.size % 2 != 0:
        raise ValueError(
            f"qpsk_modulate requires an even number of bits, got {bits.size}."
        )
    pairs = bits.reshape(-1, 2)
    index = 2 * pairs[:, 0].astype(np.int64) + pairs[:, 1]
    return _QPSK_CONSTELLATION[index]
