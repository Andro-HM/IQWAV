"""Receiver-side integer symbol-timing correction."""

import numpy as np

__all__ = ["correct_symbol_timing"]


def _validate_sps(value: object) -> int:
    """Validate a non-bool integer samples-per-symbol value >= 2."""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(
            f"samples_per_symbol must be an integer >= 2, got {value!r}."
        )
    sps = int(value)
    if sps < 2:
        raise ValueError(
            f"samples_per_symbol must be an integer >= 2, got {value!r}."
        )
    return sps


def _validate_offset(value: object, sps: int) -> int:
    """Validate a non-bool integer boundary offset in ``[0, sps)``."""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(
            f"boundary_offset must be an integer in [0, {sps}), got {value!r}."
        )
    offset = int(value)
    if not 0 <= offset < sps:
        raise ValueError(
            f"boundary_offset must be an integer in [0, {sps}), got {value!r}."
        )
    return offset


def correct_symbol_timing(
    samples: np.ndarray,
    samples_per_symbol: int,
    boundary_offset: int,
) -> np.ndarray:
    """Correct a caller-supplied integer symbol-boundary offset.

    Convention:
        ``boundary_offset`` is the number of leading samples to discard
        so the corrected output begins at the next complete symbol
        boundary. Trailing samples that do not fill a whole symbol are
        trimmed. For ``observed = clean[C:]`` and
        ``boundary_offset = (-C) % samples_per_symbol``, the returned
        block is the corresponding complete-symbol slice of ``clean``.

        There is no circular shift, interpolation, resampling, or
        timing loop. Only the supplied integer offset is applied; this
        function does not estimate timing.

    Scope:
        This ONLY aligns a rectangular integer-SPS block to a
        caller-supplied residue. It is NOT baud estimation, NOT
        fractional timing recovery, NOT a Gardner or Mueller-Muller
        tracker, NOT carrier recovery, and NOT demodulation.

    Args:
        samples: 1-D real or complex finite numeric samples. Never
            mutated.
        samples_per_symbol: Known integer samples per symbol. Must be a
            non-bool integer >= 2.
        boundary_offset: Integer leading samples to discard, in
            ``[0, samples_per_symbol)``.

    Returns:
        A new array of complete symbol-aligned samples whose length is
        a positive multiple of ``samples_per_symbol``. Dtype is
        preserved. This never returns an empty array.

    Raises:
        ValueError: If any argument is invalid, or fewer than one
            complete symbol remains after dropping the leading offset.
            An incomplete leftover is an error, not an empty result.
    """
    samples = np.asarray(samples)
    if samples.ndim != 1:
        raise ValueError(
            f"samples must be one-dimensional, got shape {samples.shape}."
        )
    if samples.size == 0:
        raise ValueError("samples must contain at least one value.")
    if samples.dtype.kind not in "fiuc":
        raise ValueError(
            f"samples must be real or complex numeric data, "
            f"got dtype {samples.dtype}."
        )
    if not np.all(np.isfinite(samples)):
        raise ValueError("samples must contain only finite values.")
    sps = _validate_sps(samples_per_symbol)
    offset = _validate_offset(boundary_offset, sps)
    n_complete = (samples.size - offset) // sps
    if n_complete < 1:
        raise ValueError(
            f"samples length {samples.size} with boundary_offset={offset} "
            f"leaves no complete symbol of {sps} sample(s)."
        )
    aligned = np.array(samples[offset : offset + n_complete * sps], copy=True)
    return aligned
