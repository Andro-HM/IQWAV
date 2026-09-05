"""Known-SPS integer symbol-timing estimation for rectangular PSK."""

from dataclasses import dataclass

import numpy as np

from .symbol_grid import _score_period, _transition_profile, _validate_sps_bound

__all__ = ["SymbolTimingEstimate", "estimate_symbol_timing"]


@dataclass(frozen=True)
class SymbolTimingEstimate:
    """Known-SPS integer symbol-boundary estimate.

    ``boundary_offset`` is the number of leading samples to discard so
    the remaining block begins at the next complete symbol boundary,
    in ``[0, samples_per_symbol)``. ``quality`` is the chance-corrected
    concentration of first-difference magnitude on that residue,
    ``(concentration - 1/SPS) / (1 - 1/SPS)``. It is a diagnostic only:
    not a calibrated confidence, probability, or SNR.
    """

    boundary_offset: int
    quality: float


def _validate_timing_samples(samples: np.ndarray) -> np.ndarray:
    """Validate 1-D finite numeric samples with at least 2 values."""
    samples = np.asarray(samples)
    if samples.ndim != 1:
        raise ValueError(
            f"samples must be one-dimensional, got shape {samples.shape}."
        )
    if samples.dtype.kind not in "fiuc":
        raise ValueError(
            f"samples must be real or complex numeric data, "
            f"got dtype {samples.dtype}."
        )
    if samples.shape[0] < 2:
        raise ValueError(
            f"samples must contain at least 2 values to observe a "
            f"symbol transition, got {samples.shape[0]}."
        )
    if not np.all(np.isfinite(samples)):
        raise ValueError("samples must contain only finite values.")
    return samples


def estimate_symbol_timing(
    samples: np.ndarray,
    samples_per_symbol: int,
) -> SymbolTimingEstimate:
    """Estimate an integer symbol-boundary offset given known SPS.

    For a rectangular piecewise-constant waveform with known integer
    ``samples_per_symbol`` (``SPS``), first-difference magnitudes

        d[n] = |x[n] - x[n-1]|

    land on one residue class of the sample index modulo ``SPS``. The
    estimator reuses the accepted transition-residue binning from
    :func:`iqwav.estimation.estimate_rectangular_symbol_grid` at that
    single known period and returns the strongest residue as
    ``boundary_offset``.

    For an aligned generator waveform cropped by ``C`` samples,
    ``observed = clean[C:]``, the true offset is ``(-C) % SPS``.
    Correcting by discarding that many leading samples, then trimming
    any trailing incomplete symbol, aligns the block for the existing
    known-timing demodulators.

    This is block-level integer timing recovery, not a timing loop,
    interpolator, resampler, Gardner/Mueller-Muller tracker, or baud
    estimator. ``SPS`` is caller-supplied and is not searched. There is
    no ``min_quality`` reject threshold: ``quality`` is reported only.
    The only identifiability reject is zero usable transition evidence
    (a constant / all-zero block). A single clean transition is
    identifiable because the period is already known.

    Args:
        samples: 1-D real or complex finite samples with at least 2
            values. Never modified.
        samples_per_symbol: Known integer samples per symbol. Must be a
            non-bool integer >= 2.

    Returns:
        The :class:`SymbolTimingEstimate` for the block.

    Raises:
        ValueError: If any argument is invalid, or the block contains
            no usable transition evidence.
    """
    samples = _validate_timing_samples(samples)
    sps = _validate_sps_bound(samples_per_symbol, "samples_per_symbol", 2)
    magnitudes, indices, total = _transition_profile(samples)
    if total <= 0.0:
        raise ValueError(
            "samples contain no usable symbol-transition evidence; "
            "a constant or all-zero block has no identifiable timing "
            "offset when samples_per_symbol is already known."
        )
    quality, _concentration, offset, _effective = _score_period(
        magnitudes, indices, total, sps
    )
    return SymbolTimingEstimate(
        boundary_offset=int(offset),
        quality=float(quality),
    )
