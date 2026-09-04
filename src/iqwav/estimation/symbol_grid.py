"""Rectangular symbol-grid estimation via transition-residue concentration."""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from .occupied_band import _validate_real_scalar

__all__ = ["RectangularSymbolGridEstimate", "estimate_rectangular_symbol_grid"]

_MIN_SAMPLES = 9
_MIN_PERIODS = 4
_MIN_EFFECTIVE_TRANSITIONS = 2.0
_DIVISOR_DOMINANCE_MARGIN = 0.05


@dataclass(frozen=True)
class RectangularSymbolGridEstimate:
    """Result of a rectangular symbol-grid estimate.

    Attributes:
        symbol_rate_hz: Estimated symbol rate in symbols per second,
            equal to ``fs / samples_per_symbol`` exactly.
        samples_per_symbol: Estimated symbol period in samples. Always an
            integer >= 2: the analysis grid is the integer-period grid,
            which is exactly the grid the rectangular waveform generators
            produce.
        boundary_offset: Estimated symbol-boundary phase: symbol-start
            sample indices are congruent to this value modulo
            ``samples_per_symbol``. It is a block-level symbol-boundary
            phase estimate, NOT timing recovery.
        quality: Chance-corrected concentration of the transition profile
            on the selected boundary phase,
            ``(concentration - 1 / P) / (1 - 1 / P)``, at most 1.0. It is
            the fraction of total first-difference magnitude attributable
            to boundary-aligned symbol transitions rather than to
            intra-symbol variation (noise, residual carrier rotation).
            1.0 means a noiseless rectangular waveform; values near 0
            mean no periodic transition structure was found.
        concentration: Raw, uncorrected fraction of total first-difference
            magnitude falling on the selected boundary phase, in
            ``[0, 1]``. Structureless input tends to ``1 / P`` by chance,
            which is why ``quality`` is the chance-corrected figure.
        symbol_count: Number of complete symbol periods the block spans
            at the estimated period and phase,
            ``(len(samples) - boundary_offset) // samples_per_symbol``.
        effective_transitions: Participation ratio
            ``(sum d)**2 / sum(d**2)`` of the boundary-aligned transition
            magnitudes: a threshold-free effective count of how many
            symbol boundaries carried observable transition energy. It
            guards against a block containing effectively a single
            transition, where every candidate period fits equally well.
        searched_samples_per_symbol: The ``(min, max)`` candidate periods
            actually searched, after the requested maximum was capped to
            what the block length can support.
    """

    symbol_rate_hz: float
    samples_per_symbol: int
    boundary_offset: int
    quality: float
    concentration: float
    symbol_count: int
    effective_transitions: float
    searched_samples_per_symbol: tuple[int, int]


def _validate_grid_samples(samples: np.ndarray) -> np.ndarray:
    """Validate 1-D finite numeric samples with at least 9 values."""
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
    if samples.shape[0] < _MIN_SAMPLES:
        raise ValueError(
            f"samples must contain at least {_MIN_SAMPLES} values to "
            f"observe {_MIN_PERIODS} periods of the shortest supported "
            f"symbol period, got {samples.shape[0]}."
        )
    if not np.all(np.isfinite(samples)):
        raise ValueError("samples must contain only finite values.")
    return samples


def _validate_sps_bound(value: object, name: str, minimum: int) -> int:
    """Validate one non-bool integer search bound with a minimum."""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer, got {value!r}.")
    if int(value) < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value!r}.")
    return int(value)


def _transition_profile(
    values: np.ndarray,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int64], float]:
    """Return transition magnitudes, their sample indices, and the total.

    ``magnitudes[i] = abs(values[i + 1] - values[i])`` is indexed by the
    sample that starts the potential new symbol, ``indices[i] = i + 1``,
    so a symbol boundary at sample ``n`` contributes to residue class
    ``n % P``. The input array is only read; the returned arrays are new.
    """
    magnitudes = np.abs(np.diff(values)).astype(np.float64)
    indices = np.arange(1, values.size, dtype=np.int64)
    return magnitudes, indices, float(np.sum(magnitudes))


def _score_period(
    magnitudes: npt.NDArray[np.float64],
    indices: npt.NDArray[np.int64],
    total: float,
    period: int,
) -> tuple[float, float, int, float]:
    """Score one candidate period and return its statistics.

    Returns ``(quality, concentration, boundary_offset,
    effective_transitions)`` for the best residue class modulo
    ``period``.
    """
    classes = indices % period
    binned = np.bincount(classes, weights=magnitudes, minlength=period)
    offset = int(np.argmax(binned))
    concentration = min(float(binned[offset]) / total, 1.0)
    chance = 1.0 / period
    quality = (concentration - chance) / (1.0 - chance)
    aligned = magnitudes[classes == offset]
    energy = float(np.sum(aligned * aligned))
    effective = float(binned[offset]) ** 2 / energy if energy > 0.0 else 0.0
    return quality, concentration, offset, effective


def estimate_rectangular_symbol_grid(
    samples: np.ndarray,
    fs: float,
    *,
    min_sps: int = 2,
    max_sps: int = 64,
    quality_ratio: float = 0.75,
    min_quality: float = 0.02,
) -> RectangularSymbolGridEstimate:
    """Estimate the symbol grid of a rectangular-pulse waveform.

    Definition:
        With ``d[n] = abs(x[n] - x[n - 1])``, the first-difference
        magnitudes of a rectangular-pulse waveform form an impulse train
        whose impulses all land on one residue class of the sample index
        modulo the true symbol period. For every integer candidate period
        ``P`` in the search range, ``d`` is grouped by
        ``sample_index % P``, the strongest residue bin's share of the
        total transition magnitude is the ``concentration``, and
        ``quality = (concentration - 1 / P) / (1 - 1 / P)`` corrects it
        for the ``1 / P`` level reached by chance.

        Integer sub-multiples of the true period concentrate the same
        impulses just as well and therefore tie at the same quality,
        while multiples of it split the impulses across several classes
        and score at most about ``1 / m`` of the best. The estimate is
        therefore the LARGEST candidate period whose quality is at least
        ``quality_ratio`` times the best quality observed, subject to a
        divisor-dominance guard that rejects a candidate when a proper
        divisor explains the transitions substantially better. The
        winning residue class is the ``boundary_offset``, and
        ``symbol_rate_hz = fs / samples_per_symbol``.

    Boundary offset:
        ``boundary_offset`` means that symbol-start sample indices are
        congruent to ``boundary_offset`` modulo ``samples_per_symbol``.
        It is 0 for the canonical waveform-generator output, and equals
        ``(-crop) % samples_per_symbol`` for a waveform cropped by
        ``crop`` samples. It is a single block-level symbol-boundary
        phase estimate, NOT timing recovery: nothing is tracked,
        interpolated, or corrected.

    Assumptions:
        This is NOT a general blind baud estimator. It assumes
        rectangular piecewise-constant symbols, an integer number of
        samples per symbol, a stationary symbol grid throughout the
        block, enough observable transitions, and a known ``fs``. It
        does NOT support RRC or other pulse-shaped signals, fractional
        samples per symbol, timing drift, timing recovery, matched
        filtering, or arbitrary real-world modulation.

    Identifiability limitations (genuine, not tuned away):
        If the symbol stream only ever changes at every m-th true
        boundary (long runs of identical symbols), the observable
        transition period really is ``m * samples_per_symbol`` and that
        is what is returned. If that observable period lies outside the
        search range, an in-range divisor of it may be returned, because
        divisors are genuinely consistent with the observed transitions.
        This is an information/identifiability limitation of the input,
        not a defect to be disguised with heuristic tuning. A block
        containing effectively a single observable transition is
        rejected: every candidate period explains one lone impulse
        equally well. A residual carrier offset makes the waveform
        non-constant within a symbol and lowers ``quality``; small
        offsets do not move the estimate, large ones eventually destroy
        the structure.

    Args:
        samples: 1-D real or complex finite numeric samples with at
            least 9 values, containing observable transitions. Never
            modified.
        fs: Positive finite real sampling rate in Hz. Not inferred.
        min_sps: Smallest candidate symbol period in samples. Must be a
            non-bool integer >= 2.
        max_sps: Largest candidate symbol period in samples. Must be a
            non-bool integer >= ``min_sps``. It is capped to
            ``(N - 1) // 4`` so that at least four candidate periods
            fit in the block; the range actually searched is reported in
            the result.
        quality_ratio: How close to the best observed quality a longer
            candidate period must come to be preferred, in ``(0, 1]``.
            Lower values prefer multiples of the true period; 1.0
            demands an exact tie.
        min_quality: Smallest returned quality accepted, in ``[0, 1)``.
            Pass 0.0 to disable the check and inspect ``quality``
            directly.

    Returns:
        The :class:`RectangularSymbolGridEstimate` for the block.

    Raises:
        ValueError: If any argument is invalid; the block is too short
            for the requested range; the block is constant or all zero;
            no candidate period concentrates transitions above chance;
            the selected quality is below ``min_quality``; or the block
            contains effectively a single transition.
    """
    samples = _validate_grid_samples(samples)
    fs = _validate_real_scalar(fs, "fs")
    if fs <= 0:
        raise ValueError(f"fs must be positive and finite, got {fs!r}.")
    period_min = _validate_sps_bound(min_sps, "min_sps", 2)
    period_max = _validate_sps_bound(max_sps, "max_sps", period_min)
    ratio = _validate_real_scalar(quality_ratio, "quality_ratio")
    if not 0.0 < ratio <= 1.0:
        raise ValueError(
            f"quality_ratio must satisfy 0 < quality_ratio <= 1, "
            f"got {quality_ratio!r}."
        )
    quality_floor = _validate_real_scalar(min_quality, "min_quality")
    if not 0.0 <= quality_floor < 1.0:
        raise ValueError(
            f"min_quality must satisfy 0 <= min_quality < 1, "
            f"got {min_quality!r}."
        )

    n = samples.shape[0]
    supported_max = (n - 1) // _MIN_PERIODS
    searched_max = min(period_max, supported_max)
    if searched_max < period_min:
        raise ValueError(
            f"samples contains {n} values, which supports candidate "
            f"symbol periods of at most {supported_max} sample(s) while "
            f"observing {_MIN_PERIODS} periods; supply a longer block or "
            f"lower min_sps."
        )

    magnitudes, indices, total = _transition_profile(samples)
    if total <= 0.0:
        raise ValueError(
            "samples never change value (constant or all-zero block), so "
            "the block contains no symbol transitions and no symbol grid "
            "can be estimated."
        )

    scores = [
        _score_period(magnitudes, indices, total, period)
        for period in range(period_min, searched_max + 1)
    ]
    qualities = [score[0] for score in scores]
    best_quality = max(qualities)
    if best_quality <= 0.0:
        raise ValueError(
            "no candidate symbol period concentrates transitions above "
            "the level expected by chance; the block shows no "
            "rectangular-pulse symbol structure in the searched range "
            f"[{period_min}, {searched_max}] samples per symbol."
        )

    threshold = ratio * best_quality
    eligible = []
    for index, quality in enumerate(qualities):
        candidate = period_min + index
        if quality < threshold:
            continue
        dominated = any(
            candidate % divisor == 0
            and divisor >= period_min
            and qualities[divisor - period_min]
            > quality + _DIVISOR_DOMINANCE_MARGIN
            for divisor in range(period_min, candidate)
        )
        if not dominated:
            eligible.append(index)

    selected = max(eligible)
    quality, concentration, offset, effective = scores[selected]
    period = period_min + selected
    if quality < quality_floor:
        raise ValueError(
            f"the estimated symbol period of {period} sample(s) reaches "
            f"quality {quality:.4g}, below min_quality={quality_floor!r}; "
            "the block shows no usable periodic symbol-transition "
            f"structure in [{period_min}, {searched_max}] samples per "
            "symbol."
        )
    if effective < _MIN_EFFECTIVE_TRANSITIONS:
        raise ValueError(
            f"the selected symbol period of {period} sample(s) is "
            f"supported by only {effective:.3g} effective transition(s); "
            "a block with a single observable transition is explained "
            "equally well by every candidate period, so the symbol grid "
            "is not identifiable."
        )

    return RectangularSymbolGridEstimate(
        symbol_rate_hz=fs / period,
        samples_per_symbol=period,
        boundary_offset=offset,
        quality=quality,
        concentration=concentration,
        symbol_count=(n - offset) // period,
        effective_transitions=effective,
        searched_samples_per_symbol=(period_min, searched_max),
    )
