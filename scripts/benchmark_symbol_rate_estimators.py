"""Head-to-head benchmark: HM vs Sayan rectangular-pulse symbol-rate estimators.

Compares HM's production ``iqwav.estimation.estimate_symbol_rate``
(transition-energy autocorrelation) against the frozen Sayan snapshot
estimator (transition-magnitude residue-class concentration), reproduced
privately below as ``_sayan_estimate_symbol_rate`` with logic kept
verbatim from ``sayan-snapshot-9e927de:src/iqwav/estimation/symbol_rate.py``
(identifiers renamed to private names only). Sayan's estimator is NOT
added to the package API and no production code is modified.

Fairness rules applied:
- Both estimators receive exactly the same sample arrays.
- Both search the same default candidate range [2, 64] samples/symbol,
  which includes every true SPS tested.
- Single globally documented adjustment: HM's ``max_sps`` is capped to
  ``len(samples) - 2`` (its own validation requirement on short blocks),
  mirroring Sayan's internal cap of ``(N - 1) // 4``. No per-case tuning.
- Each estimator keeps its own default threshold philosophy
  (HM ``min_score=0.10``; Sayan ``quality_ratio=0.75``,
  ``min_quality=0.02``).
- Outcomes are classified as correct / wrong / rejected (ValueError),
  never silently counting a rejection as a wrong estimate.
- HM ``score`` and Sayan ``quality`` are different statistics and are
  recorded separately, never compared numerically.

The boundary-offset check compares Sayan's ``boundary_offset`` against
the known crop residue. It is a block-level symbol-boundary phase
estimate, NOT timing recovery.
"""

import math
from dataclasses import dataclass
from itertools import count

import numpy as np
import numpy.typing as npt

from iqwav.dsp import add_awgn, apply_frequency_offset, apply_phase_offset
from iqwav.estimation import estimate_symbol_rate as hm_estimate_symbol_rate
from iqwav.modulation import bpsk_waveform, qpsk_waveform

FS = 96_000.0
TRUE_SPS_VALUES = [2, 3, 4, 5, 8, 12, 16, 24, 32]
DEFAULT_SYMBOL_COUNT = 4000
SHORT_SYMBOL_COUNTS = [32, 64, 128, 256, 1024]
SNR_DB_VALUES = [20.0, 10.0, 5.0, 0.0, -5.0]
CFO_FRACTION_MAGNITUDES = [0.0, 0.01, 0.05, 0.10, 0.20]
PHASES_RAD = [0.3, 1.2, 2.7]
AMPLITUDES = [0.1, 10.0]
PATHOLOGICAL_SPS_VALUES = [4, 8, 16]
HM_DEFAULT_MAX_SPS = 64
CLEAR_WIN_MARGIN_PCT = 10.0


# ---------------------------------------------------------------------------
# Private reproduction of Sayan's frozen estimator (logic verbatim,
# identifiers renamed). Not part of the package API.
# ---------------------------------------------------------------------------

_SAYAN_MIN_SAMPLES = 9
_SAYAN_MIN_PERIODS = 4
_SAYAN_MIN_EFFECTIVE_TRANSITIONS = 2.0
_SAYAN_DIVISOR_DOMINANCE_MARGIN = 0.05


@dataclass(frozen=True)
class _SayanSymbolRateEstimate:
    symbol_rate_hz: float
    samples_per_symbol: int
    sample_rate_hz: float
    symbol_rate_resolution_hz: float
    quality: float
    concentration: float
    boundary_offset: int
    symbol_count: int
    effective_transitions: float
    searched_samples_per_symbol: tuple[int, int]


def _sayan_validate_samples(samples: np.ndarray, sample_rate: float) -> np.ndarray:
    if not math.isfinite(sample_rate) or sample_rate <= 0.0:
        raise ValueError(
            f"sample_rate must be positive and finite, got {sample_rate!r}."
        )
    values = np.asarray(samples)
    if values.ndim != 1:
        raise ValueError(
            f"samples must be one-dimensional, got shape {values.shape}."
        )
    if values.size == 0:
        raise ValueError("samples must contain at least one value.")
    if values.size < _SAYAN_MIN_SAMPLES:
        raise ValueError(
            f"samples must contain at least {_SAYAN_MIN_SAMPLES} values to observe "
            f"{_SAYAN_MIN_PERIODS} periods of the shortest supported symbol period, "
            f"got {values.size}."
        )
    if not np.all(np.isfinite(values)):
        raise ValueError("samples must contain only finite values.")
    return values


def _sayan_validate_integer_period(value: object, *, name: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer, got {value!r}.")
    if int(value) < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value!r}.")
    return int(value)


def _sayan_validate_unit_fraction(
    value: object,
    *,
    name: str,
    lower: float,
    upper: float,
    include_lower: bool,
    include_upper: bool,
) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a real number, got {value!r}.") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite, got {value!r}.")
    low_ok = number >= lower if include_lower else number > lower
    high_ok = number <= upper if include_upper else number < upper
    if not (low_ok and high_ok):
        left = "[" if include_lower else "("
        right = "]" if include_upper else ")"
        raise ValueError(
            f"{name} must lie within {left}{lower}, {upper}{right}, "
            f"got {value!r}."
        )
    return number


def _sayan_transition_profile(
    values: np.ndarray,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int64], float]:
    magnitudes = np.abs(np.diff(values)).astype(np.float64)
    indices = np.arange(1, values.size, dtype=np.int64)
    return magnitudes, indices, float(np.sum(magnitudes))


def _sayan_score_period(
    magnitudes: npt.NDArray[np.float64],
    indices: npt.NDArray[np.int64],
    total: float,
    period: int,
) -> tuple[float, float, int, float]:
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


def _sayan_estimate_symbol_rate(
    samples: np.ndarray,
    sample_rate: float,
    *,
    min_samples_per_symbol: int = 2,
    max_samples_per_symbol: int = 64,
    quality_ratio: float = 0.75,
    min_quality: float = 0.02,
) -> _SayanSymbolRateEstimate:
    values = _sayan_validate_samples(samples, sample_rate)
    period_min = _sayan_validate_integer_period(
        min_samples_per_symbol, name="min_samples_per_symbol", minimum=2
    )
    period_max = _sayan_validate_integer_period(
        max_samples_per_symbol,
        name="max_samples_per_symbol",
        minimum=period_min,
    )
    ratio = _sayan_validate_unit_fraction(
        quality_ratio,
        name="quality_ratio",
        lower=0.0,
        upper=1.0,
        include_lower=False,
        include_upper=True,
    )
    quality_floor = _sayan_validate_unit_fraction(
        min_quality,
        name="min_quality",
        lower=0.0,
        upper=1.0,
        include_lower=True,
        include_upper=False,
    )

    supported_max = (values.size - 1) // _SAYAN_MIN_PERIODS
    searched_max = min(period_max, supported_max)
    if searched_max < period_min:
        raise ValueError(
            f"samples contains {values.size} values, which supports candidate "
            f"symbol periods of at most {supported_max} sample(s) while "
            f"observing {_SAYAN_MIN_PERIODS} periods, but min_samples_per_symbol is "
            f"{period_min}; supply a longer block or lower "
            "min_samples_per_symbol."
        )

    magnitudes, indices, total = _sayan_transition_profile(values)
    if total <= 0.0:
        raise ValueError(
            "samples never change value (constant or all-zero block), so the "
            "block contains no symbol transitions and no symbol rate can be "
            "estimated."
        )

    scores = [
        _sayan_score_period(magnitudes, indices, total, period)
        for period in range(period_min, searched_max + 1)
    ]
    qualities = [score[0] for score in scores]
    best_quality = max(qualities)
    if best_quality <= 0.0:
        raise ValueError(
            "no candidate symbol period concentrates transitions above the "
            "level expected by chance; the block shows no rectangular-pulse "
            "symbol structure in the searched range "
            f"[{period_min}, {searched_max}] samples per symbol."
        )

    threshold = ratio * best_quality

    eligible = []
    for index, quality in enumerate(qualities):
        period_candidate = period_min + index
        if quality < threshold:
            continue
        dominated = any(
            period_candidate % divisor == 0
            and divisor >= period_min
            and qualities[divisor - period_min]
            > quality + _SAYAN_DIVISOR_DOMINANCE_MARGIN
            for divisor in range(period_min, period_candidate)
        )
        if not dominated:
            eligible.append(index)

    selected = max(eligible)
    quality, concentration, offset, effective = scores[selected]
    period = period_min + selected
    if quality < quality_floor:
        raise ValueError(
            f"the estimated symbol period of {period} sample(s) reaches "
            f"quality {quality:.4g}, below min_quality={quality_floor!r}; the "
            "block shows no usable periodic symbol-transition structure in "
            f"[{period_min}, {searched_max}] samples per symbol."
        )
    if effective < _SAYAN_MIN_EFFECTIVE_TRANSITIONS:
        raise ValueError(
            f"the selected symbol period of {period} sample(s) is supported by "
            f"only {effective:.3g} effective transition(s); a block with a "
            "single observable transition is explained equally well by every "
            "candidate period, so the symbol period is not identifiable."
        )

    rate = float(sample_rate) / period
    return _SayanSymbolRateEstimate(
        symbol_rate_hz=rate,
        samples_per_symbol=period,
        sample_rate_hz=float(sample_rate),
        symbol_rate_resolution_hz=float(sample_rate) / (period * (period + 1)),
        quality=quality,
        concentration=concentration,
        boundary_offset=offset,
        symbol_count=(values.size - offset) // period,
        effective_transitions=effective,
        searched_samples_per_symbol=(period_min, searched_max),
    )


# ---------------------------------------------------------------------------
# Benchmark signal construction and trial execution.
# ---------------------------------------------------------------------------


@dataclass
class TrialRecord:
    modulation: str
    true_sps: int
    true_rate_hz: float
    case: str
    detail: str
    seed: int
    hm_outcome: str
    hm_sps: int | None
    hm_rate_hz: float | None
    hm_score: float | None
    sayan_outcome: str
    sayan_sps: int | None
    sayan_rate_hz: float | None
    sayan_quality: float | None
    sayan_boundary_offset: int | None
    expected_boundary_offset: int | None


def _random_bits(modulation: str, n_symbols: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n_bits = n_symbols if modulation == "BPSK" else 2 * n_symbols
    return rng.integers(0, 2, n_bits)


def _pattern_bits(modulation: str, n_symbols: int, pattern: str, seed: int) -> np.ndarray:
    if pattern == "all_identical":
        n_bits = n_symbols if modulation == "BPSK" else 2 * n_symbols
        return np.zeros(n_bits, dtype=np.int64)
    if pattern == "alternating":
        base = [0, 1] if modulation == "BPSK" else [0, 0, 1, 1]
        reps = (n_symbols if modulation == "BPSK" else 2 * n_symbols) // len(base)
        return np.tile(base, reps + 1)[: n_symbols if modulation == "BPSK" else 2 * n_symbols]
    if pattern == "period3":
        base = [0, 1, 1] if modulation == "BPSK" else [0, 0, 0, 1, 1, 1]
        need = n_symbols if modulation == "BPSK" else 2 * n_symbols
        reps = need // len(base) + 1
        return np.tile(base, reps)[:need]
    if pattern in ("runs4", "runs16"):
        run_length = 4 if pattern == "runs4" else 16
        n_base = max(n_symbols // run_length, 4)
        rng = np.random.default_rng(seed)
        if modulation == "BPSK":
            base = rng.integers(0, 2, n_base)
            return np.repeat(base, run_length)[:n_symbols]
        base = rng.integers(0, 2, 2 * n_base).reshape(-1, 2)
        return np.repeat(base, run_length, axis=0).reshape(-1)[: 2 * n_symbols]
    raise ValueError(f"unknown pattern {pattern!r}")


def _build_waveform(modulation: str, sps: int, bits: np.ndarray) -> np.ndarray:
    if modulation == "BPSK":
        return bpsk_waveform(bits, sps)
    return qpsk_waveform(bits, sps)


def _run_hm(samples: np.ndarray):
    max_sps = min(HM_DEFAULT_MAX_SPS, samples.size - 2)
    try:
        result = hm_estimate_symbol_rate(samples, FS, max_sps=max_sps)
    except ValueError:
        return "rejected", None, None, None
    outcome = "correct" if result.samples_per_symbol == _CURRENT_TRUE_SPS else "wrong"
    return outcome, result.samples_per_symbol, result.symbol_rate_hz, result.score


def _run_sayan(samples: np.ndarray):
    try:
        result = _sayan_estimate_symbol_rate(samples, FS)
    except ValueError:
        return "rejected", None, None, None, None
    outcome = "correct" if result.samples_per_symbol == _CURRENT_TRUE_SPS else "wrong"
    return (
        outcome,
        result.samples_per_symbol,
        result.symbol_rate_hz,
        result.quality,
        result.boundary_offset,
    )


_CURRENT_TRUE_SPS = 0
_SEED_COUNTER = count(1)


def _add_spec(
    specs: list[dict],
    case: str,
    detail: str,
    modulation: str,
    sps: int,
    n_symbols: int,
    pattern: str | None = None,
    phase_rad: float | None = None,
    amplitude: float | None = None,
    crop: int | None = None,
    snr_db: float | None = None,
    cfo_hz: float | None = None,
) -> None:
    specs.append(
        dict(
            case=case,
            detail=detail,
            modulation=modulation,
            sps=sps,
            n_symbols=n_symbols,
            pattern=pattern,
            phase_rad=phase_rad,
            amplitude=amplitude,
            crop=crop,
            snr_db=snr_db,
            cfo_hz=cfo_hz,
            seed=next(_SEED_COUNTER),
        )
    )


def build_specs() -> list[dict]:
    specs: list[dict] = []
    mods = ("BPSK", "QPSK")
    for modulation in mods:
        for sps in TRUE_SPS_VALUES:
            for _ in range(2):
                _add_spec(specs, "CLEAN", "clean", modulation, sps, DEFAULT_SYMBOL_COUNT)
            for phase in PHASES_RAD:
                _add_spec(
                    specs, "PHASE", f"phase={phase}", modulation, sps,
                    DEFAULT_SYMBOL_COUNT, phase_rad=phase,
                )
            for amplitude in AMPLITUDES:
                _add_spec(
                    specs, "AMPLITUDE", f"amp={amplitude}", modulation, sps,
                    DEFAULT_SYMBOL_COUNT, amplitude=amplitude,
                )
            for crop in range(1, sps):
                _add_spec(
                    specs, "CROP", f"crop={crop}", modulation, sps,
                    DEFAULT_SYMBOL_COUNT, crop=crop,
                )
            for snr_db in SNR_DB_VALUES:
                _add_spec(
                    specs, "AWGN", f"snr={snr_db}", modulation, sps,
                    DEFAULT_SYMBOL_COUNT, snr_db=snr_db,
                )
            for magnitude in CFO_FRACTION_MAGNITUDES:
                signed = [0.0] if magnitude == 0.0 else [magnitude, -magnitude]
                for fraction in signed:
                    cfo_hz = fraction * (FS / sps)
                    _add_spec(
                        specs, "CFO", f"cfo_frac={fraction:+.2f}", modulation, sps,
                        DEFAULT_SYMBOL_COUNT, cfo_hz=cfo_hz,
                    )
            for n_symbols in SHORT_SYMBOL_COUNTS:
                _add_spec(
                    specs, "SHORT", f"symbols={n_symbols}", modulation, sps, n_symbols
                )
        for sps in PATHOLOGICAL_SPS_VALUES:
            for pattern in ("all_identical", "alternating", "period3", "runs4", "runs16"):
                _add_spec(
                    specs, "PATHOLOGICAL", f"pattern={pattern}", modulation, sps,
                    DEFAULT_SYMBOL_COUNT, pattern=pattern,
                )
    return specs


def run_trial(spec: dict) -> TrialRecord:
    global _CURRENT_TRUE_SPS
    _CURRENT_TRUE_SPS = spec["sps"]
    seed = spec["seed"]
    if spec["pattern"] is not None:
        bits = _pattern_bits(spec["modulation"], spec["n_symbols"], spec["pattern"], seed)
    else:
        bits = _random_bits(spec["modulation"], spec["n_symbols"], seed)
    waveform = _build_waveform(spec["modulation"], spec["sps"], bits)
    if spec["phase_rad"] is not None:
        waveform = apply_phase_offset(waveform, spec["phase_rad"])
    if spec["amplitude"] is not None:
        waveform = waveform * spec["amplitude"]
    if spec["snr_db"] is not None:
        waveform = add_awgn(
            waveform, spec["snr_db"], rng=np.random.default_rng(seed + 500_000)
        )
    if spec["cfo_hz"] is not None and spec["cfo_hz"] != 0.0:
        waveform = apply_frequency_offset(waveform, FS, spec["cfo_hz"])
    if spec["crop"] is not None:
        waveform = waveform[spec["crop"] :]

    hm_outcome, hm_sps, hm_rate, hm_score = _run_hm(waveform)
    sayan_outcome, sayan_sps, sayan_rate, sayan_quality, sayan_offset = _run_sayan(waveform)
    expected_offset = (
        (-spec["crop"]) % spec["sps"] if spec["crop"] is not None else None
    )
    return TrialRecord(
        modulation=spec["modulation"],
        true_sps=spec["sps"],
        true_rate_hz=FS / spec["sps"],
        case=spec["case"],
        detail=spec["detail"],
        seed=seed,
        hm_outcome=hm_outcome,
        hm_sps=hm_sps,
        hm_rate_hz=hm_rate,
        hm_score=hm_score,
        sayan_outcome=sayan_outcome,
        sayan_sps=sayan_sps,
        sayan_rate_hz=sayan_rate,
        sayan_quality=sayan_quality,
        sayan_boundary_offset=sayan_offset,
        expected_boundary_offset=expected_offset,
    )


def _stats(records: list[TrialRecord]) -> dict[str, float]:
    n = len(records)
    if n == 0:
        return {"n": 0}
    hm = {"correct": 0, "rejected": 0, "wrong": 0}
    sy = {"correct": 0, "rejected": 0, "wrong": 0}
    for record in records:
        hm[record.hm_outcome] += 1
        sy[record.sayan_outcome] += 1
    return {
        "hm_correct": 100.0 * hm["correct"] / n,
        "hm_rejected": 100.0 * hm["rejected"] / n,
        "hm_wrong": 100.0 * hm["wrong"] / n,
        "sy_correct": 100.0 * sy["correct"] / n,
        "sy_rejected": 100.0 * sy["rejected"] / n,
        "sy_wrong": 100.0 * sy["wrong"] / n,
        "n": n,
    }


def _print_category_table(title: str, records: list[TrialRecord], key) -> None:
    print(f"\n{title}")
    header = (
        f"{'group':<22} {'HM ok%':>7} {'HM rej%':>8} {'HM wrong%':>9}"
        f" {'SY ok%':>7} {'SY rej%':>8} {'SY wrong%':>9} {'n':>6}"
    )
    print(header)
    print("-" * len(header))
    groups: dict[str, list[TrialRecord]] = {}
    for record in records:
        groups.setdefault(key(record), []).append(record)
    for group_name in sorted(groups, key=str):
        stats = _stats(groups[group_name])
        print(
            f"{str(group_name):<22} {stats['hm_correct']:>7.1f}"
            f" {stats['hm_rejected']:>8.1f} {stats['hm_wrong']:>9.1f}"
            f" {stats['sy_correct']:>7.1f} {stats['sy_rejected']:>8.1f}"
            f" {stats['sy_wrong']:>9.1f} {stats['n']:>6}"
        )


def _print_case_table(records: list[TrialRecord]) -> None:
    print("\nSUMMARY BY CASE CATEGORY")
    header = (
        f"{'case':<14} {'HM ok%':>7} {'HM rej%':>8} {'HM wrong%':>9}"
        f" {'SY ok%':>7} {'SY rej%':>8} {'SY wrong%':>9} {'n':>6}"
    )
    print(header)
    print("-" * len(header))
    case_order = ["CLEAN", "PHASE", "AMPLITUDE", "CROP", "AWGN", "CFO", "SHORT", "PATHOLOGICAL"]
    for case_name in case_order:
        subset = [r for r in records if r.case == case_name]
        stats = _stats(subset)
        print(
            f"{case_name:<14} {stats['hm_correct']:>7.1f}"
            f" {stats['hm_rejected']:>8.1f} {stats['hm_wrong']:>9.1f}"
            f" {stats['sy_correct']:>7.1f} {stats['sy_rejected']:>8.1f}"
            f" {stats['sy_wrong']:>9.1f} {stats['n']:>6}"
        )
    stats = _stats(records)
    print(
        f"{'OVERALL':<14} {stats['hm_correct']:>7.1f}"
        f" {stats['hm_rejected']:>8.1f} {stats['hm_wrong']:>9.1f}"
        f" {stats['sy_correct']:>7.1f} {stats['sy_rejected']:>8.1f}"
        f" {stats['sy_wrong']:>9.1f} {stats['n']:>6}"
    )


def boundary_offset_report(records: list[TrialRecord]) -> dict[str, float]:
    crop_records = [r for r in records if r.case == "CROP"]
    usable = [
        r
        for r in crop_records
        if r.sayan_outcome == "correct" and r.sayan_boundary_offset is not None
    ]
    exact = 0
    off_by_one = 0
    gross = 0
    for record in usable:
        expected = record.expected_boundary_offset
        got = record.sayan_boundary_offset
        distance = min((got - expected) % record.true_sps, (expected - got) % record.true_sps)
        if distance == 0:
            exact += 1
        elif distance == 1:
            off_by_one += 1
        else:
            gross += 1
    rate = 100.0 * exact / len(usable) if usable else float("nan")
    print("\nBOUNDARY OFFSET (crop trials, Sayan SPS correct)")
    print(f"  usable crop trials with correct Sayan SPS: {len(usable)} / {len(crop_records)}")
    print(f"  exact-match rate: {rate:.1f}%  (exact {exact}, off-by-one {off_by_one}, gross {gross})")
    return {
        "usable": float(len(usable)),
        "total_crop": float(len(crop_records)),
        "exact_rate": rate,
        "off_by_one": float(off_by_one),
        "gross": float(gross),
    }


def main() -> None:
    specs = build_specs()
    records = [run_trial(spec) for spec in specs]
    print(f"BENCHMARK COMPLETE: {len(records)} trials")
    print(f"fs = {FS:.0f} Hz, true SPS = {TRUE_SPS_VALUES}, default thresholds,")
    print("HM max_sps capped to len-2 (its own validation), Sayan internal cap unchanged.")

    _print_case_table(records)
    _print_category_table("SPLIT BY MODULATION", records, lambda r: r.modulation)
    _print_category_table("SPLIT BY TRUE SPS", records, lambda r: r.true_sps)
    _print_category_table(
        "SPLIT BY SNR (AWGN trials)", [r for r in records if r.case == "AWGN"], lambda r: r.detail
    )
    _print_category_table(
        "SPLIT BY CFO FRACTION (CFO trials)", [r for r in records if r.case == "CFO"], lambda r: r.detail
    )
    offset_stats = boundary_offset_report(records)

    overall = _stats(records)
    total = overall["n"]
    hm_wins = []
    sy_wins = []
    for case_name in ["CLEAN", "PHASE", "AMPLITUDE", "CROP", "AWGN", "CFO", "SHORT", "PATHOLOGICAL"]:
        stats = _stats([r for r in records if r.case == case_name])
        if stats["hm_correct"] - stats["sy_correct"] >= CLEAR_WIN_MARGIN_PCT:
            hm_wins.append(case_name)
        elif stats["sy_correct"] - stats["hm_correct"] >= CLEAR_WIN_MARGIN_PCT:
            sy_wins.append(case_name)
    both_fail = sum(
        1
        for r in records
        if r.hm_outcome != "correct" and r.sayan_outcome != "correct"
    )

    print("\nVERDICT DATA")
    print(f"TRIAL_COUNT: {total}")
    print(f"HM_OVERALL_ACCURACY: {overall['hm_correct'] / 100.0:.4f}")
    print(f"SAYAN_OVERALL_ACCURACY: {overall['sy_correct'] / 100.0:.4f}")
    print(f"HM_FALSE_ESTIMATE_RATE: {overall['hm_wrong'] / 100.0:.4f}")
    print(f"SAYAN_FALSE_ESTIMATE_RATE: {overall['sy_wrong'] / 100.0:.4f}")
    print(f"HM_REJECTION_RATE: {overall['hm_rejected'] / 100.0:.4f}")
    print(f"SAYAN_REJECTION_RATE: {overall['sy_rejected'] / 100.0:.4f}")
    print(
        "BOUNDARY_OFFSET_EXACT_MATCH_RATE: "
        f"{offset_stats['exact_rate'] / 100.0:.4f}"
    )
    print(f"HM_CLEAR_WINS: {hm_wins}")
    print(f"SAYAN_CLEAR_WINS: {sy_wins}")
    print(f"BOTH_FAIL_TRIALS: {both_fail}")


if __name__ == "__main__":
    main()
