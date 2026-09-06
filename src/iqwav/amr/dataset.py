"""Leakage-safe synthetic AMC dataset generation.

Tier A IQWAV synthetic records only. This module does not classify,
load public datasets, or apply Module 11 synchronization.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from iqwav.dsp import add_awgn, apply_frequency_offset, apply_phase_offset
from iqwav.modulation import (
    am_modulate,
    bpsk_waveform,
    fm_modulate,
    pm_modulate,
    qpsk_waveform,
)

from .messages import MESSAGE_FAMILIES, generate_analog_message

__all__ = [
    "LABELS",
    "DatasetSeeds",
    "NuisanceMetadata",
    "SyntheticDataset",
    "SyntheticDatasetConfig",
    "SyntheticRecord",
    "classifier_labels",
    "classifier_samples",
    "generate_synthetic_dataset",
]

LABELS = ("am", "fm", "pm", "bpsk", "qpsk")
_ANALOG_LABELS = frozenset({"am", "fm", "pm"})
_DIGITAL_LABELS = frozenset({"bpsk", "qpsk"})


def _validate_int(value: object, name: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer, got {value!r}.")
    value = int(value)
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value!r}.")
    return value


def _validate_real(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise ValueError(f"{name} must be a real scalar number, got {value!r}.")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value!r}.")
    return value


def _validate_range(
    pair: object, name: str, *, low_min: float, high_max: float
) -> tuple[float, float]:
    if not isinstance(pair, tuple) or len(pair) != 2:
        raise ValueError(f"{name} must be a length-2 tuple, got {pair!r}.")
    low = _validate_real(pair[0], f"{name}[0]")
    high = _validate_real(pair[1], f"{name}[1]")
    if not low_min <= low <= high <= high_max:
        raise ValueError(
            f"{name} must satisfy {low_min} <= low <= high <= {high_max}, "
            f"got {pair!r}."
        )
    return (low, high)


def _uniform(rng: np.random.Generator, bounds: tuple[float, float]) -> float:
    return float(rng.uniform(bounds[0], bounds[1]))


def _freeze_iq(samples: np.ndarray) -> npt.NDArray[np.complex128]:
    frozen = np.array(samples, dtype=np.complex128, copy=True)
    frozen.setflags(write=False)
    return frozen


def _spawn_rngs(
    seed: int, labels: tuple[str, ...]
) -> dict[str, np.random.Generator]:
    children = np.random.SeedSequence(seed).spawn(len(labels))
    return {
        label: np.random.default_rng(child)
        for label, child in zip(labels, children)
    }


@dataclass(frozen=True)
class DatasetSeeds:
    """Independent deterministic seeds for the four harness streams.

    ``payload`` draws analog messages and digital bit payloads.
    ``nuisance`` draws SNR, amplitude, phase, CFO, length, SPS, crop,
    and analog modulator parameters.
    ``awgn`` is used only by :func:`iqwav.dsp.add_awgn`.
    ``split`` is consumed by the group-aware splitter, not generation.
    """

    payload: int = 1
    nuisance: int = 2
    awgn: int = 3
    split: int = 4

    def __post_init__(self) -> None:
        for name in ("payload", "nuisance", "awgn", "split"):
            _validate_int(getattr(self, name), name, minimum=0)


@dataclass(frozen=True)
class NuisanceMetadata:
    """Ground-truth nuisances for audit and stratified reporting.

    These fields are not classifier inputs. Use
    :func:`classifier_samples` to obtain IQ only.
    """

    snr_db: float
    amplitude: float
    phase_rad: float
    cfo_norm: float
    cfo_hz: float
    n_samples: int
    samples_per_symbol: int | None = None
    crop: int | None = None
    message_family: str | None = None
    message_freq_norm: tuple[float, ...] | None = None
    modulation_index: float | None = None
    frequency_deviation_hz: float | None = None
    phase_deviation_rad: float | None = None


@dataclass(frozen=True, eq=False)
class SyntheticRecord:
    """One labelled synthetic IQ record.

    ``group_id`` identifies the parent information realization within
    one label. Current IDs assume information realizations are not
    shared across labels: a BPSK payload parent is not the same group
    as a QPSK payload with the same parent index, and analog messages
    are generated per label.

    ``record_id`` identifies this variant. ``nuisances`` is audit
    metadata, not a feature vector. Combination holdouts must keep all
    variants of one ``group_id`` on the same side of
    development/holdout; split groups first, then filter nuisances.
    """

    samples: npt.NDArray[np.complex128]
    label: str
    fs: float
    group_id: str
    record_id: str
    nuisances: NuisanceMetadata

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SyntheticRecord):
            return NotImplemented
        return (
            self.label == other.label
            and self.fs == other.fs
            and self.group_id == other.group_id
            and self.record_id == other.record_id
            and self.nuisances == other.nuisances
            and np.array_equal(self.samples, other.samples)
        )


@dataclass(frozen=True)
class SyntheticDatasetConfig:
    """Shared, class-independent synthetic dataset configuration.

    Discrete catalogs (SNR, SPS, record length, analog message family)
    are single tuples used by every applicable class. CFO is drawn as a
    normalized cycles-per-sample value ``cfo_norm`` with
    ``cfo_hz = cfo_norm * fs``. This is not the Module 11A PSK
    estimator ambiguity range.

    Split fractions later apply to parent groups, not records.
    Combination holdouts must preserve parent grouping.

    FM and PM share message families here, but their modulator
    parameters are not the same physical excursion. Full-domain FM/PM
    scores must later be accompanied by evaluation in an overlapping
    effective-excursion region; this config does not define that region.
    """

    fs: float = 48_000.0
    n_samples_values: tuple[int, ...] = (256,)
    labels: tuple[str, ...] = LABELS
    n_parents_per_class: int = 8
    n_variants_per_parent: int = 3
    snr_db_values: tuple[float, ...] = (0.0, 5.0, 10.0, 15.0, 20.0)
    amplitude_range: tuple[float, float] = (0.5, 2.0)
    phase_range: tuple[float, float] = (-math.pi, math.pi)
    cfo_norm_range: tuple[float, float] = (-0.02, 0.02)
    sps_values: tuple[int, ...] = (4, 8, 16)
    analog_message_families: tuple[str, ...] = MESSAGE_FAMILIES
    analog_freq_norm_range: tuple[float, float] = (0.01, 0.08)
    modulation_index_range: tuple[float, float] = (0.3, 0.9)
    fm_deviation_norm_range: tuple[float, float] = (0.02, 0.08)
    pm_phase_deviation_range: tuple[float, float] = (0.4, 1.2)
    seeds: DatasetSeeds = DatasetSeeds()


@dataclass(frozen=True)
class SyntheticDataset:
    """Generated records plus the config that produced them."""

    records: tuple[SyntheticRecord, ...]
    config: SyntheticDatasetConfig


@dataclass
class _Parent:
    label: str
    parent_index: int
    group_id: str
    bits: np.ndarray | None
    message: np.ndarray | None
    message_family: str | None
    message_freq_norm: tuple[float, ...] | None


def _validate_config(config: SyntheticDatasetConfig) -> None:
    _validate_real(config.fs, "fs")
    if config.fs <= 0.0:
        raise ValueError(f"fs must be positive, got {config.fs!r}.")
    _validate_int(config.n_parents_per_class, "n_parents_per_class", 1)
    _validate_int(config.n_variants_per_parent, "n_variants_per_parent", 1)
    if not config.labels:
        raise ValueError("labels must be a non-empty tuple.")
    unknown = [label for label in config.labels if label not in LABELS]
    if unknown:
        raise ValueError(f"unsupported labels: {unknown!r}.")
    if len(set(config.labels)) != len(config.labels):
        raise ValueError("labels must be unique.")
    if not config.n_samples_values:
        raise ValueError("n_samples_values must be non-empty.")
    for n in config.n_samples_values:
        _validate_int(n, "n_samples_values item", 1)
    if not config.snr_db_values:
        raise ValueError("snr_db_values must be non-empty.")
    for snr in config.snr_db_values:
        _validate_real(snr, "snr_db_values item")
    _validate_range(config.amplitude_range, "amplitude_range", low_min=0.0, high_max=math.inf)
    if config.amplitude_range[0] <= 0.0:
        raise ValueError("amplitude_range low must be > 0.")
    _validate_range(
        config.phase_range, "phase_range", low_min=-2.0 * math.pi, high_max=2.0 * math.pi
    )
    _validate_range(
        config.cfo_norm_range, "cfo_norm_range", low_min=-0.5, high_max=0.5
    )
    if not config.sps_values:
        raise ValueError("sps_values must be non-empty.")
    for sps in config.sps_values:
        _validate_int(sps, "sps_values item", 1)
    if not config.analog_message_families:
        raise ValueError("analog_message_families must be non-empty.")
    bad_family = [
        family
        for family in config.analog_message_families
        if family not in MESSAGE_FAMILIES
    ]
    if bad_family:
        raise ValueError(f"unsupported analog message families: {bad_family!r}.")
    low_f, high_f = config.analog_freq_norm_range
    if not 0.0 < low_f <= high_f < 0.5:
        raise ValueError(
            "analog_freq_norm_range must satisfy 0 < low <= high < 0.5, "
            f"got {config.analog_freq_norm_range!r}."
        )
    _validate_range(
        config.modulation_index_range,
        "modulation_index_range",
        low_min=0.0,
        high_max=1.0,
    )
    _validate_range(
        config.fm_deviation_norm_range,
        "fm_deviation_norm_range",
        low_min=0.0,
        high_max=0.499999,
    )
    if config.fm_deviation_norm_range[1] >= 0.5:
        raise ValueError("fm_deviation_norm_range high must be < 0.5.")
    _validate_range(
        config.pm_phase_deviation_range,
        "pm_phase_deviation_range",
        low_min=0.0,
        high_max=math.inf,
    )


def _digital_symbol_count(config: SyntheticDatasetConfig) -> int:
    max_n = max(config.n_samples_values)
    max_sps = max(config.sps_values)
    min_sps = min(config.sps_values)
    return (max_n + max_sps + min_sps - 1) // min_sps


def _make_parent(
    label: str,
    parent_index: int,
    payload_rng: np.random.Generator,
    config: SyntheticDatasetConfig,
    n_symbols: int,
    analog_parent_len: int,
) -> _Parent:
    group_id = f"{label}:p{parent_index:06d}"
    if label in _DIGITAL_LABELS:
        n_bits = n_symbols if label == "bpsk" else 2 * n_symbols
        bits = payload_rng.integers(0, 2, n_bits, dtype=np.int64)
        return _Parent(
            label=label,
            parent_index=parent_index,
            group_id=group_id,
            bits=bits,
            message=None,
            message_family=None,
            message_freq_norm=None,
        )
    family = config.analog_message_families[
        parent_index % len(config.analog_message_families)
    ]
    message, freqs = generate_analog_message(
        analog_parent_len,
        config.fs,
        family,
        payload_rng,
        config.analog_freq_norm_range,
    )
    return _Parent(
        label=label,
        parent_index=parent_index,
        group_id=group_id,
        bits=None,
        message=message,
        message_family=family,
        message_freq_norm=freqs,
    )


def _apply_nuisances(
    iq: np.ndarray,
    fs: float,
    amplitude: float,
    phase_rad: float,
    cfo_hz: float,
    snr_db: float,
    awgn_rng: np.random.Generator,
) -> npt.NDArray[np.complex128]:
    scaled = np.asarray(iq, dtype=np.complex128) * amplitude
    rotated = apply_phase_offset(scaled, phase_rad)
    shifted = apply_frequency_offset(rotated, fs, cfo_hz)
    return add_awgn(shifted, snr_db, rng=awgn_rng)


def _make_variant(
    parent: _Parent,
    variant_index: int,
    nuisance_rng: np.random.Generator,
    awgn_rng: np.random.Generator,
    config: SyntheticDatasetConfig,
) -> SyntheticRecord:
    slot = parent.parent_index * config.n_variants_per_parent + variant_index
    n_samples = int(
        config.n_samples_values[slot % len(config.n_samples_values)]
    )
    snr_db = float(config.snr_db_values[slot % len(config.snr_db_values)])
    amplitude = _uniform(nuisance_rng, config.amplitude_range)
    phase_rad = _uniform(nuisance_rng, config.phase_range)
    cfo_norm = _uniform(nuisance_rng, config.cfo_norm_range)
    cfo_hz = cfo_norm * config.fs
    record_id = f"{parent.group_id}:v{variant_index:04d}"

    samples_per_symbol = None
    crop = None
    modulation_index = None
    frequency_deviation_hz = None
    phase_deviation_rad = None

    if parent.label in _DIGITAL_LABELS:
        samples_per_symbol = int(config.sps_values[slot % len(config.sps_values)])
        if parent.label == "bpsk":
            waveform = bpsk_waveform(parent.bits, samples_per_symbol)
        else:
            waveform = qpsk_waveform(parent.bits, samples_per_symbol)
        crop = int(nuisance_rng.integers(0, samples_per_symbol))
        stop = crop + n_samples
        if stop > waveform.shape[0]:
            raise RuntimeError(
                "internal error: digital parent is shorter than the "
                f"requested crop window ({stop} > {waveform.shape[0]})."
            )
        clean = waveform[crop:stop]
    else:
        message = parent.message[:n_samples]
        if parent.label == "am":
            modulation_index = _uniform(
                nuisance_rng, config.modulation_index_range
            )
            clean = am_modulate(message, modulation_index)
        elif parent.label == "fm":
            deviation_norm = _uniform(
                nuisance_rng, config.fm_deviation_norm_range
            )
            frequency_deviation_hz = deviation_norm * config.fs
            clean = fm_modulate(message, config.fs, frequency_deviation_hz)
        else:
            phase_deviation_rad = _uniform(
                nuisance_rng, config.pm_phase_deviation_range
            )
            clean = pm_modulate(message, phase_deviation_rad)

    iq = _apply_nuisances(
        clean,
        config.fs,
        amplitude,
        phase_rad,
        cfo_hz,
        snr_db,
        awgn_rng,
    )
    nuisances = NuisanceMetadata(
        snr_db=snr_db,
        amplitude=amplitude,
        phase_rad=phase_rad,
        cfo_norm=cfo_norm,
        cfo_hz=cfo_hz,
        n_samples=n_samples,
        samples_per_symbol=samples_per_symbol,
        crop=crop,
        message_family=parent.message_family,
        message_freq_norm=parent.message_freq_norm,
        modulation_index=modulation_index,
        frequency_deviation_hz=frequency_deviation_hz,
        phase_deviation_rad=phase_deviation_rad,
    )
    return SyntheticRecord(
        samples=_freeze_iq(iq),
        label=parent.label,
        fs=float(config.fs),
        group_id=parent.group_id,
        record_id=record_id,
        nuisances=nuisances,
    )


def generate_synthetic_dataset(
    config: SyntheticDatasetConfig | None = None,
) -> SyntheticDataset:
    """Generate labelled synthetic IQ records from a frozen config.

    Parent information (bits or analog message) is drawn from the
    payload stream. Impairments are drawn from the nuisance stream and
    applied with existing DSP primitives. AWGN uses a third stream.
    """
    if config is None:
        config = SyntheticDatasetConfig()
    _validate_config(config)
    payload_rngs = _spawn_rngs(config.seeds.payload, config.labels)
    nuisance_rngs = _spawn_rngs(config.seeds.nuisance, config.labels)
    awgn_rngs = _spawn_rngs(config.seeds.awgn, config.labels)
    n_symbols = _digital_symbol_count(config)
    analog_parent_len = max(config.n_samples_values)
    records: list[SyntheticRecord] = []
    for label in config.labels:
        payload_rng = payload_rngs[label]
        nuisance_rng = nuisance_rngs[label]
        awgn_rng = awgn_rngs[label]
        for parent_index in range(config.n_parents_per_class):
            parent = _make_parent(
                label,
                parent_index,
                payload_rng,
                config,
                n_symbols,
                analog_parent_len,
            )
            for variant_index in range(config.n_variants_per_parent):
                records.append(
                    _make_variant(
                        parent,
                        variant_index,
                        nuisance_rng,
                        awgn_rng,
                        config,
                    )
                )
    return SyntheticDataset(records=tuple(records), config=config)


def classifier_samples(
    records: tuple[SyntheticRecord, ...] | list[SyntheticRecord],
) -> tuple[npt.NDArray[np.complex128], ...]:
    """Return IQ arrays only. Ground-truth nuisances are excluded."""
    return tuple(record.samples for record in records)


def classifier_labels(
    records: tuple[SyntheticRecord, ...] | list[SyntheticRecord],
) -> tuple[str, ...]:
    """Return truth labels aligned with :func:`classifier_samples`."""
    return tuple(record.label for record in records)
