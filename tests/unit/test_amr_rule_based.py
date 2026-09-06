"""Regression tests for the frozen Module 12D rule-based AMC baseline."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from iqwav.amr import (
    AMCFeatures,
    DatasetSeeds,
    PRIMARY_AMC_LABELS,
    SyntheticDatasetConfig,
    classify_amc_features,
    classify_modulation,
    extract_amc_features,
    generate_synthetic_dataset,
)


def _features(**overrides: float) -> AMCFeatures:
    values = {
        "envelope_dispersion": 0.0,
        "envelope_coherence": 0.0,
        "phase_step_sparsity": 0.0,
        "transition_periodicity": 0.0,
        "c2": 0.0,
        "c4": 1.0,
        "c2_c4_ratio": 0.0,
    }
    values.update(overrides)
    return AMCFeatures(**values)


def _reference_dataset():
    return generate_synthetic_dataset(
        SyntheticDatasetConfig(
            n_parents_per_class=2,
            n_variants_per_parent=2,
            seeds=DatasetSeeds(31, 32, 33, 34),
        )
    )


def _experiment_features(samples: np.ndarray) -> dict[str, float]:
    scripts = str(Path(__file__).resolve().parents[2] / "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    from _exp_12d1 import extract_features as frozen_extract_features

    return frozen_extract_features(samples)


def _experimental_label(features: dict[str, float]) -> str:
    if (
        features["envelope_coherence"] > 0.1488818140943029
        or features["envelope_dispersion"] > 0.24798287316712025
    ):
        return "am"
    if not (
        features["phase_sparsity"] > 0.4361237911585857
        or features["transition_periodicity"] > 0.599624718621257
    ):
        return "angle"
    if features["c2"] > 0.4141447011963815 or features["c2_over_c4"] > 2.847400439321907:
        return "bpsk"
    return "qpsk"


def test_strict_frozen_rule_boundaries_and_routing() -> None:
    assert classify_amc_features(_features(envelope_coherence=0.1488818140943029)).label == "angle"
    assert classify_amc_features(_features(envelope_coherence=np.nextafter(0.1488818140943029, np.inf))).label == "am"
    assert classify_amc_features(_features(envelope_dispersion=0.24798287316712025)).label == "angle"
    assert classify_amc_features(_features(envelope_dispersion=np.nextafter(0.24798287316712025, np.inf))).label == "am"

    assert classify_amc_features(_features(phase_step_sparsity=0.4361237911585857)).label == "angle"
    assert classify_amc_features(_features(phase_step_sparsity=np.nextafter(0.4361237911585857, np.inf))).label == "qpsk"
    assert classify_amc_features(_features(transition_periodicity=0.599624718621257)).label == "angle"
    assert classify_amc_features(_features(transition_periodicity=np.nextafter(0.599624718621257, np.inf))).label == "qpsk"

    assert classify_amc_features(_features(phase_step_sparsity=1.0, c2=0.4141447011963815)).label == "qpsk"
    assert classify_amc_features(_features(phase_step_sparsity=1.0, c2=np.nextafter(0.4141447011963815, np.inf))).label == "bpsk"
    assert classify_amc_features(_features(phase_step_sparsity=1.0, c2_c4_ratio=2.847400439321907)).label == "qpsk"
    assert classify_amc_features(_features(phase_step_sparsity=1.0, c2_c4_ratio=np.nextafter(2.847400439321907, np.inf))).label == "bpsk"


def test_fixed_frozen_feature_references() -> None:
    expected = {
        "am": (0.22086791562142177, 0.14768052209182611, 0.33661665206701974, 0.21409332613015644, 0.1120027539788578, 0.021153264732344974, 5.2948211725588692),
        "fm": (0.209709488830448, -0.048674612336649874, 0.34545325940382043, 0.19210237277594705, 0.12301431108269692, 0.023243539087293624, 5.292408811558815),
        "pm": (0.20775118994175368, -0.018167255945053973, 0.32382330096569012, 0.22305301105347008, 0.08892158227298638, 0.02801003614363594, 3.1746328999298816),
        "bpsk": (0.19178900460533027, -0.05820175487670693, 0.4173576311741193, 0.2533985338466231, 0.1180862322704966, 0.024874436105641873, 4.747292833664104),
        "qpsk": (0.24293623627489744, 0.008788960342005355, 0.38756080637592416, 0.16095040640580643, 0.02794994728364865, 0.021331967555990178, 1.3102376613398634),
    }
    for record in _reference_dataset().records:
        if not (record.group_id.endswith("000000") and record.record_id.endswith("0000")):
            continue
        actual = extract_amc_features(record.samples)
        assert tuple(actual.__dict__.values()) == pytest.approx(expected[record.label], rel=1e-12, abs=1e-12)


def test_production_matches_frozen_experiment_across_representative_nuisances() -> None:
    mapping = {
        "envelope_dispersion": "envelope_dispersion",
        "envelope_coherence": "envelope_coherence",
        "phase_step_sparsity": "phase_sparsity",
        "transition_periodicity": "transition_periodicity",
        "c2": "c2",
        "c4": "c4",
        "c2_c4_ratio": "c2_over_c4",
    }
    for record in _reference_dataset().records:
        expected = _experiment_features(record.samples)
        actual = extract_amc_features(record.samples)
        for production_name, experiment_name in mapping.items():
            assert getattr(actual, production_name) == pytest.approx(expected[experiment_name], rel=1e-12, abs=1e-12)
        assert classify_modulation(record.samples).label == _experimental_label(expected)


def test_fm_pm_are_angle_semantics_not_production_labels() -> None:
    labels = {
        record.label: classify_modulation(record.samples).label
        for record in _reference_dataset().records
        if record.label in ("fm", "pm")
    }
    assert set(labels.values()) <= set(PRIMARY_AMC_LABELS)
    assert "fm" not in labels.values()
    assert "pm" not in labels.values()


def test_accepted_amplitude_and_static_phase_invariances() -> None:
    rng = np.random.default_rng(77)
    samples = rng.normal(size=256) + 1j * rng.normal(size=256)
    base = extract_amc_features(samples)
    scaled = extract_amc_features(2.5 * samples)
    rotated = extract_amc_features(samples * np.exp(0.8j))
    assert tuple(base.__dict__.values()) == pytest.approx(tuple(scaled.__dict__.values()), rel=1e-12, abs=1e-12)
    assert tuple(base.__dict__.values()) == pytest.approx(tuple(rotated.__dict__.values()), rel=1e-12, abs=1e-12)


@pytest.mark.parametrize(
    "samples",
    [
        np.array([], dtype=np.complex128),
        np.ones((2, 2), dtype=np.complex128),
        np.ones(8, dtype=np.float64),
        np.array([1.0 + np.nan * 1j]),
        np.array([1.0 + np.inf * 1j]),
        np.zeros(8, dtype=np.complex128),
    ],
)
def test_invalid_or_degenerate_samples_are_rejected(samples: np.ndarray) -> None:
    with pytest.raises(ValueError):
        classify_modulation(samples)


def test_short_records_are_guarded_deterministic_and_input_is_not_mutated() -> None:
    samples = np.array([1.0 + 0.0j], dtype=np.complex128)
    before = samples.copy()
    first = classify_modulation(samples)
    second = classify_modulation(samples)
    assert first == second
    assert first.features.transition_periodicity == 0.0
    assert np.array_equal(samples, before)
