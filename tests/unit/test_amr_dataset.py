"""Tests for leakage-safe synthetic AMC dataset generation."""

import numpy as np
import pytest

from iqwav.amr import (
    LABELS,
    DatasetSeeds,
    SyntheticDatasetConfig,
    classifier_labels,
    classifier_samples,
    generate_synthetic_dataset,
    shortcut_audit,
)


def _config(**overrides):
    values = dict(
        fs=8_000.0,
        n_samples_values=(64,),
        n_parents_per_class=6,
        n_variants_per_parent=2,
        snr_db_values=(5.0, 10.0, 15.0),
        sps_values=(4, 8),
        analog_message_families=("tone", "multi_tone", "bandlimited"),
        seeds=DatasetSeeds(11, 22, 33, 44),
    )
    values.update(overrides)
    return SyntheticDatasetConfig(**values)


def test_generated_labels_are_the_accepted_set():
    dataset = generate_synthetic_dataset(_config())
    assert set(classifier_labels(dataset.records)) == set(LABELS)
    counts = {
        label: sum(record.label == label for record in dataset.records)
        for label in LABELS
    }
    assert counts == {label: 12 for label in LABELS}


def test_records_are_one_d_complex_iq_of_configured_length():
    dataset = generate_synthetic_dataset(_config())
    for record in dataset.records:
        assert record.samples.ndim == 1
        assert record.samples.dtype == np.complex128
        assert record.samples.shape == (64,)
        assert record.fs == 8_000.0
        assert record.nuisances.n_samples == 64


def test_group_id_identifies_parent_not_variant():
    dataset = generate_synthetic_dataset(_config())
    by_group = {}
    for record in dataset.records:
        by_group.setdefault(record.group_id, []).append(record)
        assert record.group_id in record.record_id
        assert record.group_id.startswith(record.label + ":p")
    assert len(by_group) == 5 * 6
    for members in by_group.values():
        assert len(members) == 2
        assert len({member.label for member in members}) == 1


def test_identical_config_and_seeds_are_bit_identical():
    first = generate_synthetic_dataset(_config())
    second = generate_synthetic_dataset(_config())
    assert [record.record_id for record in first.records] == [
        record.record_id for record in second.records
    ]
    for left, right in zip(first.records, second.records):
        assert left == right
        np.testing.assert_array_equal(left.samples, right.samples)


def test_awgn_seed_changes_samples_not_group_or_nuisance_catalog():
    quiet = generate_synthetic_dataset(_config(seeds=DatasetSeeds(11, 22, 33, 44)))
    noisy = generate_synthetic_dataset(_config(seeds=DatasetSeeds(11, 22, 99, 44)))
    assert [record.group_id for record in quiet.records] == [
        record.group_id for record in noisy.records
    ]
    assert [record.nuisances for record in quiet.records] == [
        record.nuisances for record in noisy.records
    ]
    assert any(
        not np.array_equal(left.samples, right.samples)
        for left, right in zip(quiet.records, noisy.records)
    )


def test_payload_seed_does_not_reuse_awgn_stream():
    first = generate_synthetic_dataset(_config(seeds=DatasetSeeds(1, 22, 33, 44)))
    second = generate_synthetic_dataset(_config(seeds=DatasetSeeds(2, 22, 33, 44)))
    assert [record.record_id for record in first.records] == [
        record.record_id for record in second.records
    ]
    assert any(
        not np.array_equal(left.samples, right.samples)
        for left, right in zip(first.records, second.records)
    )


def test_record_ids_are_unique():
    dataset = generate_synthetic_dataset(_config())
    ids = [record.record_id for record in dataset.records]
    assert len(ids) == len(set(ids))


def test_snr_catalog_is_shared_and_fully_used():
    dataset = generate_synthetic_dataset(_config())
    catalog = set(_config().snr_db_values)
    snr = {
        label: {
            record.nuisances.snr_db
            for record in dataset.records
            if record.label == label
        }
        for label in LABELS
    }
    assert all(value == catalog for value in snr.values())


def test_bpsk_and_qpsk_share_sps_support():
    dataset = generate_synthetic_dataset(_config())
    sps = {
        label: {
            record.nuisances.samples_per_symbol
            for record in dataset.records
            if record.label == label
        }
        for label in ("bpsk", "qpsk")
    }
    assert sps["bpsk"] == sps["qpsk"] == set(_config().sps_values)


def test_fm_and_pm_share_message_family_support():
    dataset = generate_synthetic_dataset(_config())
    families = {
        label: {
            record.nuisances.message_family
            for record in dataset.records
            if record.label == label
        }
        for label in ("am", "fm", "pm")
    }
    catalog = set(_config().analog_message_families)
    assert families["am"] == families["fm"] == families["pm"] == catalog


def test_record_lengths_do_not_identify_class():
    dataset = generate_synthetic_dataset(
        _config(n_samples_values=(48, 64), n_variants_per_parent=2)
    )
    lengths = {
        label: {
            record.nuisances.n_samples
            for record in dataset.records
            if record.label == label
        }
        for label in LABELS
    }
    shared = lengths["am"]
    assert all(value == shared == {48, 64} for value in lengths.values())


def test_cfo_is_normalized_times_fs():
    dataset = generate_synthetic_dataset(_config())
    for record in dataset.records:
        expected = record.nuisances.cfo_norm * record.fs
        assert record.nuisances.cfo_hz == pytest.approx(expected)
        assert -0.02 <= record.nuisances.cfo_norm <= 0.02


def test_classifier_helpers_exclude_nuisances():
    dataset = generate_synthetic_dataset(_config(n_parents_per_class=3))
    samples = classifier_samples(dataset.records)
    labels = classifier_labels(dataset.records)
    assert len(samples) == len(labels) == len(dataset.records)
    assert all(item.dtype == np.complex128 for item in samples)
    assert "samples_per_symbol" not in repr(samples)


def test_analog_messages_are_not_tone_only():
    dataset = generate_synthetic_dataset(_config())
    analog_families = {
        record.nuisances.message_family
        for record in dataset.records
        if record.label in {"am", "fm", "pm"}
    }
    assert analog_families == {"tone", "multi_tone", "bandlimited"}


def test_default_config_shortcut_audit_has_no_catalog_mismatch():
    dataset = generate_synthetic_dataset(_config())
    audit = shortcut_audit(dataset.records)
    assert audit.mismatches == ()


def test_invalid_label_rejected():
    with pytest.raises(ValueError):
        generate_synthetic_dataset(_config(labels=("am", "qam")))


def test_bool_seed_rejected():
    with pytest.raises(ValueError):
        DatasetSeeds(True, 2, 3, 4)
