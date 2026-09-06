"""Tests for group-aware splitting and combination holdout."""

import numpy as np
import pytest

from iqwav.amr import (
    DatasetSeeds,
    DatasetSplit,
    NuisanceMetadata,
    SyntheticDatasetConfig,
    SyntheticRecord,
    combination_holdout,
    filter_records,
    filter_split,
    generate_synthetic_dataset,
    split_dataset,
    split_records,
)


def _nuisances(**overrides):
    values = dict(
        snr_db=10.0,
        amplitude=1.0,
        phase_rad=0.0,
        cfo_norm=0.0,
        cfo_hz=0.0,
        n_samples=8,
    )
    values.update(overrides)
    return NuisanceMetadata(**values)


def _record(label, group_id, record_id, **nuisance_kwargs):
    n_samples = nuisance_kwargs.get("n_samples", 8)
    samples = np.zeros(n_samples, dtype=np.complex128)
    samples.setflags(write=False)
    return SyntheticRecord(
        samples=samples,
        label=label,
        fs=8_000.0,
        group_id=group_id,
        record_id=record_id,
        nuisances=_nuisances(**nuisance_kwargs),
    )


def _shared_parent_records():
    records = []
    for label in ("am", "fm", "pm", "bpsk", "qpsk"):
        for parent in range(6):
            group_id = f"{label}:p{parent:06d}"
            for variant in range(2):
                records.append(
                    _record(
                        label,
                        group_id,
                        f"{group_id}:v{variant:04d}",
                    )
                )
    return records


def test_shared_parent_never_crosses_splits():
    split = split_records(_shared_parent_records(), seed=7)
    membership = {}
    for name in ("train", "validation", "test"):
        for record in getattr(split, name):
            membership.setdefault(record.group_id, set()).add(name)
    assert membership
    assert all(len(splits) == 1 for splits in membership.values())


def test_every_record_appears_exactly_once():
    records = _shared_parent_records()
    split = split_records(records, seed=7)
    original = [record.record_id for record in records]
    combined = [record.record_id for record in split.all_records]
    assert sorted(combined) == sorted(original)
    assert len(combined) == len(set(combined)) == len(original)


def test_split_does_not_depend_on_record_order():
    records = _shared_parent_records()
    reversed_records = list(reversed(records))
    first = split_records(records, seed=3)
    second = split_records(reversed_records, seed=3)
    assert [record.record_id for record in first.train] == [
        record.record_id for record in second.train
    ]
    assert [record.record_id for record in first.validation] == [
        record.record_id for record in second.validation
    ]
    assert [record.record_id for record in first.test] == [
        record.record_id for record in second.test
    ]


def test_split_is_deterministic():
    records = _shared_parent_records()
    first = split_records(records, seed=5)
    second = split_records(records, seed=5)
    assert first == second


def test_class_counts_are_approximately_preserved():
    records = _shared_parent_records()
    split = split_records(records, seed=1)
    for name in ("train", "validation", "test"):
        bucket = getattr(split, name)
        counts = {
            label: sum(record.label == label for record in bucket)
            for label in ("am", "fm", "pm", "bpsk", "qpsk")
        }
        assert len(set(counts.values())) == 1
        assert min(counts.values()) >= 2


def test_generated_dataset_split_has_no_group_leak():
    dataset = generate_synthetic_dataset(
        SyntheticDatasetConfig(
            fs=8_000.0,
            n_samples_values=(32,),
            n_parents_per_class=6,
            n_variants_per_parent=2,
            seeds=DatasetSeeds(1, 2, 3, 4),
        )
    )
    split = split_dataset(dataset)
    seen = {}
    for name in ("train", "validation", "test"):
        for record in getattr(split, name):
            seen.setdefault(record.group_id, set()).add(name)
    assert all(len(names) == 1 for names in seen.values())
    assert len(split.all_records) == len(dataset.records)
    assert {record.record_id for record in split.all_records} == {
        record.record_id for record in dataset.records
    }


def _snr_pair_records():
    records = []
    for parent in range(6):
        group_id = f"bpsk:p{parent:06d}"
        records.append(
            _record("bpsk", group_id, f"{group_id}:v0000", snr_db=0.0)
        )
        records.append(
            _record("bpsk", group_id, f"{group_id}:v0001", snr_db=10.0)
        )
    return records


def test_filter_before_split_leaks_sibling_parents():
    records = _snr_pair_records()
    kept, held = filter_records(
        records, lambda record: record.nuisances.snr_db != 0.0
    )
    shared = {record.group_id for record in kept} & {
        record.group_id for record in held
    }
    assert shared
    leaky = DatasetSplit(train=kept, validation=(), test=held)
    with pytest.raises(ValueError, match="share parent groups"):
        combination_holdout(
            leaky,
            development=lambda record: record.nuisances.snr_db != 0.0,
            holdout=lambda record: record.nuisances.snr_db == 0.0,
        )


def test_combination_holdout_after_split_has_zero_shared_parents():
    split = split_records(_snr_pair_records(), seed=8)
    held = combination_holdout(
        split,
        development=lambda record: record.nuisances.snr_db != 0.0,
        holdout=lambda record: record.nuisances.snr_db == 0.0,
    )
    development_groups = {
        record.group_id for record in held.train + held.validation
    }
    holdout_groups = {record.group_id for record in held.test}
    assert development_groups.isdisjoint(holdout_groups)
    assert held.train or held.validation
    assert held.test
    assert all(
        record.nuisances.snr_db != 0.0
        for record in held.train + held.validation
    )
    assert all(record.nuisances.snr_db == 0.0 for record in held.test)


def test_filter_split_preserves_group_partition():
    split = split_records(_snr_pair_records(), seed=8)
    filtered = filter_split(
        split, lambda record: record.nuisances.snr_db == 10.0
    )
    membership = {}
    for name in ("train", "validation", "test"):
        for record in getattr(filtered, name):
            membership.setdefault(record.group_id, set()).add(name)
            assert record.nuisances.snr_db == 10.0
    assert membership
    assert all(len(names) == 1 for names in membership.values())


def test_same_group_same_label_stays_together():
    records = [
        _record("bpsk", "bpsk:p000000", "bpsk:p000000:v0000"),
        _record("bpsk", "bpsk:p000000", "bpsk:p000000:v0001"),
        _record("bpsk", "bpsk:p000001", "bpsk:p000001:v0000"),
        _record("bpsk", "bpsk:p000002", "bpsk:p000002:v0000"),
        _record("bpsk", "bpsk:p000003", "bpsk:p000003:v0000"),
        _record("bpsk", "bpsk:p000004", "bpsk:p000004:v0000"),
        _record("bpsk", "bpsk:p000005", "bpsk:p000005:v0000"),
    ]
    split = split_records(records, seed=2)
    membership = {}
    siblings = []
    for name in ("train", "validation", "test"):
        for record in getattr(split, name):
            membership.setdefault(record.group_id, set()).add(name)
            if record.group_id == "bpsk:p000000":
                siblings.append(record.record_id)
    assert membership["bpsk:p000000"] == {next(iter(membership["bpsk:p000000"]))}
    assert sorted(siblings) == ["bpsk:p000000:v0000", "bpsk:p000000:v0001"]
    assert all(len(names) == 1 for names in membership.values())


def test_cross_label_group_id_rejected():
    records = [
        _record("am", "SHARED", "am:SHARED:v0000"),
        _record("fm", "SHARED", "fm:SHARED:v0000"),
    ]
    with pytest.raises(ValueError, match="cross-label"):
        split_records(records, seed=0)


def test_duplicate_record_ids_rejected():
    records = [
        _record("bpsk", "bpsk:p000000", "bpsk:p000000:v0000"),
        _record("bpsk", "bpsk:p000001", "bpsk:p000000:v0000"),
        _record("bpsk", "bpsk:p000002", "bpsk:p000002:v0000"),
    ]
    with pytest.raises(ValueError, match="duplicate record_id"):
        split_records(records, seed=0)


def test_too_few_groups_rejected():
    records = [
        _record("bpsk", "bpsk:p000000", "bpsk:p000000:v0000"),
        _record("bpsk", "bpsk:p000001", "bpsk:p000001:v0000"),
    ]
    with pytest.raises(ValueError):
        split_records(records, seed=0)
