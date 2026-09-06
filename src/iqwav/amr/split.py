"""Group-aware deterministic dataset splitting."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from .dataset import LABELS, DatasetSeeds, SyntheticDataset, SyntheticRecord

__all__ = [
    "DatasetSplit",
    "combination_holdout",
    "filter_records",
    "filter_split",
    "split_dataset",
    "split_records",
]


@dataclass(frozen=True)
class DatasetSplit:
    """Train/validation/test partitions with no shared group_id.

    Combination holdouts must preserve this parent grouping: split
    first, then filter nuisances inside each partition. Filtering the
    full pool before splitting can place sibling variants of one parent
    into development and holdout at the same time.
    """

    train: tuple[SyntheticRecord, ...]
    validation: tuple[SyntheticRecord, ...]
    test: tuple[SyntheticRecord, ...]

    @property
    def all_records(self) -> tuple[SyntheticRecord, ...]:
        return self.train + self.validation + self.test


def _record_ids(records: Sequence[SyntheticRecord]) -> list[str]:
    return [record.record_id for record in records]


def _reject_duplicate_record_ids(records: Sequence[SyntheticRecord]) -> None:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for record_id in _record_ids(records):
        if record_id in seen:
            duplicates.add(record_id)
        seen.add(record_id)
    if duplicates:
        raise ValueError(
            "duplicate record_id values are not allowed: "
            f"{sorted(duplicates)!r}."
        )


def _reject_cross_label_groups(records: Sequence[SyntheticRecord]) -> None:
    """Reject a group_id that appears under more than one label.

    Same-label sibling variants of one group remain valid. Cross-label
    group IDs are invalid under the current dataset model and are not
    globally reallocated.
    """
    labels_by_group: dict[str, set[str]] = {}
    for record in records:
        labels_by_group.setdefault(record.group_id, set()).add(record.label)
    shared = {
        group_id: tuple(sorted(labels))
        for group_id, labels in labels_by_group.items()
        if len(labels) > 1
    }
    if shared:
        raise ValueError(
            "each group_id must belong to exactly one label; "
            f"cross-label groups are invalid: {shared!r}."
        )


def _groups(records: Sequence[SyntheticRecord]) -> set[str]:
    return {record.group_id for record in records}


def filter_records(
    records: Sequence[SyntheticRecord],
    predicate: Callable[[SyntheticRecord], bool],
) -> tuple[tuple[SyntheticRecord, ...], tuple[SyntheticRecord, ...]]:
    """Low-level predicate partition of a record sequence.

    Returns ``(kept, excluded)``. This is not a group-aware split and
    is not a combination holdout. Filtering the full pool before
    :func:`split_records` can leak sibling variants of one parent into
    both a development set and a later holdout set.

    Canonical combination-holdout protocol:

    1. :func:`split_records` / :func:`split_dataset` (groups first)
    2. :func:`combination_holdout` or :func:`filter_split` inside
       train/validation/test
    """
    kept: list[SyntheticRecord] = []
    excluded: list[SyntheticRecord] = []
    for record in records:
        if predicate(record):
            kept.append(record)
        else:
            excluded.append(record)
    return tuple(kept), tuple(excluded)


def filter_split(
    split: DatasetSplit,
    predicate: Callable[[SyntheticRecord], bool],
) -> DatasetSplit:
    """Apply one nuisance predicate inside each already-split partition.

    Does not reassign groups. Call only after a group-aware split.
    """
    train, _ = filter_records(split.train, predicate)
    validation, _ = filter_records(split.validation, predicate)
    test, _ = filter_records(split.test, predicate)
    return DatasetSplit(train=train, validation=validation, test=test)


def combination_holdout(
    split: DatasetSplit,
    *,
    development: Callable[[SyntheticRecord], bool],
    holdout: Callable[[SyntheticRecord], bool],
) -> DatasetSplit:
    """Group-safe unseen-SNR / held-out-SPS / held-out-frequency protocol.

    Canonical order:

    1. group-aware split
    2. keep ``development`` records in train and validation
    3. keep ``holdout`` records in test
    4. reject any remaining shared parent group

    Because groups are assigned before filtering, sibling variants of
    one parent cannot appear in both development and holdout unless the
    input ``split`` was already leaky (for example after
    filter-before-split). That case is rejected.
    """
    train, _ = filter_records(split.train, development)
    validation, _ = filter_records(split.validation, development)
    test, _ = filter_records(split.test, holdout)
    development_groups = _groups(train) | _groups(validation)
    holdout_groups = _groups(test)
    shared = development_groups & holdout_groups
    if shared:
        raise ValueError(
            "combination holdout would share parent groups across "
            f"development and holdout: {sorted(shared)!r}. "
            "Split groups first, then filter inside train/validation/test; "
            "do not filter the full pool before splitting."
        )
    return DatasetSplit(train=train, validation=validation, test=test)


def _label_sort_key(label: str) -> tuple[int, str]:
    if label in LABELS:
        return (LABELS.index(label), label)
    return (len(LABELS), label)


def _split_counts(n_groups: int, fractions: tuple[float, float, float]) -> tuple[int, int, int]:
    if n_groups < 3:
        raise ValueError(
            "each class must have at least 3 groups to fill train, "
            f"validation, and test, got {n_groups}."
        )
    # Guarantee one group in each split, then largest-remainder the rest.
    remaining = n_groups - 3
    raw = [frac * remaining for frac in fractions]
    floors = [int(value) for value in raw]
    leftover = remaining - sum(floors)
    order = sorted(
        range(3),
        key=lambda index: (raw[index] - floors[index], -index),
        reverse=True,
    )
    for index in order[:leftover]:
        floors[index] += 1
    return floors[0] + 1, floors[1] + 1, floors[2] + 1


def split_records(
    records: Sequence[SyntheticRecord],
    *,
    fractions: tuple[float, float, float] = (0.6, 0.2, 0.2),
    seed: int = 4,
) -> DatasetSplit:
    """Assign records by parent ``group_id``, not by record order.

    ``fractions`` apply to groups, not to records. Groups are sorted by
    id, shuffled with ``seed``, and allocated per class. Every record
    appears in exactly one split. A group is never split across
    train/validation/test. Duplicate ``record_id`` values are rejected.
    A ``group_id`` that appears under more than one label is rejected
    before any assignment; same-label sibling records stay together.

    Combination holdouts must be applied after this split, inside the
    partitions, so sibling variants cannot cross development and holdout.
    """
    if len(fractions) != 3:
        raise ValueError("fractions must be (train, validation, test).")
    if any(
        not isinstance(value, (int, float)) or isinstance(value, bool)
        for value in fractions
    ):
        raise ValueError("fractions must be real numbers.")
    if abs(sum(fractions) - 1.0) > 1e-12 or any(value <= 0.0 for value in fractions):
        raise ValueError(
            "fractions must be positive and sum to 1, got "
            f"{fractions!r}."
        )
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise ValueError(f"seed must be a non-negative integer, got {seed!r}.")
    seed = int(seed)
    if seed < 0:
        raise ValueError(f"seed must be a non-negative integer, got {seed!r}.")
    _reject_duplicate_record_ids(records)
    _reject_cross_label_groups(records)

    by_label: dict[str, dict[str, list[SyntheticRecord]]] = {}
    for record in records:
        label_groups = by_label.setdefault(record.label, {})
        label_groups.setdefault(record.group_id, []).append(record)

    rng = np.random.default_rng(seed)
    train: list[SyntheticRecord] = []
    validation: list[SyntheticRecord] = []
    test: list[SyntheticRecord] = []
    for label in sorted(by_label, key=_label_sort_key):
        group_ids = sorted(by_label[label])
        rng.shuffle(group_ids)
        n_train, n_val, n_test = _split_counts(len(group_ids), fractions)
        if n_train + n_val + n_test != len(group_ids):
            raise RuntimeError("internal split allocation mismatch.")
        assignments = (
            (group_ids[:n_train], train),
            (group_ids[n_train : n_train + n_val], validation),
            (group_ids[n_train + n_val :], test),
        )
        for selected, bucket in assignments:
            for group_id in selected:
                members = sorted(
                    by_label[label][group_id],
                    key=lambda record: record.record_id,
                )
                bucket.extend(members)

    return DatasetSplit(
        train=tuple(train),
        validation=tuple(validation),
        test=tuple(test),
    )


def split_dataset(
    dataset: SyntheticDataset,
    *,
    fractions: tuple[float, float, float] = (0.6, 0.2, 0.2),
    seed: int | None = None,
) -> DatasetSplit:
    """Split a generated dataset using its split seed by default.

    ``fractions`` apply to parent groups. Nuisance holdouts belong in
    :func:`combination_holdout` after this call.
    """
    if seed is None:
        seed = dataset.config.seeds.split
    if not isinstance(dataset.config.seeds, DatasetSeeds):
        raise ValueError("dataset.config.seeds must be DatasetSeeds.")
    if dataset.config.n_parents_per_class < 3:
        raise ValueError(
            "n_parents_per_class must be >= 3 to fill train/validation/test."
        )
    return split_records(dataset.records, fractions=fractions, seed=seed)
