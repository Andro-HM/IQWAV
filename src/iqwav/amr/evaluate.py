"""Classifier-independent AMC evaluation and shortcut audit."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from .dataset import LABELS, NuisanceMetadata, SyntheticRecord

__all__ = [
    "ClassificationReport",
    "FieldSummary",
    "NuisanceBucket",
    "ShortcutAudit",
    "accuracy_by_nuisance",
    "evaluate_predictions",
    "shortcut_audit",
]

# Discrete catalogs whose support and frequencies must not identify the class.
_DISCRETE_FIELDS = (
    "snr_db",
    "n_samples",
    "samples_per_symbol",
    "message_family",
)
_CONTINUOUS_FIELDS = ("amplitude", "phase_rad", "cfo_norm")
_SUPPORT_FIELDS = ("snr_db", "n_samples")
_DIGITAL_COMPARE = ("bpsk", "qpsk")
_ANALOG_COMPARE = ("am", "fm", "pm")


def _as_label_tuple(values: Sequence[str], name: str) -> tuple[str, ...]:
    if len(values) == 0:
        raise ValueError(f"{name} must be non-empty.")
    return tuple(values)


@dataclass(frozen=True)
class ClassificationReport:
    """Confusion matrix and per-class scores. No confidence calibration."""

    labels: tuple[str, ...]
    confusion_matrix: npt.NDArray[np.int64]
    accuracy: float
    balanced_accuracy: float
    precision: dict[str, float]
    recall: dict[str, float]
    f1: dict[str, float]


def evaluate_predictions(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    *,
    labels: tuple[str, ...] = LABELS,
) -> ClassificationReport:
    """Score predicted labels against truth.

    Rows of the confusion matrix are true labels, columns are predicted
    labels, both in ``labels`` order. Undefined precision/recall/F1
    (zero denominator) is reported as 0.0.

    ``y_true`` and ``y_pred`` must be non-empty; empty truth is rejected
    and has no balanced-accuracy result.

    For a valid non-empty evaluation, balanced accuracy is the
    unweighted mean of per-class recall over classes that have nonzero
    truth support. Classes listed in ``labels`` but absent from
    ``y_true`` remain in the per-class precision/recall/F1 outputs with
    recall 0.0 and are excluded from that mean.
    """
    y_true = _as_label_tuple(y_true, "y_true")
    y_pred = _as_label_tuple(y_pred, "y_pred")
    if len(y_true) != len(y_pred):
        raise ValueError(
            f"y_true and y_pred must have the same length, got "
            f"{len(y_true)} and {len(y_pred)}."
        )
    if not labels:
        raise ValueError("labels must be a non-empty tuple.")
    if len(set(labels)) != len(labels):
        raise ValueError("labels must be unique.")
    index = {label: i for i, label in enumerate(labels)}
    unknown = [label for label in y_true + y_pred if label not in index]
    if unknown:
        raise ValueError(f"labels not in the evaluation set: {unknown!r}.")

    n_class = len(labels)
    confusion = np.zeros((n_class, n_class), dtype=np.int64)
    for truth, pred in zip(y_true, y_pred):
        confusion[index[truth], index[pred]] += 1

    total = int(confusion.sum())
    accuracy = float(np.trace(confusion) / total) if total else 0.0
    precision: dict[str, float] = {}
    recall: dict[str, float] = {}
    f1: dict[str, float] = {}
    recalls = []
    for i, label in enumerate(labels):
        tp = float(confusion[i, i])
        pred_pos = float(confusion[:, i].sum())
        true_pos = float(confusion[i, :].sum())
        prec = tp / pred_pos if pred_pos else 0.0
        rec = tp / true_pos if true_pos else 0.0
        precision[label] = prec
        recall[label] = rec
        f1[label] = (
            2.0 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        )
        if true_pos:
            recalls.append(rec)
    balanced = float(np.mean(recalls)) if recalls else 0.0
    return ClassificationReport(
        labels=tuple(labels),
        confusion_matrix=confusion,
        accuracy=accuracy,
        balanced_accuracy=balanced,
        precision=precision,
        recall=recall,
        f1=f1,
    )


@dataclass(frozen=True)
class NuisanceBucket:
    """Accuracy and support count for one nuisance value."""

    accuracy: float
    n: int


def accuracy_by_nuisance(
    records: Sequence[SyntheticRecord],
    y_true: Sequence[str],
    y_pred: Sequence[str],
    field: str,
) -> dict[object, NuisanceBucket]:
    """Accuracy grouped by a :class:`NuisanceMetadata` field such as SNR.

    Each bucket reports ``accuracy`` and support count ``n``.
    """
    if field not in NuisanceMetadata.__dataclass_fields__:
        raise ValueError(f"unknown nuisance field {field!r}.")
    y_true = _as_label_tuple(y_true, "y_true")
    y_pred = _as_label_tuple(y_pred, "y_pred")
    if not (len(records) == len(y_true) == len(y_pred)):
        raise ValueError("records, y_true, and y_pred must have the same length.")
    correct: dict[object, list[int]] = {}
    for record, truth, pred in zip(records, y_true, y_pred):
        key = getattr(record.nuisances, field)
        bucket = correct.setdefault(key, [0, 0])
        bucket[1] += 1
        if truth == pred:
            bucket[0] += 1
    return {
        key: NuisanceBucket(accuracy=hits / count, n=count)
        for key, (hits, count) in correct.items()
    }


@dataclass(frozen=True)
class FieldSummary:
    """Observed values of one nuisance field inside one class.

    Discrete catalogs populate ``unique`` and ``counts``. ``counts``
    entries are absolute occurrence counts ``(value, n)``, not
    normalized frequencies. Continuous fields populate ``minimum`` /
    ``maximum`` / ``mean`` and leave ``unique``/``counts`` empty. This
    is an audit, not a hypothesis test.
    """

    n: int
    unique: tuple
    counts: tuple[tuple[object, int], ...]
    minimum: float | None
    maximum: float | None
    mean: float | None


@dataclass(frozen=True)
class ShortcutAudit:
    """Per-class nuisance summaries and obvious support/frequency mismatches."""

    by_class: dict[str, dict[str, FieldSummary]]
    mismatches: tuple[str, ...]

    def format_report(self) -> str:
        lines: list[str] = []
        for label, fields in self.by_class.items():
            lines.append(f"class {label}")
            for name, summary in fields.items():
                if summary.counts:
                    lines.append(
                        f"  {name}: n={summary.n} counts={summary.counts}"
                    )
                else:
                    lines.append(
                        f"  {name}: n={summary.n} min={summary.minimum} "
                        f"max={summary.maximum} mean={summary.mean}"
                    )
        if self.mismatches:
            lines.append("mismatches:")
            lines.extend(f"  {item}" for item in self.mismatches)
        else:
            lines.append("mismatches: none")
        return "\n".join(lines)


def _pythonize(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    return value


def _frequency_table(values: list) -> tuple[tuple[object, int], ...]:
    counts: dict[object, int] = {}
    for value in values:
        key = _pythonize(value)
        counts[key] = counts.get(key, 0) + 1
    return tuple(sorted(counts.items(), key=lambda item: repr(item[0])))


def _numeric_values(values: list) -> list[float]:
    numeric: list[float] = []
    for value in values:
        value = _pythonize(value)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            numeric.append(float(value))
    return numeric


def _summarize_discrete(values: list) -> FieldSummary:
    counts = _frequency_table(values)
    unique = tuple(key for key, _count in counts)
    numeric = _numeric_values(values)
    return FieldSummary(
        n=len(values),
        unique=unique,
        counts=counts,
        minimum=min(numeric) if numeric else None,
        maximum=max(numeric) if numeric else None,
        mean=None,
    )


def _summarize_continuous(values: list) -> FieldSummary:
    numeric = _numeric_values(values)
    if not numeric:
        return FieldSummary(
            n=len(values),
            unique=(),
            counts=(),
            minimum=None,
            maximum=None,
            mean=None,
        )
    return FieldSummary(
        n=len(numeric),
        unique=(),
        counts=(),
        minimum=min(numeric),
        maximum=max(numeric),
        mean=float(np.mean(numeric)),
    )


def _counts(audit_fields: dict[str, FieldSummary], name: str) -> dict[object, int]:
    if name not in audit_fields:
        return {}
    return dict(audit_fields[name].counts)


def _flag_discrete_mismatch(
    mismatches: list[str],
    by_class: dict[str, dict[str, FieldSummary]],
    labels: Sequence[str],
    field: str,
) -> None:
    nonempty = {
        label: _counts(by_class[label], field)
        for label in labels
        if label in by_class and _counts(by_class[label], field)
    }
    if len(nonempty) < 2:
        return
    reference_label, reference = next(iter(nonempty.items()))
    reference_support = set(reference)
    for label, counts in nonempty.items():
        support = set(counts)
        if support != reference_support:
            mismatches.append(
                f"{field} support differs: {reference_label}="
                f"{sorted(reference_support, key=repr)} vs {label}="
                f"{sorted(support, key=repr)}"
            )
            return
        if counts != reference:
            mismatches.append(
                f"{field} frequencies differ: {reference_label}={reference} "
                f"vs {label}={counts}"
            )
            return


def shortcut_audit(records: Sequence[SyntheticRecord]) -> ShortcutAudit:
    """Summarize nuisance support and frequencies by class.

    This is a leak screen, not a hypothesis test. Discrete catalogs
    (SNR, SPS, record length, message family) get frequency tables.
    Amplitude, phase, and CFO get count/min/max/mean only.

    Full-domain FM versus PM scores must later be accompanied by
    evaluation in an overlapping effective-excursion region; this audit
    does not define that region.
    """
    values: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for record in records:
        nuisances = record.nuisances
        for field in nuisances.__dataclass_fields__:
            value = getattr(nuisances, field)
            if value is None:
                continue
            values[record.label][field].append(value)

    by_class: dict[str, dict[str, FieldSummary]] = {}
    for label in sorted(
        values,
        key=lambda item: (LABELS.index(item) if item in LABELS else 100, item),
    ):
        summaries: dict[str, FieldSummary] = {}
        for field, field_values in values[label].items():
            if field in _CONTINUOUS_FIELDS:
                summaries[field] = _summarize_continuous(field_values)
            elif field in _DISCRETE_FIELDS:
                summaries[field] = _summarize_discrete(field_values)
            else:
                continue
        by_class[label] = summaries

    mismatches: list[str] = []
    present = [label for label in LABELS if label in by_class]
    for field in _SUPPORT_FIELDS:
        _flag_discrete_mismatch(mismatches, by_class, present, field)

    digital = [label for label in _DIGITAL_COMPARE if label in by_class]
    _flag_discrete_mismatch(mismatches, by_class, digital, "samples_per_symbol")

    analog = [label for label in _ANALOG_COMPARE if label in by_class]
    _flag_discrete_mismatch(mismatches, by_class, analog, "message_family")

    return ShortcutAudit(by_class=by_class, mismatches=tuple(mismatches))
