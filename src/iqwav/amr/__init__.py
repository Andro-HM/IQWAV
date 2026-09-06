"""Automatic modulation recognition utilities.

Module 12C provides leakage-safe synthetic dataset generation and
classifier-independent evaluation. It does not include a classifier.

Combination holdouts must preserve parent grouping: split groups first
(fractions apply to groups), then filter nuisances inside
train/validation/test. Current group IDs assume information
realizations are not shared across labels. FM/PM full-domain scores
must later be accompanied by evaluation in an overlapping
effective-excursion region.
"""

from .dataset import (
    LABELS,
    DatasetSeeds,
    NuisanceMetadata,
    SyntheticDataset,
    SyntheticDatasetConfig,
    SyntheticRecord,
    classifier_labels,
    classifier_samples,
    generate_synthetic_dataset,
)
from .evaluate import (
    ClassificationReport,
    FieldSummary,
    NuisanceBucket,
    ShortcutAudit,
    accuracy_by_nuisance,
    evaluate_predictions,
    shortcut_audit,
)
from .messages import MESSAGE_FAMILIES, generate_analog_message
from .split import (
    DatasetSplit,
    combination_holdout,
    filter_records,
    filter_split,
    split_dataset,
    split_records,
)

__all__ = [
    "LABELS",
    "MESSAGE_FAMILIES",
    "ClassificationReport",
    "DatasetSeeds",
    "DatasetSplit",
    "FieldSummary",
    "NuisanceBucket",
    "NuisanceMetadata",
    "ShortcutAudit",
    "SyntheticDataset",
    "SyntheticDatasetConfig",
    "SyntheticRecord",
    "accuracy_by_nuisance",
    "classifier_labels",
    "classifier_samples",
    "combination_holdout",
    "evaluate_predictions",
    "filter_records",
    "filter_split",
    "generate_analog_message",
    "generate_synthetic_dataset",
    "shortcut_audit",
    "split_dataset",
    "split_records",
]
