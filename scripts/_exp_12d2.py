"""Private Module 12D2 hierarchical rule fitting.

Experiment only. Not a production classify_modulation() API.
Thresholds are fit on TRAIN only. Feature mathematics stay 12D1.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from _exp_12d1 import extract_features, primary_label

# Predeclared family order is the simplicity ranking (lower = simpler).
NODE_A_FAMILIES = (
    ("coherence", ("envelope_coherence",), "single"),
    ("Ev", ("envelope_dispersion",), "single"),
    ("coherence_AND_Ev", ("envelope_coherence", "envelope_dispersion"), "and"),
    ("coherence_OR_Ev", ("envelope_coherence", "envelope_dispersion"), "or"),
)
NODE_B_FAMILIES = (
    ("sparsity", ("phase_sparsity",), "single"),
    ("periodicity", ("transition_periodicity",), "single"),
    ("sparsity_AND_periodicity", ("phase_sparsity", "transition_periodicity"), "and"),
    ("sparsity_OR_periodicity", ("phase_sparsity", "transition_periodicity"), "or"),
)
NODE_C_FAMILIES = (
    ("C2", ("c2",), "single"),
    ("C2_over_C4", ("c2_over_c4",), "single"),
    ("C2_AND_C2_over_C4", ("c2", "c2_over_c4"), "and"),
    ("C2_OR_C2_over_C4", ("c2", "c2_over_c4"), "or"),
)

MAX_UNIQUE_THRESHOLDS = 128
PRIMARY_LABELS = ("am", "angle", "bpsk", "qpsk")


@dataclass(frozen=True)
class FittedRule:
    node: str
    family: str
    features: tuple[str, ...]
    combine: str
    thresholds: tuple[float, ...]
    train_ba: float
    margin: float
    simplicity: int


def featurize_records(records) -> list[dict]:
    rows = []
    for record in records:
        feats = extract_features(record.samples)
        rows.append(
            {
                "record_id": record.record_id,
                "label": record.label,
                "primary": primary_label(record.label),
                "snr_db": record.nuisances.snr_db,
                "sps": record.nuisances.samples_per_symbol,
                "family": record.nuisances.message_family,
                "n_samples": record.nuisances.n_samples,
                **feats,
            }
        )
    return rows


def binary_balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean recall over classes with nonzero truth support."""
    recalls = []
    for value in (True, False):
        mask = y_true == value
        if not np.any(mask):
            continue
        recalls.append(float(np.mean(y_pred[mask] == value)))
    if not recalls:
        return 0.0
    return float(np.mean(recalls))


def candidate_thresholds(values: np.ndarray) -> np.ndarray:
    """Adjacent unique midpoints, or a 101-point quantile grid if too large."""
    values = np.asarray(values, dtype=np.float64)
    uniq = np.unique(values)
    if uniq.size == 1:
        return np.array([uniq[0] - 1.0, uniq[0]], dtype=np.float64)
    if uniq.size - 1 > MAX_UNIQUE_THRESHOLDS:
        uniq = np.unique(np.quantile(values, np.linspace(0.0, 1.0, 101)))
        if uniq.size == 1:
            return np.array([uniq[0] - 1.0, uniq[0]], dtype=np.float64)
    mids = 0.5 * (uniq[:-1] + uniq[1:])
    below = uniq[0] - (abs(uniq[0]) + 1.0) * 1e-9
    return np.unique(np.concatenate(([below], mids, [uniq[-1]])))


def _margin(threshold: float, med_pos: float, med_neg: float) -> float:
    lo, hi = sorted((med_neg, med_pos))
    if lo < threshold < hi:
        return float(min(threshold - lo, hi - threshold))
    return float(-min(abs(threshold - med_pos), abs(threshold - med_neg)))


def _rule_margin(thresholds: tuple[float, ...], med_pos, med_neg) -> float:
    return float(
        min(
            _margin(thr, med_pos[i], med_neg[i])
            for i, thr in enumerate(thresholds)
        )
    )


def _apply_combine(flags: list[np.ndarray], combine: str) -> np.ndarray:
    if combine == "single":
        return flags[0]
    if combine == "and":
        return flags[0] & flags[1]
    if combine == "or":
        return flags[0] | flags[1]
    raise ValueError(f"unknown combine {combine!r}")


def apply_rule(rows: list[dict], rule: FittedRule) -> np.ndarray:
    flags = [
        np.array([row[name] > thr for row in rows], dtype=bool)
        for name, thr in zip(rule.features, rule.thresholds)
    ]
    return _apply_combine(flags, rule.combine)


def fit_family(
    rows: list[dict],
    *,
    node: str,
    family: str,
    features: tuple[str, ...],
    combine: str,
    simplicity: int,
    y_positive: np.ndarray,
) -> FittedRule:
    arrays = [np.array([row[name] for row in rows], dtype=np.float64) for name in features]
    grids = [candidate_thresholds(arr) for arr in arrays]
    med_pos = []
    med_neg = []
    pos = y_positive
    neg = ~y_positive
    for arr in arrays:
        med_pos.append(float(np.median(arr[pos])) if np.any(pos) else 0.0)
        med_neg.append(float(np.median(arr[neg])) if np.any(neg) else 0.0)
    best = None
    if combine == "single":
        pairs = ((float(t),) for t in grids[0])
    else:
        pairs = ((float(t1), float(t2)) for t1 in grids[0] for t2 in grids[1])
    for thresholds in pairs:
        flags = [arr > thr for arr, thr in zip(arrays, thresholds)]
        pred = _apply_combine(flags, combine)
        ba = binary_balanced_accuracy(y_positive, pred)
        margin = _rule_margin(thresholds, med_pos, med_neg)
        key = (
            ba,
            -simplicity,
            margin,
            -thresholds[0],
            -(thresholds[1] if len(thresholds) > 1 else 0.0),
        )
        if best is None or key > best[0]:
            best = (key, thresholds, ba, margin)
    _, thresholds, ba, margin = best
    return FittedRule(
        node=node,
        family=family,
        features=features,
        combine=combine,
        thresholds=thresholds,
        train_ba=ba,
        margin=margin,
        simplicity=simplicity,
    )


def fit_node(rows: list[dict], node: str, families, y_positive: np.ndarray) -> list[FittedRule]:
    fitted = []
    for simplicity, (family, features, combine) in enumerate(families):
        fitted.append(
            fit_family(
                rows,
                node=node,
                family=family,
                features=features,
                combine=combine,
                simplicity=simplicity,
                y_positive=y_positive,
            )
        )
    return fitted


def select_family(fitted: list[FittedRule], val_ba: list[float]) -> FittedRule:
    """Choose a frozen family from validation BA; no threshold refit.

    Ties: simpler family, then larger TRAIN margin, then family name.
    """
    best = None
    for rule, ba in zip(fitted, val_ba):
        key = (ba, -rule.simplicity, rule.margin, rule.family)
        if best is None or key > best[0]:
            best = (key, rule)
    return best[1]


def predict_hierarchy(rows: list[dict], rule_a: FittedRule, rule_b: FittedRule, rule_c: FittedRule) -> list[str]:
    am_flag = apply_rule(rows, rule_a)
    psk_flag = apply_rule(rows, rule_b)
    bpsk_flag = apply_rule(rows, rule_c)
    labels = []
    for i, _row in enumerate(rows):
        if am_flag[i]:
            labels.append("am")
        elif not psk_flag[i]:
            labels.append("angle")
        elif bpsk_flag[i]:
            labels.append("bpsk")
        else:
            labels.append("qpsk")
    return labels


def node_masks(rows: list[dict]) -> dict[str, np.ndarray]:
    primary = np.array([row["primary"] for row in rows])
    return {
        "A": np.ones(len(rows), dtype=bool),
        "B": primary != "am",
        "C": (primary == "bpsk") | (primary == "qpsk"),
    }


def node_positive(rows: list[dict], node: str) -> np.ndarray:
    primary = np.array([row["primary"] for row in rows])
    if node == "A":
        return primary == "am"
    if node == "B":
        return (primary == "bpsk") | (primary == "qpsk")
    if node == "C":
        return primary == "bpsk"
    raise ValueError(node)
