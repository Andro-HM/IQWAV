"""Module 12D2: train/validation fitting of the physics-rule AMC baseline.

Experiment only. Does not open the sealed test split. Does not create
classify_modulation(). Reuses frozen 12D1 feature definitions.
"""

from __future__ import annotations

import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _exp_12d1 import PRIMARY_LABELS  # noqa: E402
from _exp_12d2 import (  # noqa: E402
    NODE_A_FAMILIES,
    NODE_B_FAMILIES,
    NODE_C_FAMILIES,
    FittedRule,
    apply_rule,
    binary_balanced_accuracy,
    featurize_records,
    fit_node,
    node_masks,
    node_positive,
    predict_hierarchy,
    select_family,
)
from iqwav.amr import (  # noqa: E402
    DatasetSeeds,
    SyntheticDatasetConfig,
    evaluate_predictions,
    generate_synthetic_dataset,
    split_dataset,
)

FS = 48_000.0
ORIGINAL_SEEDS = DatasetSeeds(101, 202, 303, 404)
FRESH_SEEDS = DatasetSeeds(1001, 2002, 3003, 4004)
LENGTH_SEEDS = {
    128: DatasetSeeds(5101, 5202, 5303, 5404),
    256: DatasetSeeds(6101, 6202, 6303, 6404),
    512: DatasetSeeds(7101, 7202, 7303, 7404),
}
N_PARENTS = 30
N_VARIANTS = 5
CHANCE_4CLASS_BA = 0.25
CHANCE_BINARY_BA = 0.5
GATE_BA = 0.80
GATE_RECALL = 0.65
GATE_NODE_BA = 0.80
GATE_BUCKET_BA = 0.60


def _config(seeds: DatasetSeeds, n_samples: int) -> SyntheticDatasetConfig:
    return SyntheticDatasetConfig(
        fs=FS,
        n_samples_values=(n_samples,),
        n_parents_per_class=N_PARENTS,
        n_variants_per_parent=N_VARIANTS,
        seeds=seeds,
    )


def development_rows(seeds: DatasetSeeds, n_samples: int = 256):
    dataset = generate_synthetic_dataset(_config(seeds, n_samples))
    split = split_dataset(dataset)
    test_ids = {record.record_id for record in split.test}
    train_rows = featurize_records(split.train)
    val_rows = featurize_records(split.validation)
    if any(row["record_id"] in test_ids for row in train_rows + val_rows):
        raise RuntimeError("sealed test split leaked into development.")
    return {
        "train": train_rows,
        "val": val_rows,
        "n_test_sealed": len(split.test),
        "n_total": len(dataset.records),
    }


def _fmt(value: float) -> str:
    if not math.isfinite(value):
        return "nan"
    return f"{value:.4f}"


def print_report(name: str, report) -> None:
    print(f"  accuracy={report.accuracy:.4f}  balanced={report.balanced_accuracy:.4f}")
    print("  confusion (rows=true am,angle,bpsk,qpsk):")
    print(report.confusion_matrix)
    for label in PRIMARY_LABELS:
        print(
            f"  {label:<6} P={report.precision[label]:.4f} "
            f"R={report.recall[label]:.4f} F1={report.f1[label]:.4f}"
        )


def evaluate_rows(rows, rule_a, rule_b, rule_c):
    y_true = [row["primary"] for row in rows]
    y_pred = predict_hierarchy(rows, rule_a, rule_b, rule_c)
    report = evaluate_predictions(y_true, y_pred, labels=PRIMARY_LABELS)
    return y_true, y_pred, report


def node_ba_on_subset(rows, rule: FittedRule, node: str) -> float:
    mask = node_masks(rows)[node]
    subset = [row for row, keep in zip(rows, mask) if keep]
    if not subset:
        return float("nan")
    y = node_positive(subset, node)
    pred = apply_rule(subset, rule)
    return binary_balanced_accuracy(y, pred)


def cascaded_node_ba(rows, rule_a, rule_b, rule_c) -> dict[str, float]:
    am_flag = apply_rule(rows, rule_a)
    rest = [row for row, flag in zip(rows, am_flag) if not flag]
    psk_flag_rest = apply_rule(rest, rule_b) if rest else np.array([], dtype=bool)
    psk_rows = [row for row, flag in zip(rest, psk_flag_rest) if flag]
    metrics = {"A": node_ba_on_subset(rows, rule_a, "A")}
    if rest:
        y_b = node_positive(rest, "B")
        # leaked AM into rest: those rows are not PSK/ANGLE; exclude from B BA
        keep = np.array([row["primary"] != "am" for row in rest], dtype=bool)
        if np.any(keep):
            metrics["B"] = binary_balanced_accuracy(y_b[keep], psk_flag_rest[keep])
        else:
            metrics["B"] = float("nan")
    else:
        metrics["B"] = float("nan")
    if psk_rows:
        keep = np.array(
            [row["primary"] in ("bpsk", "qpsk") for row in psk_rows], dtype=bool
        )
        pred_c = apply_rule(psk_rows, rule_c)
        y_c = node_positive(psk_rows, "C")
        if np.any(keep):
            metrics["C"] = binary_balanced_accuracy(y_c[keep], pred_c[keep])
        else:
            metrics["C"] = float("nan")
    else:
        metrics["C"] = float("nan")
    return metrics


def breakdown(rows, y_true, y_pred, field: str) -> None:
    buckets = defaultdict(lambda: {"true": [], "pred": []})
    for row, truth, pred in zip(rows, y_true, y_pred):
        key = row[field]
        if key is None:
            continue
        buckets[key]["true"].append(truth)
        buckets[key]["pred"].append(pred)
    print(f"  by {field}:")
    for key in sorted(buckets, key=lambda item: (str(type(item)), item)):
        report = evaluate_predictions(
            buckets[key]["true"], buckets[key]["pred"], labels=PRIMARY_LABELS
        )
        print(
            f"    {key!s:<12} n={len(buckets[key]['true']):3d} "
            f"acc={report.accuracy:.4f} ba={report.balanced_accuracy:.4f} "
            + " ".join(
                f"{lab}R={report.recall[lab]:.2f}" for lab in PRIMARY_LABELS
            )
        )


def fit_original():
    print("=== Original development dataset (12D1 seeds, N=256) ===")
    data = development_rows(ORIGINAL_SEEDS, 256)
    train, val = data["train"], data["val"]
    print(
        f"total={data['n_total']} train={len(train)} val={len(val)} "
        f"test_sealed={data['n_test_sealed']}"
    )
    nodes = {
        "A": (NODE_A_FAMILIES, train, val),
        "B": (NODE_B_FAMILIES, [r for r in train if r["primary"] != "am"],
              [r for r in val if r["primary"] != "am"]),
        "C": (
            NODE_C_FAMILIES,
            [r for r in train if r["primary"] in ("bpsk", "qpsk")],
            [r for r in val if r["primary"] in ("bpsk", "qpsk")],
        ),
    }
    selected = {}
    print()
    print("Predeclared tie-break: max train BA, then simpler family,")
    print("then larger margin from class medians, then smaller t1, t2.")
    for node, (families, train_n, val_n) in nodes.items():
        y_tr = node_positive(train_n, node)
        fitted = fit_node(train_n, node, families, y_tr)
        val_scores = []
        print(f"\n--- Node {node} TRAIN fit / VAL frozen family scores ---")
        for rule in fitted:
            ba_val = binary_balanced_accuracy(
                node_positive(val_n, node), apply_rule(val_n, rule)
            )
            val_scores.append(ba_val)
            print(
                f"  {rule.family:<28} thr={tuple(round(t, 6) for t in rule.thresholds)} "
                f"trainBA={rule.train_ba:.4f} valBA={ba_val:.4f} "
                f"margin={rule.margin:.4f} simplicity={rule.simplicity}"
            )
        chosen = select_family(fitted, val_scores)
        selected[node] = chosen
        print(
            f"  SELECTED {chosen.family} thresholds={chosen.thresholds} "
            f"(trainBA={chosen.train_ba:.4f})"
        )
    return selected, train, val


def report_split(title: str, rows, rule_a, rule_b, rule_c, *, with_breakdown: bool):
    print()
    print(title)
    y_true, y_pred, report = evaluate_rows(rows, rule_a, rule_b, rule_c)
    print_report(title, report)
    node_ind = {
        "A": node_ba_on_subset(rows, rule_a, "A"),
        "B": node_ba_on_subset(rows, rule_b, "B"),
        "C": node_ba_on_subset(rows, rule_c, "C"),
    }
    node_cas = cascaded_node_ba(rows, rule_a, rule_b, rule_c)
    print(
        "  independent node BA: "
        + " ".join(f"{k}={node_ind[k]:.4f}" for k in ("A", "B", "C"))
    )
    print(
        "  cascaded node BA:    "
        + " ".join(f"{k}={node_cas[k]:.4f}" for k in ("A", "B", "C"))
    )
    fm_pm = defaultdict(lambda: defaultdict(int))
    for row, pred in zip(rows, y_pred):
        if row["label"] in ("fm", "pm"):
            fm_pm[row["label"]][pred] += 1
    print("  FM/PM diagnostic (true fm/pm -> predicted primary, no FM/PM rule):")
    for src in ("fm", "pm"):
        total = sum(fm_pm[src].values())
        dist = " ".join(f"{lab}={fm_pm[src][lab]}" for lab in PRIMARY_LABELS)
        print(f"    true {src} n={total} {dist}")
    if with_breakdown:
        breakdown(rows, y_true, y_pred, "snr_db")
        breakdown(rows, y_true, y_pred, "sps")
        breakdown(rows, y_true, y_pred, "family")
    return report, node_ind, y_true, y_pred


def gate_status(val_report, val_nodes, fresh_report, fresh_nodes, val_rows, val_true, val_pred) -> bool:
    print()
    print("=== Development gate (research criteria, not a product spec) ===")
    reasons = []
    if val_report.balanced_accuracy < GATE_BA:
        reasons.append(
            f"val BA {val_report.balanced_accuracy:.4f} < {GATE_BA}"
        )
    if fresh_report.balanced_accuracy < GATE_BA:
        reasons.append(
            f"fresh-seed BA {fresh_report.balanced_accuracy:.4f} < {GATE_BA}"
        )
    for lab in PRIMARY_LABELS:
        if val_report.recall[lab] < GATE_RECALL:
            reasons.append(f"val {lab} recall {val_report.recall[lab]:.4f} < {GATE_RECALL}")
        if fresh_report.recall[lab] < GATE_RECALL:
            reasons.append(
                f"fresh {lab} recall {fresh_report.recall[lab]:.4f} < {GATE_RECALL}"
            )
    for node in ("A", "B", "C"):
        if val_nodes[node] < GATE_NODE_BA:
            reasons.append(f"val node {node} BA {val_nodes[node]:.4f} < {GATE_NODE_BA}")
        if fresh_nodes[node] < GATE_NODE_BA:
            reasons.append(
                f"fresh node {node} BA {fresh_nodes[node]:.4f} < {GATE_NODE_BA}"
            )
    by_snr = defaultdict(lambda: {"true": [], "pred": []})
    for row, truth, pred in zip(val_rows, val_true, val_pred):
        by_snr[row["snr_db"]]["true"].append(truth)
        by_snr[row["snr_db"]]["pred"].append(pred)
    print("  SNR characterization (0 dB allowed to fail):")
    for snr in (0.0, 5.0, 10.0, 15.0, 20.0):
        if snr not in by_snr:
            continue
        report = evaluate_predictions(
            by_snr[snr]["true"], by_snr[snr]["pred"], labels=PRIMARY_LABELS
        )
        tag = "char" if snr == 0.0 else "gate"
        print(
            f"    SNR={snr:4.0f} n={len(by_snr[snr]['true']):3d} "
            f"acc={report.accuracy:.4f} ba={report.balanced_accuracy:.4f} [{tag}]"
        )
        if snr >= 5.0 and report.balanced_accuracy < GATE_BUCKET_BA:
            reasons.append(f"SNR {snr} val BA {report.balanced_accuracy:.4f} collapsed")
        if snr >= 5.0 and report.balanced_accuracy < CHANCE_4CLASS_BA + 0.15:
            reasons.append(f"SNR {snr} did not materially beat chance")
    passed = not reasons
    print("  result:", "PASS" if passed else "FAIL")
    for reason in reasons:
        print("   -", reason)
    if not passed:
        print("  Do NOT open the sealed test split.")
    return passed


def main() -> None:
    selected, train, val = fit_original()
    rule_a, rule_b, rule_c = selected["A"], selected["B"], selected["C"]
    print()
    print("=== Frozen selected hierarchy ===")
    for rule in (rule_a, rule_b, rule_c):
        print(
            f"  Node {rule.node}: {rule.family} "
            f"{rule.features} > {rule.thresholds}  combine={rule.combine}"
        )

    print()
    print("=== TRAIN four-class (reference only; families/thresholds from train nodes) ===")
    train_report, train_nodes, _, _ = report_split(
        "TRAIN hierarchy", train, rule_a, rule_b, rule_c, with_breakdown=False
    )

    val_report, val_nodes, val_true, val_pred = report_split(
        "VALIDATION four-class (frozen rules)",
        val,
        rule_a,
        rule_b,
        rule_c,
        with_breakdown=True,
    )

    print()
    print("=== Fresh-seed development dataset (no refit) ===")
    fresh = development_rows(FRESH_SEEDS, 256)
    print(
        f"total={fresh['n_total']} train={len(fresh['train'])} "
        f"val={len(fresh['val'])} test_sealed={fresh['n_test_sealed']}"
    )
    fresh_dev = fresh["train"] + fresh["val"]
    fresh_report, fresh_nodes, _, _ = report_split(
        "FRESH-SEED train+val hierarchy (frozen rules)",
        fresh_dev,
        rule_a,
        rule_b,
        rule_c,
        with_breakdown=True,
    )
    fresh_val_report, fresh_val_nodes, _, _ = report_split(
        "FRESH-SEED validation-only (frozen rules)",
        fresh["val"],
        rule_a,
        rule_b,
        rule_c,
        with_breakdown=False,
    )

    print()
    print("=== Record-length robustness (frozen rules, fresh seeds) ===")
    for n_samples, seeds in LENGTH_SEEDS.items():
        pack = development_rows(seeds, n_samples)
        dev = pack["train"] + pack["val"]
        _, _, report = evaluate_rows(dev, rule_a, rule_b, rule_c)
        nodes = {
            "A": node_ba_on_subset(dev, rule_a, "A"),
            "B": node_ba_on_subset(dev, rule_b, "B"),
            "C": node_ba_on_subset(dev, rule_c, "C"),
        }
        print(
            f"  N={n_samples:<3} n={len(dev):3d} acc={report.accuracy:.4f} "
            f"ba={report.balanced_accuracy:.4f} "
            + " ".join(f"{lab}R={report.recall[lab]:.2f}" for lab in PRIMARY_LABELS)
            + "  "
            + " ".join(f"{k}={nodes[k]:.3f}" for k in ("A", "B", "C"))
        )

    gate_status(
        val_report,
        val_nodes,
        fresh_report,
        fresh_nodes,
        val,
        val_true,
        val_pred,
    )
    print(
        "  (fresh-seed gate uses the full fresh train+val development pool; "
        f"fresh val-only BA={fresh_val_report.balanced_accuracy:.4f})"
    )
    print()
    print(
        f"Chance baselines: 4-class BA={CHANCE_4CLASS_BA}, "
        f"binary node BA={CHANCE_BINARY_BA}."
    )
    print("Sealed original test split was not opened.")


if __name__ == "__main__":
    main()
