"""Tests for classifier-independent metrics and shortcut audit."""

import numpy as np
import pytest

from iqwav.amr import (
    LABELS,
    NuisanceMetadata,
    SyntheticRecord,
    accuracy_by_nuisance,
    evaluate_predictions,
    shortcut_audit,
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


def _record(label, record_id, **nuisance_kwargs):
    n_samples = nuisance_kwargs.get("n_samples", 8)
    samples = np.ones(n_samples, dtype=np.complex128)
    return SyntheticRecord(
        samples=samples,
        label=label,
        fs=8_000.0,
        group_id=record_id.rsplit(":", 1)[0],
        record_id=record_id,
        nuisances=_nuisances(**nuisance_kwargs),
    )


def test_confusion_matrix_and_scores_on_known_truth():
    y_true = ["am", "am", "fm", "fm", "pm"]
    y_pred = ["am", "fm", "fm", "fm", "pm"]
    labels = ("am", "fm", "pm")
    report = evaluate_predictions(y_true, y_pred, labels=labels)
    np.testing.assert_array_equal(
        report.confusion_matrix,
        np.array(
            [
                [1, 1, 0],
                [0, 2, 0],
                [0, 0, 1],
            ]
        ),
    )
    assert report.accuracy == pytest.approx(4 / 5)
    assert report.recall["am"] == pytest.approx(0.5)
    assert report.recall["fm"] == pytest.approx(1.0)
    assert report.precision["fm"] == pytest.approx(2 / 3)
    assert report.f1["pm"] == pytest.approx(1.0)
    assert report.balanced_accuracy == pytest.approx((0.5 + 1.0 + 1.0) / 3)


def test_balanced_accuracy_excludes_zero_truth_support():
    y_true = ["am", "am", "fm"]
    y_pred = ["am", "fm", "fm"]
    report = evaluate_predictions(y_true, y_pred, labels=("am", "fm", "pm"))
    assert report.recall["pm"] == 0.0
    assert report.precision["pm"] == 0.0
    assert report.balanced_accuracy == pytest.approx((0.5 + 1.0) / 2)
    assert report.balanced_accuracy != pytest.approx((0.5 + 1.0 + 0.0) / 3)


def test_accuracy_grouped_by_snr():
    records = [
        _record("am", "am:p0:v0", snr_db=0.0),
        _record("am", "am:p1:v0", snr_db=0.0),
        _record("fm", "fm:p0:v0", snr_db=10.0),
        _record("fm", "fm:p1:v0", snr_db=10.0),
    ]
    y_true = ["am", "am", "fm", "fm"]
    y_pred = ["am", "fm", "fm", "fm"]
    by_snr = accuracy_by_nuisance(records, y_true, y_pred, "snr_db")
    assert by_snr[0.0].accuracy == pytest.approx(0.5)
    assert by_snr[0.0].n == 2
    assert by_snr[10.0].accuracy == pytest.approx(1.0)
    assert by_snr[10.0].n == 2


def test_audit_flags_class_specific_sps():
    records = [
        _record("bpsk", "bpsk:p0:v0", samples_per_symbol=4),
        _record("bpsk", "bpsk:p1:v0", samples_per_symbol=4),
        _record("qpsk", "qpsk:p0:v0", samples_per_symbol=16),
        _record("qpsk", "qpsk:p1:v0", samples_per_symbol=16),
    ]
    audit = shortcut_audit(records)
    assert any("samples_per_symbol" in item for item in audit.mismatches)


def test_audit_flags_fm_pm_family_mismatch():
    records = [
        _record("fm", "fm:p0:v0", message_family="tone"),
        _record("fm", "fm:p1:v0", message_family="tone"),
        _record("pm", "pm:p0:v0", message_family="bandlimited"),
        _record("pm", "pm:p1:v0", message_family="bandlimited"),
    ]
    audit = shortcut_audit(records)
    assert any("message_family" in item for item in audit.mismatches)


def test_audit_flags_class_specific_record_length():
    records = [
        _record("am", "am:p0:v0", n_samples=32),
        _record("bpsk", "bpsk:p0:v0", n_samples=64),
    ]
    audit = shortcut_audit(records)
    assert any("n_samples" in item for item in audit.mismatches)


def test_audit_flags_class_specific_snr_catalog():
    records = [
        _record("am", "am:p0:v0", snr_db=0.0),
        _record("fm", "fm:p0:v0", snr_db=20.0),
    ]
    audit = shortcut_audit(records)
    assert any("snr_db" in item for item in audit.mismatches)


def test_audit_exposes_identical_support_different_frequencies():
    records = [
        _record("bpsk", "bpsk:p0:v0", samples_per_symbol=4),
        _record("bpsk", "bpsk:p1:v0", samples_per_symbol=4),
        _record("bpsk", "bpsk:p2:v0", samples_per_symbol=4),
        _record("bpsk", "bpsk:p3:v0", samples_per_symbol=8),
        _record("qpsk", "qpsk:p0:v0", samples_per_symbol=4),
        _record("qpsk", "qpsk:p1:v0", samples_per_symbol=8),
        _record("qpsk", "qpsk:p2:v0", samples_per_symbol=8),
        _record("qpsk", "qpsk:p3:v0", samples_per_symbol=8),
    ]
    audit = shortcut_audit(records)
    bpsk = dict(audit.by_class["bpsk"]["samples_per_symbol"].counts)
    qpsk = dict(audit.by_class["qpsk"]["samples_per_symbol"].counts)
    assert set(bpsk) == set(qpsk) == {4, 8}
    assert bpsk != qpsk
    assert any(
        "frequencies differ" in item and "samples_per_symbol" in item
        for item in audit.mismatches
    )


def test_audit_reports_discrete_counts_and_continuous_moments():
    records = [
        _record(
            "am",
            "am:p0:v0",
            snr_db=5.0,
            amplitude=1.0,
            phase_rad=0.0,
            cfo_norm=-0.01,
            message_family="tone",
        ),
        _record(
            "am",
            "am:p1:v0",
            snr_db=5.0,
            amplitude=3.0,
            phase_rad=2.0,
            cfo_norm=0.03,
            message_family="tone",
        ),
    ]
    audit = shortcut_audit(records)
    snr = audit.by_class["am"]["snr_db"]
    assert snr.counts == ((5.0, 2),)
    amplitude = audit.by_class["am"]["amplitude"]
    assert amplitude.n == 2
    assert amplitude.minimum == pytest.approx(1.0)
    assert amplitude.maximum == pytest.approx(3.0)
    assert amplitude.mean == pytest.approx(2.0)
    assert amplitude.counts == ()
    assert audit.by_class["am"]["phase_rad"].mean == pytest.approx(1.0)
    assert audit.by_class["am"]["cfo_norm"].mean == pytest.approx(0.01)


def test_unknown_label_rejected():
    with pytest.raises(ValueError):
        evaluate_predictions(["am"], ["qam"], labels=LABELS)


def test_length_mismatch_rejected():
    with pytest.raises(ValueError):
        evaluate_predictions(["am"], ["am", "fm"])


def test_empty_truth_is_rejected():
    with pytest.raises(ValueError, match="non-empty"):
        evaluate_predictions([], [], labels=("am", "fm"))
