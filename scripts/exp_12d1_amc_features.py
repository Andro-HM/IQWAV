"""Module 12D1: experimental AMC features and FM/PM identifiability.

Experiment only. Not a production classifier. Does not inspect the
sealed test split. Does not call Module 11 known-modulation estimators.
"""

from __future__ import annotations

import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _exp_12d1 import (  # noqa: E402
    FEATURE_NAMES,
    PRIMARY_LABELS,
    extract_features,
    fm_pm_identity_experiment,
    primary_label,
    summarize,
)
from iqwav.dsp import apply_frequency_offset, apply_phase_offset  # noqa: E402
from iqwav.modulation import (  # noqa: E402
    am_modulate,
    bpsk_waveform,
    fm_modulate,
    pm_modulate,
    qpsk_waveform,
)
from iqwav.amr import (  # noqa: E402
    DatasetSeeds,
    SyntheticDatasetConfig,
    generate_synthetic_dataset,
    split_dataset,
)

FS = 48_000.0
INVARIANCE_ATOL = 1e-10
INVARIANCE_RTOL = 1e-8
AMP_SCALE = 2.5
PHASE_RAD = 0.8
CFO_NORM = 0.005


def _fmt(value: float) -> str:
    if not math.isfinite(value):
        return "nan"
    if abs(value) >= 100:
        return f"{value:8.3f}"
    if abs(value) >= 1:
        return f"{value:8.4f}"
    return f"{value:8.5f}"


def _print_summary_table(title: str, rows: dict[str, dict[str, dict[str, float]]]) -> None:
    print()
    print(title)
    header = (
        f"{'class':<8} {'feature':<24} {'n':>5} {'mean':>8} {'std':>8} "
        f"{'p05':>8} {'p25':>8} {'med':>8} {'p75':>8} {'p95':>8}"
    )
    print(header)
    print("-" * len(header))
    for label, features in rows.items():
        for name in FEATURE_NAMES:
            stats = features[name]
            print(
                f"{label:<8} {name:<24} {int(stats['count']):5d} "
                f"{_fmt(stats['mean'])} {_fmt(stats['std'])} "
                f"{_fmt(stats['p05'])} {_fmt(stats['p25'])} "
                f"{_fmt(stats['median'])} {_fmt(stats['p75'])} "
                f"{_fmt(stats['p95'])}"
            )


def _collect(records, label_fn=lambda record: record.label):
    buckets = defaultdict(lambda: defaultdict(list))
    skipped = 0
    for record in records:
        try:
            feats = extract_features(record.samples)
        except Exception:
            skipped += 1
            continue
        key = label_fn(record)
        for name, value in feats.items():
            buckets[key][name].append(value)
    tables = {}
    for key in buckets:
        tables[key] = {
            name: summarize(np.array(values, dtype=np.float64))
            for name, values in buckets[key].items()
        }
    return tables, skipped


def _overlap(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    if a.size == 0 or b.size == 0:
        return {"delta_median": math.nan, "p05_gap": math.nan}
    return {
        "delta_median": float(np.median(a) - np.median(b)),
        "p05_gap": float(np.percentile(a, 5) - np.percentile(b, 95)),
    }


def run_identity() -> None:
    result = fm_pm_identity_experiment()
    print("=== A. FM/PM exact-identifiability ===")
    print(f"N={result.n}  fs={result.fs}  f_norm={result.f_norm}  beta={result.beta}")
    print(f"difference amplitude A={result.difference_amplitude:.12f}")
    print(f"delta_f_norm={result.delta_f_norm:.12f}  (accepted FM range 0.02..0.08)")
    print(
        "analytic: phi_FM[n] = beta*(m_p[n]-m_p[0]) = phi_PM[n] + "
        f"{result.constant_phase_rad:.12f}"
    )
    print(f"increment identity max|err| = {result.increment_max_abs_err:.3e}")
    print("before constant-phase compensation:")
    print(f"  max |s_fm - s_pm| = {result.max_abs_before:.6e}")
    print(f"  RMS |s_fm - s_pm| = {result.rms_before:.6e}")
    print("after multiplying s_fm by exp(-j*phi_FM_minus_PM):")
    print(f"  max |s_fm_aligned - s_pm| = {result.max_abs_after:.6e}")
    print(f"  RMS |s_fm_aligned - s_pm| = {result.rms_after:.6e}")
    print("identical amplitude/CFO and the SAME AWGN after alignment:")
    print(f"  max |err| = {result.max_abs_impaired:.6e}")
    print(f"  RMS |err| = {result.rms_impaired:.6e}")
    print(
        "Conclusion: under this discrete-difference construction the FM "
        "and PM waveforms differ only by a constant phase. fm and pm "
        "cannot be universal separate production AMC outputs on the "
        "unrestricted message domain."
    )


def _clean_records() -> dict[str, np.ndarray]:
    n = 256
    t = np.arange(n, dtype=np.float64)
    message = np.cos(2.0 * np.pi * 0.04 * t)
    bits = np.array([0, 1, 1, 0, 1, 0, 0, 1] * 16, dtype=np.int64)
    qpsk_bits = bits[:64]
    return {
        "am": am_modulate(message, 0.6),
        "fm": fm_modulate(message, FS, 0.05 * FS),
        "pm": pm_modulate(message, 0.8),
        "bpsk": bpsk_waveform(bits, 8)[:n],
        "qpsk": qpsk_waveform(qpsk_bits, 8)[:n],
    }


def run_invariance() -> dict[str, dict[str, str]]:
    print()
    print("=== C. Controlled invariance (clean paired copies) ===")
    print(
        f"amplitude x{AMP_SCALE}, phase {PHASE_RAD} rad, "
        f"CFO {CFO_NORM}*fs ; rtol={INVARIANCE_RTOL} atol={INVARIANCE_ATOL}"
    )
    verdicts = {name: {"amp": "INVARIANT", "phase": "INVARIANT", "cfo": "INVARIANT"} for name in FEATURE_NAMES}
    for label, clean in _clean_records().items():
        base = extract_features(clean)
        copies = {
            "amp": extract_features(AMP_SCALE * clean),
            "phase": extract_features(apply_phase_offset(clean, PHASE_RAD)),
            "cfo": extract_features(apply_frequency_offset(clean, FS, CFO_NORM * FS)),
        }
        print(f"class {label}")
        for name in FEATURE_NAMES:
            for kind, feats in copies.items():
                ok = math.isclose(
                    base[name],
                    feats[name],
                    rel_tol=INVARIANCE_RTOL,
                    abs_tol=INVARIANCE_ATOL,
                )
                if not ok:
                    verdicts[name][kind] = "NOT INVARIANT"
                status = "ok" if ok else "FAIL"
                print(
                    f"  {name:<24} {kind:<5} base={base[name]:.6e} "
                    f"copy={feats[name]:.6e} {status}"
                )
    print("per-feature invariance summary:")
    for name in FEATURE_NAMES:
        print(f"  {name:<24} {verdicts[name]}")
    return verdicts


def run_campaign():
    print()
    print("=== D. Feature-distribution campaign (train+validation only) ===")
    config = SyntheticDatasetConfig(
        fs=FS,
        n_samples_values=(256,),
        n_parents_per_class=30,
        n_variants_per_parent=5,
        seeds=DatasetSeeds(101, 202, 303, 404),
    )
    dataset = generate_synthetic_dataset(config)
    split = split_dataset(dataset)
    development = split.train + split.validation
    test_ids = {record.record_id for record in split.test}
    if any(record.record_id in test_ids for record in development):
        raise RuntimeError("test split leaked into development.")
    print(
        f"generated {len(dataset.records)} records; "
        f"train={len(split.train)} val={len(split.validation)} "
        f"test={len(split.test)} (sealed, unused)"
    )
    five_class, skipped = _collect(development, lambda record: record.label)
    four_class, _ = _collect(development, lambda record: primary_label(record.label))
    print(f"skipped zero-power/failed records: {skipped}")
    ordered_five = {label: five_class[label] for label in ("am", "fm", "pm", "bpsk", "qpsk")}
    ordered_four = {label: four_class[label] for label in PRIMARY_LABELS}
    _print_summary_table("Five-class diagnostic (fm and pm kept separate)", ordered_five)
    _print_summary_table("Primary four-class (fm+pm -> angle)", ordered_four)

    print()
    print("--- Nuisance breakdown (primary labels) ---")
    by_snr = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    by_sps = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    by_family = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for record in development:
        feats = extract_features(record.samples)
        lab = primary_label(record.label)
        snr = record.nuisances.snr_db
        for name, value in feats.items():
            by_snr[snr][lab][name].append(value)
        if record.nuisances.samples_per_symbol is not None:
            for name, value in feats.items():
                by_sps[record.nuisances.samples_per_symbol][lab][name].append(value)
        if record.nuisances.message_family is not None:
            for name, value in feats.items():
                by_family[record.nuisances.message_family][lab][name].append(value)

    def _print_breakdown(title, store, keys_of_interest):
        print()
        print(title)
        for bucket in sorted(store):
            print(f"  bucket={bucket}")
            for lab in PRIMARY_LABELS:
                if lab not in store[bucket]:
                    continue
                for name in keys_of_interest:
                    stats = summarize(np.array(store[bucket][lab][name], dtype=np.float64))
                    print(
                        f"    {lab:<6} {name:<24} n={int(stats['count']):3d} "
                        f"med={_fmt(stats['median'])} p05={_fmt(stats['p05'])} "
                        f"p95={_fmt(stats['p95'])}"
                    )

    _print_breakdown(
        "By SNR (envelope, C4)",
        by_snr,
        ("envelope_dispersion", "envelope_coherence", "c4", "c2_over_c4"),
    )
    _print_breakdown(
        "By digital SPS (sparsity, periodicity, C2/C4)",
        by_sps,
        ("phase_sparsity", "transition_periodicity", "c2", "c4", "c2_over_c4"),
    )
    _print_breakdown(
        "By analog message family (envelope, sparsity)",
        by_family,
        ("envelope_dispersion", "envelope_coherence", "phase_sparsity"),
    )
    return development, five_class


def run_ablation(development) -> None:
    print()
    print("=== E. Feature ablation / overlap diagnostics (no classifier) ===")
    grouped = defaultdict(lambda: defaultdict(list))
    grouped_snr = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    grouped_sps = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    grouped_family = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for record in development:
        feats = extract_features(record.samples)
        lab = primary_label(record.label)
        raw = record.label
        for name, value in feats.items():
            grouped[lab][name].append(value)
            grouped[raw][name].append(value)
            grouped_snr[record.nuisances.snr_db][lab][name].append(value)
            if record.nuisances.samples_per_symbol is not None:
                grouped_sps[record.nuisances.samples_per_symbol][lab][name].append(value)
            if record.nuisances.message_family is not None:
                grouped_family[record.nuisances.message_family][lab][name].append(value)

    def arr(label, name):
        return np.array(grouped[label][name], dtype=np.float64)

    print("AM vs everything else (envelope):")
    rest_ev = np.concatenate(
        [arr(label, "envelope_dispersion") for label in ("angle", "bpsk", "qpsk")]
    )
    rest_coh = np.concatenate(
        [arr(label, "envelope_coherence") for label in ("angle", "bpsk", "qpsk")]
    )
    print("  Ev     AM vs rest", _overlap(arr("am", "envelope_dispersion"), rest_ev))
    print("  coh    AM vs rest", _overlap(arr("am", "envelope_coherence"), rest_coh))
    print(
        "  Ev-only p05(AM)-p95(rest) "
        f"{_overlap(arr('am', 'envelope_dispersion'), rest_ev)['p05_gap']:.5f} "
        "(positive => 5/95 envelope-dispersion gap with no coherence)"
    )

    psk_sp = np.concatenate([arr("bpsk", "phase_sparsity"), arr("qpsk", "phase_sparsity")])
    psk_per = np.concatenate(
        [arr("bpsk", "transition_periodicity"), arr("qpsk", "transition_periodicity")]
    )
    print("ANGLE vs BPSK/QPSK:")
    print("  sparsity", _overlap(arr("angle", "phase_sparsity"), psk_sp))
    print("  period. ", _overlap(arr("angle", "transition_periodicity"), psk_per))

    print("BPSK vs QPSK (C2/C4):")
    print("  C2     ", _overlap(arr("bpsk", "c2"), arr("qpsk", "c2")))
    print("  C4     ", _overlap(arr("bpsk", "c4"), arr("qpsk", "c4")))
    print("  C2/C4  ", _overlap(arr("bpsk", "c2_over_c4"), arr("qpsk", "c2_over_c4")))

    print("FM vs PM (diagnostic only, no production threshold):")
    print("  Ev     ", _overlap(arr("fm", "envelope_dispersion"), arr("pm", "envelope_dispersion")))
    print("  spars. ", _overlap(arr("fm", "phase_sparsity"), arr("pm", "phase_sparsity")))
    print("  C2     ", _overlap(arr("fm", "c2"), arr("pm", "c2")))

    print("C4 vs SNR (BPSK and QPSK medians):")
    for snr in sorted(grouped_snr):
        line = f"  SNR={snr:>4}"
        for lab in ("bpsk", "qpsk", "angle", "am"):
            values = grouped_snr[snr][lab]["c4"]
            if values:
                line += f"  {lab} med={np.median(values):.4f}"
        print(line)

    print("SPS keying (BPSK/QPSK median sparsity and periodicity):")
    for sps in sorted(grouped_sps):
        line = f"  SPS={sps}"
        for lab in ("bpsk", "qpsk"):
            if lab not in grouped_sps[sps]:
                continue
            sp = np.median(grouped_sps[sps][lab]["phase_sparsity"])
            pe = np.median(grouped_sps[sps][lab]["transition_periodicity"])
            line += f"  {lab} spars={sp:.4f} per={pe:.4f}"
        print(line)

    print("Message-family keying (AM/ANGLE median Ev):")
    for family in sorted(grouped_family):
        line = f"  family={family}"
        for lab in ("am", "angle"):
            if lab not in grouped_family[family]:
                continue
            line += f"  {lab} Ev med={np.median(grouped_family[family][lab]['envelope_dispersion']):.5f}"
        print(line)


def main() -> None:
    run_identity()
    run_invariance()
    development, _five = run_campaign()
    run_ablation(development)


if __name__ == "__main__":
    main()
