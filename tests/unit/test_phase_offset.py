"""Unit tests for blind static phase estimation in iqwav.estimation.phase_offset."""

import numpy as np
import pytest

from iqwav.dsp import add_awgn, apply_frequency_offset, apply_phase_offset
from iqwav.estimation import estimate_frequency_offset, estimate_phase_offset
from iqwav.modulation import bpsk_modulate, bpsk_waveform, qpsk_modulate, qpsk_waveform
from iqwav.synchronization import correct_frequency_offset, correct_phase_offset

FS = 80_000.0


def _wrap_error(error, period):
    """Wrap a phase error into (-period/2, period/2]."""
    return (error + period / 2.0) % period - period / 2.0


def _bpsk_clean(seed=101, nbits=4096, sps=8):
    bits = np.random.default_rng(seed).integers(0, 2, nbits)
    return bpsk_waveform(bits, sps)


def _qpsk_clean(seed=102, nbits=8192, sps=8):
    bits = np.random.default_rng(seed).integers(0, 2, nbits)
    return qpsk_waveform(bits, sps)


def _modulation_params(modulation):
    if modulation == "bpsk":
        return 2, np.pi
    return 4, np.pi / 2.0


def test_bpsk_canonical_symbols_square_to_plus_one():
    symbols = bpsk_modulate(np.array([0, 1], dtype=np.int64))
    np.testing.assert_allclose(symbols, [1.0 + 0.0j, -1.0 + 0.0j])
    np.testing.assert_allclose(symbols**2, 1.0, atol=1e-12)


def test_qpsk_canonical_symbols_fourth_power_is_minus_one():
    """Every ideal IQWAV Gray-mapped QPSK symbol satisfies s**4 == -1;
    this is the constellation reference the estimator compensates."""
    for pair in ((0, 0), (0, 1), (1, 1), (1, 0)):
        symbol = qpsk_modulate(np.array(pair, dtype=np.int64))[0]
        np.testing.assert_allclose(np.abs(symbol), 1.0, atol=1e-12)
        np.testing.assert_allclose(symbol**4, -1.0, atol=1e-12)


@pytest.mark.parametrize("phi", [0.0, 0.4, -0.7, 1.3, 2.5, 4.0, -4.4])
def test_clean_bpsk_estimates_injected_phase_modulo_pi(phi):
    waveform = _bpsk_clean()
    est = estimate_phase_offset(apply_phase_offset(waveform, phi), "bpsk")
    assert -np.pi / 2 <= est.phase_offset_rad < np.pi / 2
    assert abs(_wrap_error(est.phase_offset_rad - phi, np.pi)) < 1e-7
    assert est.symmetry > 0.9999


@pytest.mark.parametrize(
    ("phi", "expected"),
    [
        (np.pi / 2 - 1e-9, np.pi / 2 - 1e-9),
        (np.pi / 2, -np.pi / 2),
        (np.pi / 2 + 1e-9, -np.pi / 2 + 1e-9),
        (-np.pi / 2 + 1e-9, -np.pi / 2 + 1e-9),
    ],
)
def test_clean_bpsk_canonical_boundary_behavior(phi, expected):
    """Deterministic canonical wrapping: the returned range is
    [-pi/2, +pi/2); an injected +pi/2 maps to -pi/2 rather than depending
    accidentally on np.angle's branch choice at +/-pi."""
    waveform = _bpsk_clean()
    est = estimate_phase_offset(apply_phase_offset(waveform, phi), "bpsk")
    np.testing.assert_allclose(est.phase_offset_rad, expected, atol=1e-9)
    assert abs(_wrap_error(est.phase_offset_rad - phi, np.pi)) < 1e-7


def test_clean_qpsk_zero_phase_estimates_zero_not_quarter_pi():
    """CRITICAL regression: IQWAV QPSK symbols satisfy s**4 = -1, so the
    fourth-power constellation reference MUST be compensated. A naive
    angle(mean(x**4))/4 estimator reports pi/4 for a zero-phase IQWAV
    QPSK signal (measured naive value: 0.785398163 rad); the compensated
    estimator must report approximately 0."""
    est = estimate_phase_offset(_qpsk_clean(), "qpsk")
    assert abs(est.phase_offset_rad) < 1e-7
    assert (
        min(
                abs(est.phase_offset_rad - np.pi / 4),
                abs(est.phase_offset_rad + np.pi / 4),
        )
        > 0.1
    )
    assert est.symmetry > 0.9999


@pytest.mark.parametrize("phi", [0.0, 0.3, -0.6, 1.2, -2.0, 3.5])
def test_clean_qpsk_estimates_injected_phase_modulo_half_pi(phi):
    waveform = _qpsk_clean()
    est = estimate_phase_offset(apply_phase_offset(waveform, phi), "qpsk")
    assert -np.pi / 4 <= est.phase_offset_rad < np.pi / 4
    assert abs(_wrap_error(est.phase_offset_rad - phi, np.pi / 2)) < 1e-7
    assert est.symmetry > 0.9999


@pytest.mark.parametrize(
    ("phi", "expected"),
    [
        (np.pi / 4 - 1e-9, np.pi / 4 - 1e-9),
        (np.pi / 4, -np.pi / 4),
        (np.pi / 4 + 1e-9, -np.pi / 4 + 1e-9),
        (-np.pi / 4 + 1e-9, -np.pi / 4 + 1e-9),
    ],
)
def test_clean_qpsk_canonical_boundary_behavior(phi, expected):
    """Deterministic canonical wrapping: the returned range is
    [-pi/4, +pi/4); an injected +pi/4 maps to -pi/4."""
    waveform = _qpsk_clean()
    est = estimate_phase_offset(apply_phase_offset(waveform, phi), "qpsk")
    np.testing.assert_allclose(est.phase_offset_rad, expected, atol=1e-9)
    assert abs(_wrap_error(est.phase_offset_rad - phi, np.pi / 2)) < 1e-7


@pytest.mark.parametrize("modulation", ["bpsk", "qpsk"])
def test_estimate_then_correct_independent_residual_truth(modulation):
    """Blind PSK correction can leave a rotational ambiguity (-clean for
    BPSK, j**k * clean for QPSK), so absolute transmitted bit labeling is
    NOT demanded. The PRIMARY post-correction truth is the independent
    reference-ratio measurement: residual = angle(mean(corrected /
    clean)), wrapped modulo the PSK rotational symmetry. It does NOT call
    estimate_phase_offset() after correction."""
    order, period = _modulation_params(modulation)
    waveform = _bpsk_clean() if modulation == "bpsk" else _qpsk_clean()
    received = apply_phase_offset(waveform, 0.8)
    est = estimate_phase_offset(received, modulation)
    corrected = correct_phase_offset(received, est.phase_offset_rad)

    residual = float(np.angle(np.mean(corrected / waveform)))
    assert abs(_wrap_error(residual, period)) < 1e-7

    ambiguity = np.exp(2j * np.pi * np.arange(order) / order)
    assert any(
        np.allclose(corrected, waveform * rotation, atol=1e-9)
        for rotation in ambiguity
    )


def test_rotational_ambiguity_prevents_absolute_bit_labeling():
    """Rotating clean BPSK/QPSK by an allowed blind residual (pi or pi/2)
    maps each constellation onto itself but changes the demodulated bits;
    absolute bit labeling cannot be recovered from PSK rotational
    symmetry alone (pilots, differential coding, framing or known
    headers are needed)."""
    from iqwav.demod import bpsk_demodulate, qpsk_demodulate

    bits = np.array([0, 0, 0, 1, 1, 1, 1, 0], dtype=np.int64)
    qpsk = qpsk_waveform(bits, 8)
    assert not np.array_equal(qpsk_demodulate(1j * qpsk, 8), bits)

    bpsk_bits = np.array([0, 1, 1, 0, 1, 0], dtype=np.int64)
    bpsk = bpsk_waveform(bpsk_bits, 8)
    np.testing.assert_array_equal(
        bpsk_demodulate(-bpsk, 8), 1 - bpsk_bits
    )


@pytest.mark.parametrize("modulation", ["bpsk", "qpsk"])
def test_reestimate_after_correction_is_self_consistency_only(modulation):
    """Optional consistency check only, never proof of correctness:
    re-running the same estimator after correcting by its own estimate
    reports approximately zero. Like the Module 11A CFO lesson,
    same-estimator chains can hide bias; the proof is the independent
    reference-ratio residual in the test above."""
    waveform = _bpsk_clean() if modulation == "bpsk" else _qpsk_clean()
    received = apply_phase_offset(waveform, 0.8)
    est = estimate_phase_offset(received, modulation)
    corrected = correct_phase_offset(received, est.phase_offset_rad)
    re_est = estimate_phase_offset(corrected, modulation)
    assert abs(re_est.phase_offset_rad) < 1e-6
    assert re_est.symmetry > 0.99


@pytest.mark.parametrize("modulation", ["bpsk", "qpsk"])
def test_phase_estimation_after_true_cfo_correction(modulation):
    """CFO isolation: the frequency offset is removed with the TRUE value
    (no CFO-estimator bias can contaminate the phase estimate), then the
    static phase is estimated."""
    _, period = _modulation_params(modulation)
    waveform = _bpsk_clean() if modulation == "bpsk" else _qpsk_clean()
    received = apply_frequency_offset(apply_phase_offset(waveform, 0.8), FS, 1000.0)
    corrected = correct_frequency_offset(received, FS, 1000.0)
    est = estimate_phase_offset(corrected, modulation)
    assert abs(_wrap_error(est.phase_offset_rad - 0.8, period)) < 1e-6
    assert est.symmetry > 0.99


def test_bpsk_after_coarse_cfo_correction_estimates_phase():
    """Scientifically stable integration: the lag-1 coarse CFO estimator
    is essentially exact on this deterministic BPSK block (Module 11A),
    so a coarse-CFO-estimate-then-correct chain still recovers the
    static phase."""
    waveform = _bpsk_clean()
    received = apply_frequency_offset(apply_phase_offset(waveform, 0.8), FS, 1000.0)
    cfo = estimate_frequency_offset(received, FS)
    corrected = correct_frequency_offset(received, FS, cfo.frequency_offset_hz)
    est = estimate_phase_offset(corrected, "bpsk")
    assert abs(_wrap_error(est.phase_offset_rad - 0.8, np.pi)) < 0.05
    assert est.symmetry > 0.99


def test_qpsk_after_coarse_cfo_correction_is_degraded_or_rejected():
    """Scientific finding: the lag-1 coarse CFO estimator has a
    finite-record QPSK bias (Module 11A: +4.433 Hz on this deterministic
    block). Over the 0.4096 s observation that residual rotates the
    constellation through many cycles, collapsing the fourth-power
    concentration, so the estimator refuses rather than presenting an
    arbitrary phase as reliable truth. Module 11B correctness does NOT
    depend on the CFO estimator being exact; no PLL / joint
    phase-frequency tracking is attempted in this milestone."""
    waveform = _qpsk_clean()
    received = apply_frequency_offset(apply_phase_offset(waveform, 0.8), FS, 1000.0)
    cfo = estimate_frequency_offset(received, FS)
    corrected = correct_frequency_offset(received, FS, cfo.frequency_offset_hz)
    with pytest.raises(ValueError):
        estimate_phase_offset(corrected, "qpsk")
    diagnostic = estimate_phase_offset(corrected, "qpsk", min_symmetry=0.0)
    assert diagnostic.symmetry < 0.3


@pytest.mark.parametrize("modulation", ["bpsk", "qpsk"])
def test_residual_cfo_degrades_block_phase_estimation(modulation):
    """Residual-CFO characterization: correcting with CFO knowledge stale
    by 5 Hz leaves a residual that rotates the constellation by ~2*pi
    across the 0.2048 s block; the constant-block phase estimator's
    symmetry collapses and the default reliability threshold rejects
    the input. This degradation is documented, not solved with a PLL."""
    waveform = (
        bpsk_waveform(np.random.default_rng(201).integers(0, 2, 2048), 8)
        if modulation == "bpsk"
        else qpsk_waveform(np.random.default_rng(202).integers(0, 2, 4096), 8)
    )
    received = apply_frequency_offset(waveform, FS, 1000.0)
    corrected = correct_frequency_offset(received, FS, 995.0)

    n = np.arange(corrected.size, dtype=np.float64)
    slope = np.polyfit(n, np.unwrap(np.angle(corrected / waveform)), 1)[0]
    np.testing.assert_allclose(slope * FS / (2.0 * np.pi), 5.0, atol=0.1)

    with pytest.raises(ValueError):
        estimate_phase_offset(corrected, modulation)

    degraded = estimate_phase_offset(corrected, modulation, min_symmetry=0.0)
    static = estimate_phase_offset(apply_phase_offset(waveform, 0.3), modulation)
    assert degraded.symmetry < 0.1
    assert degraded.symmetry < 0.5 * static.symmetry


AWGN_CASES = [
    # (modulation, snr_db, max_abs_error_rad, min_symmetry)
    ("bpsk", 20.0, 0.01, 0.95),
    ("bpsk", 10.0, 0.01, 0.85),
    ("bpsk", 0.0, 0.05, 0.40),
    ("qpsk", 20.0, 0.01, 0.90),
    ("qpsk", 10.0, 0.01, 0.60),
    ("qpsk", 0.0, 0.10, 0.10),
]


@pytest.mark.parametrize(("modulation", "snr_db", "max_err", "min_sym"), AWGN_CASES)
def test_awgn_phase_estimation(modulation, snr_db, max_err, min_sym):
    """Deterministic-seed AWGN characterization with true phase 0.8.

    Measured worst wrapped phase errors over noise seeds 1-3
    (Module 11B verification):

    ========= ==== ================= ==============
    modulation SNR  worst |error|    symmetry range
    ========= ==== ================= ==============
    bpsk       20dB 2.7e-4 rad       0.990
    bpsk       10dB 1.1e-3 rad       0.909
    bpsk        0dB 5.6e-3 rad       0.498 - 0.504
    qpsk       20dB 6.0e-4 rad       0.961
    qpsk       10dB 1.1e-3 rad       0.700 - 0.707
    qpsk        0dB 1.5e-2 rad       0.134 - 0.150
    ========= ==== ================= ==============

    Also measured but deliberately NOT asserted (no universal low-SNR
    robustness is claimed): at -3 dB, bpsk stays accurate (worst 1.1e-2
    rad, symmetry ~0.33) while qpsk symmetry drops to 0.054 - 0.068,
    near the default 0.05 reliability threshold.
    """
    _, period = _modulation_params(modulation)
    waveform = _bpsk_clean() if modulation == "bpsk" else _qpsk_clean()
    rotated = apply_phase_offset(waveform, 0.8)
    for seed in (1, 2, 3):
        noisy = add_awgn(rotated, snr_db, np.random.default_rng(seed))
        est = estimate_phase_offset(noisy, modulation)
        assert abs(_wrap_error(est.phase_offset_rad - 0.8, period)) < max_err
        assert est.symmetry >= min_sym
        assert 0.0 <= est.symmetry <= 1.0


@pytest.mark.parametrize("modulation", ["bpsk", "qpsk"])
def test_awgn_symmetry_decreases_with_snr(modulation):
    waveform = _bpsk_clean() if modulation == "bpsk" else _qpsk_clean()
    rotated = apply_phase_offset(waveform, 0.8)
    symmetries = []
    for snr_db in (20.0, 10.0, 0.0):
        noisy = add_awgn(rotated, snr_db, np.random.default_rng(1))
        symmetries.append(estimate_phase_offset(noisy, modulation).symmetry)
    assert symmetries[0] > symmetries[1] > symmetries[2]


@pytest.mark.parametrize("modulation", ["bpsk", "qpsk"])
def test_pure_noise_rejected_by_default_reliability_threshold(modulation):
    """Long pure-noise records have M-th-power symmetry of order
    1/sqrt(N), far below the default reliability threshold, so the
    estimator refuses to present an arbitrary phase as truth."""
    noise = (
        np.random.default_rng(7).normal(0.0, 1.0, 65_536)
        + 1j * np.random.default_rng(8).normal(0.0, 1.0, 65_536)
    )
    with pytest.raises(ValueError):
        estimate_phase_offset(noise, modulation)


@pytest.mark.parametrize("modulation", ["bpsk", "qpsk"])
def test_pure_noise_diagnostic_mode_exposes_low_symmetry(modulation):
    """With the threshold lowered to 0 (diagnostic use only), a short
    noise-only record returns an estimate whose explicitly low symmetry
    flags it as unreliable. The symmetry is a reliability measure, NOT
    calibrated confidence, and the returned phase is meaningless."""
    noise = (
        np.random.default_rng(9).normal(0.0, 1.0, 4096)
        + 1j * np.random.default_rng(10).normal(0.0, 1.0, 4096)
    )
    est = estimate_phase_offset(noise, modulation, min_symmetry=0.0)
    assert 0.0 <= est.symmetry <= 1.0
    assert est.symmetry < 0.1


@pytest.mark.parametrize(
    "bad_samples",
    [
        np.array([], dtype=np.complex128),
        np.ones((4, 4), dtype=np.complex128),
        np.ones(16),
        np.ones(16, dtype=np.bool_),
        np.array([1.0 + 2.0j, np.nan + 1.0j]),
        np.array([1.0 + np.inf * 1j, 2.0 + 1.0j]),
        np.zeros(16, dtype=np.complex128),
    ],
)
def test_invalid_samples_raise(bad_samples):
    with pytest.raises(ValueError):
        estimate_phase_offset(bad_samples, "bpsk")


@pytest.mark.parametrize(
    "modulation",
    ["8psk", "QPSK", "bpsk ", "", "pi/2-bpsk", None, 5],
)
def test_unsupported_modulation_raises(modulation):
    samples = np.ones(16, dtype=np.complex128)
    with pytest.raises(ValueError):
        estimate_phase_offset(samples, modulation)


@pytest.mark.parametrize(
    "min_symmetry",
    [True, False, -0.1, 1.5, float("nan"), float("inf"), "0.5", None],
)
def test_invalid_min_symmetry_raises(min_symmetry):
    samples = np.ones(16, dtype=np.complex128)
    with pytest.raises(ValueError):
        estimate_phase_offset(samples, "bpsk", min_symmetry=min_symmetry)


def test_min_symmetry_zero_is_diagnostic_and_accepted():
    est = estimate_phase_offset(
        apply_phase_offset(_bpsk_clean(), 0.5), "bpsk", min_symmetry=0.0
    )
    assert abs(_wrap_error(est.phase_offset_rad - 0.5, np.pi)) < 1e-7


@pytest.mark.parametrize(
    ("modulation", "seed", "nbits", "phi"),
    [("bpsk", 12, 4096, -1.1), ("qpsk", 11, 8192, 0.6)],
)
def test_operates_on_symbol_rate_samples_without_oversampling(
    modulation, seed, nbits, phi
):
    """No samples_per_symbol or symbol timing is required: the estimator
    works directly on already-extracted symbol-rate samples too."""
    _, period = _modulation_params(modulation)
    bits = np.random.default_rng(seed).integers(0, 2, nbits)
    symbols = bpsk_modulate(bits) if modulation == "bpsk" else qpsk_modulate(bits)
    est = estimate_phase_offset(apply_phase_offset(symbols, phi), modulation)
    assert abs(_wrap_error(est.phase_offset_rad - phi, period)) < 1e-7
    assert est.symmetry > 0.9999


def test_single_sample_is_valid_input():
    est = estimate_phase_offset(np.array([np.exp(1j * 0.5)]), "bpsk")
    np.testing.assert_allclose(est.phase_offset_rad, 0.5, atol=1e-12)
    np.testing.assert_allclose(est.symmetry, 1.0, atol=1e-12)
