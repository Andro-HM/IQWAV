"""Unit tests for whole-block residual CFO refinement."""

import numpy as np
import pytest

from iqwav.dsp import add_awgn, apply_frequency_offset, apply_phase_offset
from iqwav.estimation import (
    ResidualFrequencyOffsetEstimate,
    estimate_frequency_offset,
    estimate_phase_offset,
    estimate_residual_frequency_offset,
)
from iqwav.modulation import bpsk_modulate, bpsk_waveform, qpsk_modulate, qpsk_waveform
from iqwav.synchronization import correct_frequency_offset

FS = 80_000.0
PHI = 0.8
CFO_HZ = 1000.0


def _bpsk_clean(seed=101, nbits=4096, sps=8):
    bits = np.random.default_rng(seed).integers(0, 2, nbits)
    return bpsk_waveform(bits, sps)


def _qpsk_clean(seed=102, nbits=8192, sps=8):
    bits = np.random.default_rng(seed).integers(0, 2, nbits)
    return qpsk_waveform(bits, sps)


def _order(modulation):
    return 2 if modulation == "bpsk" else 4


def _wrap_phase_error(error, period):
    return (error + period / 2.0) % period - period / 2.0


def _independent_residual_hz(corrected, clean, fs):
    nonzero = np.abs(clean) > 0.0
    n = np.nonzero(nonzero)[0].astype(np.float64)
    phase = np.unwrap(np.angle(corrected[nonzero] / clean[nonzero]))
    slope = np.polyfit(n, phase, 1)[0]
    return float(slope * fs / (2.0 * np.pi))


def _physical_residual_hz(true_cfo, *corrections):
    leftover = true_cfo
    for value in corrections:
        leftover -= value
    return float(leftover)


def test_estimate_field_consistency():
    samples = apply_frequency_offset(_bpsk_clean(), FS, 5.0)
    estimate = estimate_residual_frequency_offset(samples, FS, "bpsk")
    assert isinstance(estimate, ResidualFrequencyOffsetEstimate)
    np.testing.assert_allclose(
        estimate.residual_frequency_hz,
        FS * estimate.phase_increment_rad / (2.0 * np.pi),
    )
    assert np.isfinite(estimate.residual_frequency_hz)
    assert np.isfinite(estimate.phase_increment_rad)
    assert not hasattr(estimate, "coherence")


def test_result_is_frozen_dataclass():
    estimate = estimate_residual_frequency_offset(_bpsk_clean(), FS, "bpsk")
    with pytest.raises(Exception):
        estimate.residual_frequency_hz = 0.0


@pytest.mark.parametrize("modulation", ["bpsk", "qpsk"])
@pytest.mark.parametrize("offset_hz", [0.0, 1.0, -1.0, 4.433, -5.0, 50.0, -50.0, 200.0])
def test_clean_rectangular_waveform_recovers_injected_residual(modulation, offset_hz):
    waveform = _bpsk_clean() if modulation == "bpsk" else _qpsk_clean()
    received = apply_frequency_offset(waveform, FS, offset_hz)
    estimate = estimate_residual_frequency_offset(received, FS, modulation)
    np.testing.assert_allclose(estimate.residual_frequency_hz, offset_hz, atol=1e-4)
    half = FS / (2.0 * _order(modulation))
    assert -half <= estimate.residual_frequency_hz < half


@pytest.mark.parametrize("modulation", ["bpsk", "qpsk"])
def test_reported_hz_is_original_residual_not_m_times_residual(modulation):
    waveform = _bpsk_clean() if modulation == "bpsk" else _qpsk_clean()
    estimate = estimate_residual_frequency_offset(
        apply_frequency_offset(waveform, FS, 5.0), FS, modulation
    )
    np.testing.assert_allclose(estimate.residual_frequency_hz, 5.0, atol=1e-4)
    assert abs(estimate.residual_frequency_hz - 5.0 * _order(modulation)) > 1.0


def test_zero_cfo_qpsk_reports_zero_not_orientation_bias():
    estimate = estimate_residual_frequency_offset(_qpsk_clean(), FS, "qpsk")
    np.testing.assert_allclose(estimate.residual_frequency_hz, 0.0, atol=1e-4)


def test_complex64_qpsk_recovers_injected_residual():
    """Representative dtype regression: complex64 input must still recover
    a known leftover under the accepted whole-block estimator."""
    waveform = apply_frequency_offset(_qpsk_clean(), FS, 5.0).astype(np.complex64)
    estimate = estimate_residual_frequency_offset(waveform, FS, "qpsk")
    np.testing.assert_allclose(estimate.residual_frequency_hz, 5.0, atol=1e-3)
    assert np.isfinite(estimate.residual_frequency_hz)
    assert np.isfinite(estimate.phase_increment_rad)


def test_sign_convention():
    waveform = _bpsk_clean()
    positive = estimate_residual_frequency_offset(
        apply_frequency_offset(waveform, FS, 17.0), FS, "bpsk"
    )
    negative = estimate_residual_frequency_offset(
        apply_frequency_offset(waveform, FS, -17.0), FS, "bpsk"
    )
    assert positive.residual_frequency_hz > 0.0
    assert negative.residual_frequency_hz < 0.0
    np.testing.assert_allclose(
        positive.residual_frequency_hz,
        -negative.residual_frequency_hz,
        atol=1e-4,
    )


@pytest.mark.parametrize("modulation", ["bpsk", "qpsk"])
def test_static_phase_invariance(modulation):
    waveform = _bpsk_clean() if modulation == "bpsk" else _qpsk_clean()
    received = apply_frequency_offset(waveform, FS, 5.0)
    base = estimate_residual_frequency_offset(received, FS, modulation)
    rotated = estimate_residual_frequency_offset(
        apply_phase_offset(received, PHI), FS, modulation
    )
    np.testing.assert_allclose(
        rotated.residual_frequency_hz, base.residual_frequency_hz, atol=1e-4
    )


@pytest.mark.parametrize("modulation", ["bpsk", "qpsk"])
def test_amplitude_scaling_invariance(modulation):
    waveform = _bpsk_clean() if modulation == "bpsk" else _qpsk_clean()
    estimate = estimate_residual_frequency_offset(
        apply_frequency_offset(waveform, FS, 5.0) * 3.7, FS, modulation
    )
    np.testing.assert_allclose(estimate.residual_frequency_hz, 5.0, atol=1e-4)


@pytest.mark.parametrize(
    ("modulation", "seed", "nbits", "offset_hz"),
    [("bpsk", 12, 4096, -11.0), ("qpsk", 11, 8192, 9.0)],
)
def test_operates_on_ideal_symbol_rate_samples(modulation, seed, nbits, offset_hz):
    bits = np.random.default_rng(seed).integers(0, 2, nbits)
    symbols = bpsk_modulate(bits) if modulation == "bpsk" else qpsk_modulate(bits)
    estimate = estimate_residual_frequency_offset(
        apply_frequency_offset(symbols, FS, offset_hz), FS, modulation
    )
    np.testing.assert_allclose(estimate.residual_frequency_hz, offset_hz, atol=1e-4)


@pytest.mark.parametrize(
    ("modulation", "offset_hz"),
    [
        ("bpsk", FS / 4.0 - 50.0),
        ("bpsk", -FS / 4.0 + 50.0),
        ("bpsk", -FS / 4.0),
        ("qpsk", FS / 8.0 - 50.0),
        ("qpsk", -FS / 8.0 + 50.0),
        ("qpsk", -FS / 8.0),
    ],
)
def test_near_canonical_boundary_recovery(modulation, offset_hz):
    waveform = _bpsk_clean() if modulation == "bpsk" else _qpsk_clean()
    estimate = estimate_residual_frequency_offset(
        apply_frequency_offset(waveform, FS, offset_hz), FS, modulation
    )
    half = FS / (2.0 * _order(modulation))
    assert -half <= estimate.residual_frequency_hz < half
    np.testing.assert_allclose(estimate.residual_frequency_hz, offset_hz, atol=1e-3)


def test_alias_limitation_is_not_true_offset_recovery():
    """QPSK leftover outside +/-fs/8 aliases with period fs/4.
    +12000 Hz aliases to -8000 Hz; that is not true-offset recovery."""
    waveform = _qpsk_clean()
    estimate = estimate_residual_frequency_offset(
        apply_frequency_offset(waveform, FS, 12_000.0), FS, "qpsk"
    )
    np.testing.assert_allclose(estimate.residual_frequency_hz, -8000.0, atol=1e-3)
    assert abs(estimate.residual_frequency_hz - 12_000.0) > 1000.0
    half = FS / 8.0
    assert -half <= estimate.residual_frequency_hz < half


def test_qpsk_after_coarse_correction_matches_independent_leftover():
    waveform = _qpsk_clean()
    received = apply_frequency_offset(waveform, FS, CFO_HZ)
    coarse = estimate_frequency_offset(received, FS)
    after_coarse = correct_frequency_offset(
        received, FS, coarse.frequency_offset_hz
    )
    independent = _independent_residual_hz(after_coarse, waveform, FS)
    estimate = estimate_residual_frequency_offset(after_coarse, FS, "qpsk")
    np.testing.assert_allclose(
        estimate.residual_frequency_hz, independent, atol=1e-3
    )
    np.testing.assert_allclose(
        estimate.residual_frequency_hz,
        CFO_HZ - coarse.frequency_offset_hz,
        atol=1e-3,
    )


def test_qpsk_coarse_then_residual_independent_residual_near_zero():
    waveform = _qpsk_clean()
    received = apply_frequency_offset(waveform, FS, CFO_HZ)
    coarse = estimate_frequency_offset(received, FS)
    after_coarse = correct_frequency_offset(
        received, FS, coarse.frequency_offset_hz
    )
    residual = estimate_residual_frequency_offset(after_coarse, FS, "qpsk")
    corrected = correct_frequency_offset(
        after_coarse, FS, residual.residual_frequency_hz
    )
    measured = _independent_residual_hz(corrected, waveform, FS)
    phys = _physical_residual_hz(
        CFO_HZ, coarse.frequency_offset_hz, residual.residual_frequency_hz
    )
    assert abs(measured) < 1e-3
    assert abs(phys) < 1e-3


def test_qpsk_thousand_hz_11a_residual_then_phase_accepts():
    """Original motivating chain: QPSK +1000 Hz + phase 0.8.
    After coarse 11A alone, 11B rejects. After whole-block 11C, 11B
    accepts. Absolute bit labeling is not demanded."""
    waveform = _qpsk_clean()
    received = apply_frequency_offset(
        apply_phase_offset(waveform, PHI), FS, CFO_HZ
    )
    coarse = estimate_frequency_offset(received, FS)
    after_coarse = correct_frequency_offset(
        received, FS, coarse.frequency_offset_hz
    )
    with pytest.raises(ValueError):
        estimate_phase_offset(after_coarse, "qpsk")

    residual = estimate_residual_frequency_offset(after_coarse, FS, "qpsk")
    after_residual = correct_frequency_offset(
        after_coarse, FS, residual.residual_frequency_hz
    )
    phys = _physical_residual_hz(
        CFO_HZ, coarse.frequency_offset_hz, residual.residual_frequency_hz
    )
    assert abs(phys) < 0.005
    est = estimate_phase_offset(after_residual, "qpsk")
    assert abs(_wrap_phase_error(est.phase_offset_rad - PHI, np.pi / 2.0)) < 0.05
    assert est.symmetry > 0.9


@pytest.mark.parametrize("noise_seed", [1, 2, 3])
def test_qpsk_20db_chain_restores_11b(noise_seed):
    """Deterministic 20 dB regression of the failed lag-1 11C chain."""
    waveform = _qpsk_clean()
    received = apply_frequency_offset(
        apply_phase_offset(waveform, PHI), FS, CFO_HZ
    )
    noisy = add_awgn(received, 20.0, np.random.default_rng(noise_seed))
    coarse = estimate_frequency_offset(noisy, FS)
    after_coarse = correct_frequency_offset(
        noisy, FS, coarse.frequency_offset_hz
    )
    residual = estimate_residual_frequency_offset(after_coarse, FS, "qpsk")
    after_residual = correct_frequency_offset(
        after_coarse, FS, residual.residual_frequency_hz
    )
    phys = _physical_residual_hz(
        CFO_HZ, coarse.frequency_offset_hz, residual.residual_frequency_hz
    )
    assert abs(phys) < 0.005
    est = estimate_phase_offset(after_residual, "qpsk")
    assert abs(_wrap_phase_error(est.phase_offset_rad - PHI, np.pi / 2.0)) < 0.05
    assert est.symmetry > 0.9


@pytest.mark.parametrize("true_hz", [0.0, 0.001, -0.001])
def test_near_zero_safety_at_20db(true_hz):
    """No deadband: a noisy estimate at true 0 Hz may be nonzero, but
    must stay inside the 0.005 Hz research target and leave 11B usable."""
    waveform = _qpsk_clean()
    received = apply_frequency_offset(
        apply_phase_offset(waveform, PHI), FS, true_hz
    )
    noisy = add_awgn(received, 20.0, np.random.default_rng(1))
    estimate = estimate_residual_frequency_offset(noisy, FS, "qpsk")
    phys = true_hz - estimate.residual_frequency_hz
    assert abs(phys) < 0.005
    corrected = correct_frequency_offset(noisy, FS, estimate.residual_frequency_hz)
    est = estimate_phase_offset(corrected, "qpsk")
    assert abs(_wrap_phase_error(est.phase_offset_rad - PHI, np.pi / 2.0)) < 0.05
    assert est.symmetry > 0.9


def test_stale_five_hz_residual_then_phase_accepts():
    waveform = qpsk_waveform(np.random.default_rng(202).integers(0, 2, 4096), 8)
    received = apply_frequency_offset(waveform, FS, CFO_HZ)
    stale = correct_frequency_offset(received, FS, 995.0)
    with pytest.raises(ValueError):
        estimate_phase_offset(stale, "qpsk")
    residual = estimate_residual_frequency_offset(stale, FS, "qpsk")
    refined = correct_frequency_offset(stale, FS, residual.residual_frequency_hz)
    independent = _independent_residual_hz(refined, waveform, FS)
    assert abs(independent) < 0.005
    est = estimate_phase_offset(apply_phase_offset(refined, 0.3), "qpsk")
    assert abs(_wrap_phase_error(est.phase_offset_rad - 0.3, np.pi / 2.0)) < 0.05
    assert est.symmetry > 0.9


def test_bpsk_after_true_cfo_then_residual_still_works():
    waveform = _bpsk_clean()
    received = apply_frequency_offset(
        apply_phase_offset(waveform, PHI), FS, CFO_HZ
    )
    residual = estimate_residual_frequency_offset(received, FS, "bpsk")
    corrected = correct_frequency_offset(received, FS, residual.residual_frequency_hz)
    phys = _physical_residual_hz(CFO_HZ, residual.residual_frequency_hz)
    assert abs(phys) < 0.005
    est = estimate_phase_offset(corrected, "bpsk")
    assert abs(_wrap_phase_error(est.phase_offset_rad - PHI, np.pi)) < 0.05
    assert est.symmetry > 0.9


def test_input_not_mutated():
    samples = apply_frequency_offset(_bpsk_clean(nbits=256), FS, 5.0)
    before = samples.copy()
    estimate_residual_frequency_offset(samples, FS, "bpsk")
    np.testing.assert_array_equal(samples, before)


@pytest.mark.parametrize(
    "bad_samples",
    [
        np.array([], dtype=np.complex128),
        np.array([1.0 + 1.0j], dtype=np.complex128),
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
        estimate_residual_frequency_offset(bad_samples, FS, "bpsk")


@pytest.mark.parametrize(
    "fs",
    [0.0, -1000.0, float("nan"), float("inf"), True, 1.0 + 2.0j],
)
def test_invalid_fs_raises(fs):
    samples = np.ones(16, dtype=np.complex128)
    with pytest.raises(ValueError):
        estimate_residual_frequency_offset(samples, fs, "bpsk")


@pytest.mark.parametrize(
    "modulation",
    ["8psk", "QPSK", "bpsk ", "", "pi/2-bpsk", None, 5],
)
def test_unsupported_modulation_raises(modulation):
    samples = np.ones(16, dtype=np.complex128)
    with pytest.raises(ValueError):
        estimate_residual_frequency_offset(samples, FS, modulation)


def test_min_coherence_is_not_part_of_the_public_api():
    samples = np.ones(16, dtype=np.complex128)
    with pytest.raises(TypeError):
        estimate_residual_frequency_offset(
            samples, FS, "bpsk", min_coherence=0.0
        )
