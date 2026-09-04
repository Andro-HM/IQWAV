"""Unit tests for known-CFO correction in iqwav.synchronization.frequency."""

import numpy as np
import pytest

from iqwav.demod import bpsk_demodulate, qpsk_demodulate
from iqwav.dsp import apply_frequency_offset, apply_phase_offset
from iqwav.estimation import estimate_frequency_offset
from iqwav.modulation import bpsk_waveform, generate_iq_tone, qpsk_waveform
from iqwav.synchronization import correct_frequency_offset


def _iq_tone(fs=1000.0, freq=100.0, duration=0.5, amplitude=1.5):
    _, samples = generate_iq_tone(
        fs=fs, freq=freq, duration=duration, amplitude=amplitude
    )
    return samples


def test_zero_offset_leaves_signal_unchanged():
    samples = _iq_tone(duration=0.05)
    corrected = correct_frequency_offset(samples, 1000.0, 0.0)
    np.testing.assert_array_equal(corrected, samples)


def test_positive_offset_round_trip():
    samples = _iq_tone()
    received = apply_frequency_offset(samples, 1000.0, 250.0)
    corrected = correct_frequency_offset(received, 1000.0, 250.0)
    np.testing.assert_allclose(corrected, samples, atol=1e-9)


def test_negative_offset_round_trip():
    samples = _iq_tone()
    received = apply_frequency_offset(samples, 1000.0, -180.0)
    corrected = correct_frequency_offset(received, 1000.0, -180.0)
    np.testing.assert_allclose(corrected, samples, atol=1e-9)


@pytest.mark.parametrize("fs", [500.0, 1000.0, 44100.0, 2_000_000.0])
def test_multiple_sample_rates_round_trip(fs):
    samples = _iq_tone(fs=fs, freq=fs / 20.0, duration=0.01, amplitude=0.7)
    offset = fs / 37.0
    received = apply_frequency_offset(samples, fs, offset)
    corrected = correct_frequency_offset(received, fs, offset)
    np.testing.assert_allclose(corrected, samples, atol=1e-9)


def test_magnitude_preserved():
    samples = _iq_tone(freq=123.0)
    corrected = correct_frequency_offset(samples, 1000.0, 321.0)
    np.testing.assert_allclose(np.abs(corrected), np.abs(samples))


def test_length_dtype_and_new_array():
    samples = _iq_tone(duration=0.05)
    corrected = correct_frequency_offset(samples, 1000.0, 42.0)
    assert corrected.shape == samples.shape
    assert corrected.dtype == np.complex128
    assert np.iscomplexobj(corrected)
    assert corrected is not samples


def test_input_not_mutated():
    samples = _iq_tone(duration=0.05)
    before = samples.copy()
    correct_frequency_offset(samples, 1000.0, 42.0)
    np.testing.assert_array_equal(samples, before)


def test_deterministic_output():
    samples = _iq_tone(duration=0.05)
    first = correct_frequency_offset(samples, 1000.0, 17.0)
    second = correct_frequency_offset(samples, 1000.0, 17.0)
    np.testing.assert_array_equal(first, second)


def test_bpsk_injected_cfo_correction_and_demodulation():
    fs = 2000.0
    bits = np.array([1, 0, 0, 1, 1, 0, 1, 1, 0, 0], dtype=np.int64)
    waveform = bpsk_waveform(bits, 8)
    received = apply_frequency_offset(waveform, fs, 60.0)
    corrected = correct_frequency_offset(received, fs, 60.0)
    np.testing.assert_allclose(corrected, waveform, atol=1e-9)
    np.testing.assert_array_equal(bpsk_demodulate(corrected, 8), bits)


def test_qpsk_injected_cfo_correction_and_demodulation():
    fs = 2000.0
    bits = np.array([0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0], dtype=np.int64)
    waveform = qpsk_waveform(bits, 10)
    received = apply_frequency_offset(waveform, fs, -95.0)
    corrected = correct_frequency_offset(received, fs, -95.0)
    np.testing.assert_allclose(corrected, waveform, atol=1e-9)
    np.testing.assert_array_equal(qpsk_demodulate(corrected, 10), bits)


def test_offset_larger_than_symbol_rate_still_corrects():
    fs = 2000.0
    bits = np.array([0, 1, 0, 1, 1, 0], dtype=np.int64)
    waveform = bpsk_waveform(bits, 4)
    received = apply_frequency_offset(waveform, fs, 900.0)
    corrected = correct_frequency_offset(received, fs, 900.0)
    np.testing.assert_allclose(corrected, waveform, atol=1e-9)


def test_static_phase_remains_after_cfo_correction():
    fs = 1000.0
    bits = np.random.default_rng(5).integers(0, 2, 512)
    waveform = bpsk_waveform(bits, 8)
    rotated = apply_phase_offset(waveform, 0.8)
    received = apply_frequency_offset(rotated, fs, 137.0)
    corrected = correct_frequency_offset(received, fs, 137.0)
    np.testing.assert_allclose(corrected, rotated, atol=1e-9)
    assert not np.allclose(corrected, waveform, atol=1e-6)


@pytest.mark.parametrize(
    "fs",
    [0.0, -1000.0, float("nan"), float("inf"), True, 1.0 + 2.0j],
)
def test_invalid_fs_raises(fs):
    samples = np.ones(16, dtype=np.complex128)
    with pytest.raises(ValueError):
        correct_frequency_offset(samples, fs, 10.0)


@pytest.mark.parametrize(
    "frequency_offset_hz",
    [float("nan"), float("inf"), -float("inf"), True, False],
)
def test_invalid_frequency_offset_raises(frequency_offset_hz):
    samples = np.ones(16, dtype=np.complex128)
    with pytest.raises(ValueError):
        correct_frequency_offset(samples, 1000.0, frequency_offset_hz)


def test_real_valued_input_rejected():
    with pytest.raises(ValueError):
        correct_frequency_offset(np.ones(16), 1000.0, 10.0)


def test_empty_input_rejected():
    with pytest.raises(ValueError):
        correct_frequency_offset(np.array([], dtype=np.complex128), 1000.0, 10.0)


def test_multidimensional_input_rejected():
    with pytest.raises(ValueError):
        correct_frequency_offset(np.ones((4, 4), dtype=np.complex128), 1000.0, 10.0)


def test_non_finite_input_rejected():
    with pytest.raises(ValueError):
        correct_frequency_offset(
            np.array([1.0 + 2.0j, np.nan + 1.0j, 3.0 + 4.0j]), 1000.0, 10.0
        )
    with pytest.raises(ValueError):
        correct_frequency_offset(
            np.array([1.0 + np.inf * 1j, 2.0 + 1j]), 1000.0, 10.0
        )


@pytest.mark.parametrize("modulation", ["bpsk", "qpsk"])
def test_estimator_self_consistency_after_correction(modulation):
    """Estimator self-consistency: re-measuring with the same lag-1
    estimator after correcting by its own estimate reports near zero,
    because the estimator's finite-record phase bias cancels itself.
    This is NOT proof of a zero physical residual CFO; the independent
    truth-based residual is checked separately below."""
    fs = 80_000.0
    if modulation == "bpsk":
        bits = np.random.default_rng(101).integers(0, 2, 4096)
        waveform = bpsk_waveform(bits, 8)
    else:
        bits = np.random.default_rng(102).integers(0, 2, 8192)
        waveform = qpsk_waveform(bits, 8)
    received = apply_frequency_offset(waveform, fs, 1000.0)
    initial = estimate_frequency_offset(received, fs)
    corrected = correct_frequency_offset(
        received, fs, initial.frequency_offset_hz
    )
    residual = estimate_frequency_offset(corrected, fs)
    assert abs(residual.frequency_offset_hz) < 50.0
    assert abs(residual.frequency_offset_hz) < abs(initial.frequency_offset_hz)
    assert residual.coherence > 0.5


@pytest.mark.parametrize(
    ("modulation", "expect_near_zero"),
    [("qpsk", False), ("bpsk", True)],
)
def test_independent_truth_based_residual_measurement(
    modulation, expect_near_zero
):
    """Independent truth-based residual: with the clean reference known,
    the residual CFO is measured by unwrapping corrected/clean phase and
    fitting its slope, which does not share the estimator's self-cancelling
    bias. The measured residual must match ``f_true - f_hat``; for QPSK it
    is genuinely nonzero when the finite-record estimator bias is nonzero,
    and is not falsely required to be exactly zero."""
    fs = 80_000.0
    if modulation == "qpsk":
        bits = np.random.default_rng(102).integers(0, 2, 8192)
        clean = qpsk_waveform(bits, 8)
    else:
        bits = np.random.default_rng(101).integers(0, 2, 4096)
        clean = bpsk_waveform(bits, 8)
    f_true = 1000.0
    received = apply_frequency_offset(clean, fs, f_true)
    estimate = estimate_frequency_offset(received, fs)
    corrected = correct_frequency_offset(
        received, fs, estimate.frequency_offset_hz
    )

    nonzero = np.abs(clean) > 0.0
    n = np.nonzero(nonzero)[0].astype(np.float64)
    ratio = corrected[nonzero] / clean[nonzero]
    phase = np.unwrap(np.angle(ratio))
    slope = np.polyfit(n, phase, 1)[0]
    measured_residual_hz = slope * fs / (2.0 * np.pi)
    expected_residual_hz = f_true - estimate.frequency_offset_hz

    np.testing.assert_allclose(
        measured_residual_hz, expected_residual_hz, atol=0.5
    )
    assert abs(measured_residual_hz) < 50.0
    if expect_near_zero:
        assert abs(measured_residual_hz) < 0.5
    else:
        assert abs(estimate.frequency_offset_hz - f_true) > 0.1
        assert abs(measured_residual_hz) > 0.5
