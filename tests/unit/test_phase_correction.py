"""Unit tests for known-phase correction in iqwav.synchronization.phase."""

import numpy as np
import pytest

from iqwav.demod import bpsk_demodulate, qpsk_demodulate
from iqwav.dsp import apply_frequency_offset, apply_phase_offset
from iqwav.modulation import bpsk_waveform, qpsk_waveform
from iqwav.synchronization import correct_frequency_offset, correct_phase_offset

FS = 80_000.0


def _tone(count=64):
    return np.exp(1j * np.linspace(0.0, 6.0, count))


def test_zero_phase_returns_equal_new_copy():
    samples = _tone()
    corrected = correct_phase_offset(samples, 0.0)
    np.testing.assert_array_equal(corrected, samples)
    assert corrected is not samples
    assert not np.shares_memory(corrected, samples)


@pytest.mark.parametrize("phase", [0.7, -1.2, 3.0, 10.5, -7.25])
def test_round_trip_inverts_apply_phase_offset(phase):
    samples = _tone(128)
    received = apply_phase_offset(samples, phase)
    corrected = correct_phase_offset(received, phase)
    np.testing.assert_allclose(corrected, samples, atol=1e-9)


def test_magnitude_preserved():
    bits = np.random.default_rng(3).integers(0, 2, 256)
    samples = qpsk_waveform(bits, 8)
    corrected = correct_phase_offset(samples, 2.1)
    np.testing.assert_allclose(np.abs(corrected), np.abs(samples))


def test_length_dtype_and_new_array():
    samples = _tone()
    corrected = correct_phase_offset(samples, -0.35)
    assert corrected.shape == samples.shape
    assert corrected.dtype == np.complex128
    assert np.iscomplexobj(corrected)
    assert corrected is not samples
    assert not np.shares_memory(corrected, samples)


def test_input_not_mutated():
    samples = _tone()
    before = samples.copy()
    correct_phase_offset(samples, 1.234)
    np.testing.assert_array_equal(samples, before)


def test_deterministic_output():
    samples = _tone()
    first = correct_phase_offset(samples, 0.77)
    second = correct_phase_offset(samples, 0.77)
    np.testing.assert_array_equal(first, second)


def test_output_memory_independence():
    samples = _tone()
    before = samples.copy()
    corrected = correct_phase_offset(samples, 0.4)
    corrected[0] = 123.0 + 456.0j
    np.testing.assert_array_equal(samples, before)


def test_complex64_input_returns_complex128():
    samples = np.ones(16, dtype=np.complex64)
    corrected = correct_phase_offset(samples, 0.5)
    assert corrected.dtype == np.complex128


def test_correction_does_not_estimate_phase():
    """Only the SUPPLIED phase is removed: correcting with a wrong value
    leaves the constellation rotated (no internal estimation)."""
    samples = np.ones(16, dtype=np.complex128)
    corrected = correct_phase_offset(apply_phase_offset(samples, 0.4), 0.9)
    assert not np.allclose(corrected, samples, atol=1e-3)
    np.testing.assert_allclose(
        corrected, np.exp(1j * (0.4 - 0.9)) * samples, atol=1e-12
    )


def test_correction_with_true_phase_recovers_clean_bpsk():
    """Known-true-phase correction verified independently against the
    clean reference; no phase estimator is involved."""
    bits = np.random.default_rng(101).integers(0, 2, 512)
    waveform = bpsk_waveform(bits, 8)
    received = apply_phase_offset(waveform, 0.9)
    corrected = correct_phase_offset(received, 0.9)
    np.testing.assert_allclose(corrected, waveform, atol=1e-9)
    np.testing.assert_array_equal(bpsk_demodulate(corrected, 8), bits)


def test_correction_with_true_phase_recovers_clean_qpsk():
    bits = np.random.default_rng(102).integers(0, 2, 1024)
    waveform = qpsk_waveform(bits, 8)
    received = apply_phase_offset(waveform, -2.3)
    corrected = correct_phase_offset(received, -2.3)
    np.testing.assert_allclose(corrected, waveform, atol=1e-9)
    np.testing.assert_array_equal(qpsk_demodulate(corrected, 8), bits)


def test_cfo_then_phase_correction_chain_recovers_clean():
    """Static phase remains after CFO correction (Module 11A); removing
    the known CFO first and then the known static phase recovers the
    clean waveform exactly."""
    bits = np.random.default_rng(102).integers(0, 2, 1024)
    waveform = qpsk_waveform(bits, 8)
    received = apply_frequency_offset(apply_phase_offset(waveform, 0.9), FS, 750.0)
    fixed = correct_frequency_offset(received, FS, 750.0)
    corrected = correct_phase_offset(fixed, 0.9)
    np.testing.assert_allclose(corrected, waveform, atol=1e-9)
    np.testing.assert_array_equal(qpsk_demodulate(corrected, 8), bits)


@pytest.mark.parametrize(
    "phase",
    [float("nan"), float("inf"), -float("inf"), True, False],
)
def test_invalid_phase_raises(phase):
    samples = np.ones(16, dtype=np.complex128)
    with pytest.raises(ValueError):
        correct_phase_offset(samples, phase)


@pytest.mark.parametrize(
    "bad_samples",
    [
        np.array([], dtype=np.complex128),
        np.ones((4, 4), dtype=np.complex128),
        np.ones(16),
        np.ones(16, dtype=np.bool_),
        np.array([1.0 + 2.0j, np.nan + 1.0j]),
        np.array([1.0 + np.inf * 1j, 2.0 + 1.0j]),
    ],
)
def test_invalid_samples_raise(bad_samples):
    with pytest.raises(ValueError):
        correct_phase_offset(bad_samples, 0.5)
