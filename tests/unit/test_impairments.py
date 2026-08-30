"""Unit tests for IQ impairment utilities in iqwav.dsp.impairments."""

import numpy as np
import pytest

from iqwav.dsp import apply_frequency_offset, apply_phase_offset, magnitude_spectrum
from iqwav.modulation import generate_iq_tone


def test_zero_frequency_offset_leaves_signal_unchanged():
    _, samples = generate_iq_tone(fs=1000.0, freq=100.0, duration=0.05, amplitude=1.5)
    shifted = apply_frequency_offset(samples, fs=1000.0, freq_offset_hz=0.0)
    np.testing.assert_allclose(shifted, samples)


def test_positive_frequency_offset_shifts_spectrum_up():
    fs = 1000.0
    _, samples = generate_iq_tone(fs=fs, freq=100.0, duration=1.0)
    shifted = apply_frequency_offset(samples, fs=fs, freq_offset_hz=100.0)
    freqs, magnitude = magnitude_spectrum(shifted, fs=fs)
    assert freqs[np.argmax(magnitude)] == 200.0
    assert np.count_nonzero(magnitude > 0.99 * magnitude.max()) == 1


def test_negative_frequency_offset_shifts_spectrum_down():
    fs = 1000.0
    _, samples = generate_iq_tone(fs=fs, freq=250.0, duration=1.0)
    shifted = apply_frequency_offset(samples, fs=fs, freq_offset_hz=-100.0)
    freqs, magnitude = magnitude_spectrum(shifted, fs=fs)
    assert freqs[np.argmax(magnitude)] == 150.0


def test_frequency_offset_preserves_magnitude():
    _, samples = generate_iq_tone(fs=1000.0, freq=123.0, duration=0.5, amplitude=1.5)
    shifted = apply_frequency_offset(samples, fs=1000.0, freq_offset_hz=77.0)
    np.testing.assert_allclose(np.abs(shifted), np.abs(samples))


def test_zero_phase_offset_leaves_signal_unchanged():
    _, samples = generate_iq_tone(fs=1000.0, freq=100.0, duration=0.05, amplitude=1.5)
    rotated = apply_phase_offset(samples, phase_rad=0.0)
    np.testing.assert_allclose(rotated, samples)


def test_phase_offset_rotates_known_iq_sample():
    _, ones = generate_iq_tone(fs=1000.0, freq=0.0, duration=0.001, amplitude=1.0)
    assert np.all(np.isclose(ones, 1.0))
    np.testing.assert_allclose(apply_phase_offset(ones, phase_rad=np.pi / 2), 1j)
    np.testing.assert_allclose(apply_phase_offset(ones, phase_rad=np.pi), -1.0)


def test_phase_offset_preserves_magnitude():
    _, samples = generate_iq_tone(fs=1000.0, freq=123.0, duration=0.5, amplitude=1.5)
    rotated = apply_phase_offset(samples, phase_rad=0.7)
    np.testing.assert_allclose(np.abs(rotated), np.abs(samples))


def test_output_shape_and_complexity():
    _, samples = generate_iq_tone(fs=1000.0, freq=100.0, duration=0.05)
    for output in (
        apply_frequency_offset(samples, fs=1000.0, freq_offset_hz=50.0),
        apply_phase_offset(samples, phase_rad=0.4),
    ):
        assert output.shape == samples.shape
        assert output.dtype == np.complex128
        assert np.iscomplexobj(output)


@pytest.mark.parametrize("fs", [0.0, -1000.0, float("nan"), float("inf")])
def test_invalid_fs_raises(fs):
    samples = np.ones(16, dtype=np.complex128)
    with pytest.raises(ValueError):
        apply_frequency_offset(samples, fs=fs, freq_offset_hz=10.0)


@pytest.mark.parametrize("offset", [float("nan"), float("inf"), -float("inf")])
def test_invalid_freq_offset_raises(offset):
    samples = np.ones(16, dtype=np.complex128)
    with pytest.raises(ValueError):
        apply_frequency_offset(samples, fs=1000.0, freq_offset_hz=offset)


@pytest.mark.parametrize("phase", [float("nan"), float("inf"), -float("inf")])
def test_invalid_phase_raises(phase):
    samples = np.ones(16, dtype=np.complex128)
    with pytest.raises(ValueError):
        apply_phase_offset(samples, phase_rad=phase)


@pytest.mark.parametrize(
    "apply",
    [
        lambda s: apply_frequency_offset(s, fs=1000.0, freq_offset_hz=10.0),
        lambda s: apply_phase_offset(s, phase_rad=0.5),
    ],
)
def test_invalid_samples_raise(apply):
    for bad in (
        np.ones(16),
        np.ones((4, 4), dtype=np.complex128),
        np.array([], dtype=np.complex128),
        np.array([1.0 + 2.0j, np.nan + 1.0j]),
    ):
        with pytest.raises(ValueError):
            apply(bad)
