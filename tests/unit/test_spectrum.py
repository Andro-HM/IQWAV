"""Unit tests for the magnitude-spectrum utility in iqwav.dsp.spectrum."""

import numpy as np
import pytest

from iqwav.dsp import magnitude_spectrum
from iqwav.modulation import generate_iq_tone, generate_real_tone


def test_output_length_and_dtype():
    _, samples = generate_real_tone(fs=1000.0, freq=100.0, duration=0.01)
    for flag in (True, False):
        freqs, magnitude = magnitude_spectrum(samples, fs=1000.0, fftshift=flag)
        assert freqs.shape == samples.shape
        assert magnitude.shape == samples.shape
        assert freqs.dtype == np.float64
        assert magnitude.dtype == np.float64


def test_unshifted_frequency_axis_ordering():
    _, samples = generate_real_tone(fs=1000.0, freq=100.0, duration=0.01)
    freqs, _ = magnitude_spectrum(samples, fs=1000.0, fftshift=False)
    n = samples.size
    assert freqs[0] == 0.0
    np.testing.assert_allclose(freqs[1], 1000.0 / n)
    np.testing.assert_allclose(freqs[n // 2], -500.0)
    np.testing.assert_allclose(freqs[-1], -1000.0 / n)


def test_shifted_frequency_axis_ordering():
    _, samples = generate_real_tone(fs=1000.0, freq=100.0, duration=0.01)
    freqs, _ = magnitude_spectrum(samples, fs=1000.0, fftshift=True)
    n = samples.size
    np.testing.assert_allclose(freqs[0], -500.0)
    np.testing.assert_allclose(freqs[n // 2], 0.0)
    np.testing.assert_allclose(freqs[-1], 500.0 - 1000.0 / n)
    assert np.all(np.diff(freqs) > 0)


def test_real_tone_peaks_at_plus_minus_frequency():
    _, samples = generate_real_tone(fs=1000.0, freq=100.0, duration=1.0, amplitude=0.8)
    freqs, magnitude = magnitude_spectrum(samples, fs=1000.0, fftshift=True)
    peak_freqs = np.sort(freqs[magnitude > 0.99 * magnitude.max()])
    np.testing.assert_allclose(peak_freqs, [-100.0, 100.0])
    np.testing.assert_allclose(magnitude.max(), 400.0, rtol=1e-6)


def test_iq_tone_single_peak_at_signed_frequency():
    _, samples = generate_iq_tone(fs=1000.0, freq=-100.0, duration=1.0, amplitude=0.8)
    freqs, magnitude = magnitude_spectrum(samples, fs=1000.0, fftshift=True)
    assert freqs[np.argmax(magnitude)] == -100.0
    np.testing.assert_allclose(magnitude.max(), 800.0, rtol=1e-6)
    assert np.count_nonzero(magnitude > 0.99 * magnitude.max()) == 1


def test_iq_tone_unshifted_bin_index():
    _, samples = generate_iq_tone(fs=1000.0, freq=-100.0, duration=1.0, amplitude=0.8)
    freqs, magnitude = magnitude_spectrum(samples, fs=1000.0, fftshift=False)
    assert np.argmax(magnitude) == 900
    assert freqs[900] == -100.0


def test_shifting_reorders_but_preserves_values():
    _, samples = generate_iq_tone(
        fs=1000.0, freq=-100.0, duration=0.02, amplitude=1.0, phase=0.3
    )
    freqs_shift, mag_shift = magnitude_spectrum(samples, fs=1000.0, fftshift=True)
    freqs_plain, mag_plain = magnitude_spectrum(samples, fs=1000.0, fftshift=False)
    np.testing.assert_allclose(np.sort(mag_shift), np.sort(mag_plain))
    np.testing.assert_allclose(np.sort(freqs_shift), np.sort(freqs_plain))


@pytest.mark.parametrize("fs", [0.0, -1000.0, float("nan"), float("inf")])
def test_invalid_fs_raises(fs):
    with pytest.raises(ValueError):
        magnitude_spectrum(np.ones(8), fs=fs)


def test_two_dimensional_samples_raise():
    with pytest.raises(ValueError):
        magnitude_spectrum(np.ones((4, 4)), fs=1000.0)


def test_empty_samples_raise():
    with pytest.raises(ValueError):
        magnitude_spectrum(np.array([]), fs=1000.0)


def test_nonfinite_samples_raise():
    with pytest.raises(ValueError):
        magnitude_spectrum(np.array([1.0, np.nan, 2.0]), fs=1000.0)
    with pytest.raises(ValueError):
        magnitude_spectrum(np.array([1.0 + 2.0j, 1.0 + np.inf * 1j]), fs=1000.0)
