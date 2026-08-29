"""Unit tests for the PSD utilities in iqwav.dsp.psd."""

import numpy as np
import pytest

from iqwav.dsp import periodogram_psd, welch_psd
from iqwav.modulation import generate_iq_tone, generate_real_tone

ESTIMATORS = [periodogram_psd, welch_psd]


def test_periodogram_output_length():
    _, samples = generate_iq_tone(fs=1000.0, freq=-100.0, duration=1.0)
    freqs, psd = periodogram_psd(samples, fs=1000.0)
    assert freqs.shape == samples.shape
    assert psd.shape == samples.shape
    assert freqs.dtype == np.float64
    assert psd.dtype == np.float64


def test_welch_default_output_length():
    _, samples = generate_iq_tone(fs=1000.0, freq=-100.0, duration=1.0)
    freqs, psd = welch_psd(samples, fs=1000.0)
    assert freqs.shape == psd.shape
    assert freqs.shape == (256,)
    assert freqs.dtype == np.float64
    assert psd.dtype == np.float64


def test_welch_explicit_nperseg_output_length():
    _, samples = generate_iq_tone(fs=1000.0, freq=-100.0, duration=1.0)
    freqs, psd = welch_psd(samples, fs=1000.0, nperseg=250)
    assert freqs.shape == psd.shape
    assert freqs.shape == (250,)
    assert freqs.dtype == np.float64
    assert psd.dtype == np.float64


def test_periodogram_frequency_ordering():
    _, samples = generate_iq_tone(fs=1000.0, freq=-100.0, duration=1.0)
    freqs, _ = periodogram_psd(samples, fs=1000.0)
    np.testing.assert_allclose(freqs[0], -500.0)
    np.testing.assert_allclose(freqs[samples.size // 2], 0.0)
    np.testing.assert_allclose(freqs[-1], 499.0)
    assert np.all(np.diff(freqs) > 0)


def test_welch_frequency_ordering():
    _, samples = generate_iq_tone(fs=1000.0, freq=-100.0, duration=1.0)
    freqs, _ = welch_psd(samples, fs=1000.0, nperseg=250)
    np.testing.assert_allclose(freqs[0], -500.0)
    np.testing.assert_allclose(freqs[125], 0.0)
    assert np.all(np.diff(freqs) > 0)


def test_iq_tone_periodogram_peak_at_signed_frequency():
    _, samples = generate_iq_tone(
        fs=1000.0, freq=-100.0, duration=1.0, amplitude=1.0
    )
    freqs, psd = periodogram_psd(samples, fs=1000.0)
    assert freqs[np.argmax(psd)] == -100.0
    np.testing.assert_allclose(psd.max(), 1.0, rtol=1e-6)
    assert np.count_nonzero(psd > 0.99 * psd.max()) == 1


def test_iq_tone_welch_peak_at_signed_frequency():
    _, samples = generate_iq_tone(
        fs=1000.0, freq=-100.0, duration=1.0, amplitude=1.0
    )
    freqs, psd = welch_psd(samples, fs=1000.0, nperseg=250)
    assert freqs[np.argmax(psd)] == -100.0
    assert np.count_nonzero(psd > 0.99 * psd.max()) == 1


def test_welch_default_peak_near_signed_frequency():
    _, samples = generate_iq_tone(fs=1000.0, freq=-100.0, duration=1.0)
    freqs, psd = welch_psd(samples, fs=1000.0)
    peak_freq = freqs[np.argmax(psd)]
    assert abs(peak_freq - (-100.0)) <= 1000.0 / 256


def test_real_tone_symmetric_peaks():
    _, samples = generate_real_tone(
        fs=1000.0, freq=100.0, duration=1.0, amplitude=1.0
    )
    freqs, psd = periodogram_psd(samples, fs=1000.0)
    peak_freqs = np.sort(freqs[psd > 0.99 * psd.max()])
    np.testing.assert_allclose(peak_freqs, [-100.0, 100.0])
    np.testing.assert_allclose(
        psd[np.isclose(freqs, 100.0)], psd[np.isclose(freqs, -100.0)]
    )
    np.testing.assert_allclose(psd.max(), 0.25, rtol=1e-6)


@pytest.mark.parametrize("estimator", ESTIMATORS)
@pytest.mark.parametrize("fs", [0.0, -1000.0, float("nan"), float("inf")])
def test_invalid_fs_raises(estimator, fs):
    with pytest.raises(ValueError):
        estimator(np.ones(64), fs=fs)


@pytest.mark.parametrize("estimator", ESTIMATORS)
def test_invalid_samples_raise(estimator):
    with pytest.raises(ValueError):
        estimator(np.ones((4, 4)), fs=1000.0)
    with pytest.raises(ValueError):
        estimator(np.array([]), fs=1000.0)
    with pytest.raises(ValueError):
        estimator(np.array([1.0, np.nan]), fs=1000.0)
    with pytest.raises(ValueError):
        estimator(np.array([1.0 + np.inf * 1j, 2.0]), fs=1000.0)


@pytest.mark.parametrize("nperseg", [0, -100, 100.5, "256"])
def test_invalid_nperseg_raises(nperseg):
    _, samples = generate_iq_tone(fs=1000.0, freq=-100.0, duration=0.1)
    with pytest.raises(ValueError):
        welch_psd(samples, fs=1000.0, nperseg=nperseg)
