"""Unit tests for the spectrogram utility in iqwav.dsp.spectrogram."""

import numpy as np
import pytest

from iqwav.dsp import spectrogram_data
from iqwav.modulation import generate_iq_tone


def test_output_dimensions_default_parameters():
    _, samples = generate_iq_tone(fs=1000.0, freq=-100.0, duration=2.0)
    time, freqs, power = spectrogram_data(samples, fs=1000.0)
    assert time.shape == (8,)
    assert freqs.shape == (256,)
    assert power.shape == (256, 8)
    assert time.dtype == np.float64
    assert freqs.dtype == np.float64
    assert power.dtype == np.float64


def test_output_dimensions_explicit_parameters():
    _, samples = generate_iq_tone(fs=1000.0, freq=-100.0, duration=2.0)
    time, freqs, power = spectrogram_data(
        samples, fs=1000.0, nperseg=250, noverlap=125
    )
    assert time.shape == (15,)
    assert freqs.shape == (250,)
    assert power.shape == (250, 15)


def test_time_axis_is_monotonic_and_in_range():
    _, samples = generate_iq_tone(fs=1000.0, freq=-100.0, duration=2.0)
    time, _, _ = spectrogram_data(
        samples, fs=1000.0, nperseg=250, noverlap=125
    )
    assert np.all(np.diff(time) > 0)
    assert time[0] >= 0.0
    np.testing.assert_allclose(time[0], 0.125)
    assert time[-1] <= 2.0


def test_frequency_axis_is_centered():
    _, samples = generate_iq_tone(fs=1000.0, freq=-100.0, duration=2.0)
    _, freqs, _ = spectrogram_data(
        samples, fs=1000.0, nperseg=250, noverlap=125
    )
    np.testing.assert_allclose(freqs[0], -500.0)
    np.testing.assert_allclose(freqs[125], 0.0)
    np.testing.assert_allclose(freqs[-1], 496.0)
    assert np.all(np.diff(freqs) > 0)


def test_stationary_iq_tone_peak_across_time():
    _, samples = generate_iq_tone(
        fs=1000.0, freq=-100.0, duration=2.0, amplitude=1.0
    )
    _, freqs, power = spectrogram_data(
        samples, fs=1000.0, nperseg=250, noverlap=125
    )
    peak_rows = np.argmax(power, axis=0)
    assert np.all(freqs[peak_rows] == -100.0)
    for column in power.T:
        assert np.count_nonzero(column > 0.99 * column.max()) == 1


@pytest.mark.parametrize("fs", [0.0, -1000.0, float("nan"), float("inf")])
def test_invalid_fs_raises(fs):
    with pytest.raises(ValueError):
        spectrogram_data(np.ones(64), fs=fs)


def test_invalid_samples_raise():
    for bad in (
        np.ones((4, 4)),
        np.array([]),
        np.array([1.0, np.nan]),
        np.array([1.0 + np.inf * 1j, 2.0]),
    ):
        with pytest.raises(ValueError):
            spectrogram_data(bad, fs=1000.0)


@pytest.mark.parametrize("nperseg", [0, -100, 100.5, "256"])
def test_invalid_nperseg_raises(nperseg):
    with pytest.raises(ValueError):
        spectrogram_data(np.ones(64), fs=1000.0, nperseg=nperseg)


@pytest.mark.parametrize("noverlap", [-1, 250, 300, 12.5, "10"])
def test_invalid_noverlap_raises(noverlap):
    with pytest.raises(ValueError):
        spectrogram_data(np.ones(64), fs=1000.0, nperseg=250, noverlap=noverlap)
