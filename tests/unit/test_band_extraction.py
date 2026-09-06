import numpy as np
import pytest

from iqwav.dsp import extract_band
from iqwav.dsp import magnitude_spectrum


def _tone(fs, frequency, n=4096):
    return np.exp(2j * np.pi * frequency * np.arange(n) / fs)


@pytest.mark.parametrize("frequency", [120.0, -120.0, 0.0])
def test_extract_band_tunes_selected_signed_band_to_dc(frequency):
    fs = 1000.0
    result = extract_band(_tone(fs, frequency), fs, frequency - 40, frequency + 40, transition_hz=30, numtaps=51)
    valid = result.samples[result.valid_start_sample:]
    freqs, magnitude = magnitude_spectrum(valid, fs)
    assert abs(freqs[np.argmax(magnitude)]) < 2.0
    assert result.center_hz == pytest.approx(frequency)
    assert result.output_center_hz == 0.0
    assert result.samples.size == 4096


def test_extract_band_attenuates_out_of_band_and_validates():
    fs = 1000.0; n = 4096
    x = _tone(fs, 100, n) + _tone(fs, 300, n)
    result = extract_band(x, fs, 60, 140, transition_hz=25, numtaps=101)
    valid = result.samples[result.valid_start_sample:]
    freqs, magnitude = magnitude_spectrum(valid, fs)
    assert magnitude[np.argmin(abs(freqs))] > 10 * magnitude[np.argmin(abs(freqs - 200))]
    assert result.filter_delay_samples == 50
    with pytest.raises(ValueError): extract_band(x, fs, 100, 50, transition_hz=10)
    with pytest.raises(ValueError): extract_band(x, fs, -501, -450, transition_hz=10)
