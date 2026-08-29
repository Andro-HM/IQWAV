"""Unit tests for the FIR filter utilities in iqwav.dsp.filters."""

import numpy as np
import pytest

from iqwav.dsp import (
    apply_fir_filter,
    design_bandpass_fir,
    design_highpass_fir,
    design_lowpass_fir,
)
from iqwav.modulation import generate_iq_tone, generate_real_tone


def _steady_amplitude(filtered: np.ndarray) -> float:
    return float(np.max(np.abs(filtered[200:])))


def test_lowpass_coefficients():
    taps = design_lowpass_fir(fs=1000.0, cutoff=200.0, numtaps=101)
    assert taps.shape == (101,)
    assert taps.dtype == np.float64
    assert np.all(np.isfinite(taps))
    taps2 = design_lowpass_fir(fs=1000.0, cutoff=200.0, numtaps=2)
    assert taps2.shape == (2,)
    assert np.all(np.isfinite(taps2))


def test_highpass_coefficients():
    taps = design_highpass_fir(fs=1000.0, cutoff=200.0, numtaps=101)
    assert taps.shape == (101,)
    assert taps.dtype == np.float64
    assert np.all(np.isfinite(taps))


def test_bandpass_coefficients():
    taps = design_bandpass_fir(
        fs=1000.0, lowcut=150.0, highcut=350.0, numtaps=101
    )
    assert taps.shape == (101,)
    assert taps.dtype == np.float64
    assert np.all(np.isfinite(taps))


def test_lowpass_passes_low_tone_better_than_high_tone():
    fs = 1000.0
    taps = design_lowpass_fir(fs=fs, cutoff=200.0)
    _, low = generate_real_tone(fs=fs, freq=50.0, duration=1.0)
    _, high = generate_real_tone(fs=fs, freq=450.0, duration=1.0)
    low_out = _steady_amplitude(apply_fir_filter(low, taps))
    high_out = _steady_amplitude(apply_fir_filter(high, taps))
    assert low_out > 0.9
    assert high_out < 0.01
    assert low_out > 50.0 * high_out


def test_highpass_passes_high_tone_better_than_low_tone():
    fs = 1000.0
    taps = design_highpass_fir(fs=fs, cutoff=200.0)
    _, low = generate_real_tone(fs=fs, freq=50.0, duration=1.0)
    _, high = generate_real_tone(fs=fs, freq=450.0, duration=1.0)
    low_out = _steady_amplitude(apply_fir_filter(low, taps))
    high_out = _steady_amplitude(apply_fir_filter(high, taps))
    assert high_out > 0.9
    assert low_out < 0.01
    assert high_out > 50.0 * low_out


def test_bandpass_passes_inband_tone_better_than_outofband_tones():
    fs = 1000.0
    taps = design_bandpass_fir(fs=fs, lowcut=150.0, highcut=350.0)
    _, inband = generate_real_tone(fs=fs, freq=250.0, duration=1.0)
    _, below = generate_real_tone(fs=fs, freq=50.0, duration=1.0)
    _, above = generate_real_tone(fs=fs, freq=450.0, duration=1.0)
    inband_out = _steady_amplitude(apply_fir_filter(inband, taps))
    below_out = _steady_amplitude(apply_fir_filter(below, taps))
    above_out = _steady_amplitude(apply_fir_filter(above, taps))
    assert inband_out > 0.9
    assert below_out < 0.01
    assert above_out < 0.01
    assert inband_out > 50.0 * below_out
    assert inband_out > 50.0 * above_out


def test_real_input_stays_real_and_length_preserved():
    fs = 1000.0
    taps = design_lowpass_fir(fs=fs, cutoff=200.0)
    _, samples = generate_real_tone(fs=fs, freq=50.0, duration=0.5)
    filtered = apply_fir_filter(samples, taps)
    assert filtered.shape == samples.shape
    assert filtered.dtype == np.float64
    assert np.isrealobj(filtered)


def test_complex_iq_input_supported():
    fs = 1000.0
    taps = design_lowpass_fir(fs=fs, cutoff=200.0)
    _, samples = generate_iq_tone(fs=fs, freq=50.0, duration=1.0)
    filtered = apply_fir_filter(samples, taps)
    assert filtered.shape == samples.shape
    assert filtered.dtype == np.complex128
    assert np.iscomplexobj(filtered)
    np.testing.assert_allclose(np.abs(filtered[200:]), 1.0, rtol=0.01)


@pytest.mark.parametrize("fs", [0.0, -1000.0, float("nan"), float("inf")])
def test_design_invalid_fs_raises(fs):
    with pytest.raises(ValueError):
        design_lowpass_fir(fs=fs, cutoff=200.0)
    with pytest.raises(ValueError):
        design_highpass_fir(fs=fs, cutoff=200.0)
    with pytest.raises(ValueError):
        design_bandpass_fir(fs=fs, lowcut=150.0, highcut=350.0)


@pytest.mark.parametrize("numtaps", [0, 1, -5, 100.5, "101"])
def test_design_invalid_numtaps_raises(numtaps):
    with pytest.raises(ValueError):
        design_lowpass_fir(fs=1000.0, cutoff=200.0, numtaps=numtaps)
    with pytest.raises(ValueError):
        design_highpass_fir(fs=1000.0, cutoff=200.0, numtaps=numtaps)
    with pytest.raises(ValueError):
        design_bandpass_fir(
            fs=1000.0, lowcut=150.0, highcut=350.0, numtaps=numtaps
        )


@pytest.mark.parametrize(
    "cutoff",
    [0.0, -100.0, 500.0, 600.0, float("nan"), float("inf")],
)
def test_design_invalid_cutoff_raises(cutoff):
    with pytest.raises(ValueError):
        design_lowpass_fir(fs=1000.0, cutoff=cutoff)
    with pytest.raises(ValueError):
        design_highpass_fir(fs=1000.0, cutoff=cutoff)


@pytest.mark.parametrize(
    ("lowcut", "highcut"),
    [
        (0.0, 350.0),
        (-10.0, 350.0),
        (350.0, 150.0),
        (250.0, 250.0),
        (150.0, 500.0),
        (150.0, 600.0),
        (float("nan"), 350.0),
        (150.0, float("inf")),
    ],
)
def test_design_invalid_band_raises(lowcut, highcut):
    with pytest.raises(ValueError):
        design_bandpass_fir(fs=1000.0, lowcut=lowcut, highcut=highcut)


def test_apply_invalid_samples_raise():
    taps = design_lowpass_fir(fs=1000.0, cutoff=200.0)
    for bad in (np.ones((4, 4)), np.array([]), np.array([1.0, np.nan])):
        with pytest.raises(ValueError):
            apply_fir_filter(bad, taps)


def test_apply_invalid_taps_raise():
    samples = np.ones(64)
    for bad in (np.ones((4, 4)), np.array([]), np.array([1.0, np.nan])):
        with pytest.raises(ValueError):
            apply_fir_filter(samples, bad)
