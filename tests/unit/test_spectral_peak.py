"""Unit tests for the dominant spectral peak estimator in iqwav.estimation."""

import numpy as np
import pytest

from iqwav.estimation import PeakFrequencyEstimate, estimate_peak_frequency
from iqwav.modulation import generate_iq_tone, generate_real_tone

FS = 1000.0
N = 1000


def test_complex_positive_exact_bin_tone():
    _, samples = generate_iq_tone(fs=FS, freq=137.0, duration=1.0)
    result = estimate_peak_frequency(samples, FS)
    assert isinstance(result, PeakFrequencyEstimate)
    assert result.bin_frequency_hz == 137.0
    assert result.bin_index == 137
    assert result.refined is True
    assert result.frequency_hz == pytest.approx(137.0, abs=0.05)
    assert result.frequency_hz > 0.0


def test_complex_negative_exact_bin_tone():
    _, samples = generate_iq_tone(fs=FS, freq=-137.0, duration=1.0)
    result = estimate_peak_frequency(samples, FS)
    assert result.bin_frequency_hz == -137.0
    assert result.bin_index == 863
    assert result.frequency_hz < 0.0
    assert result.frequency_hz == pytest.approx(-137.0, abs=0.05)


def test_real_tone_returns_nonnegative_frequency():
    _, positive = generate_real_tone(fs=FS, freq=137.0, duration=1.0)
    result = estimate_peak_frequency(positive, FS)
    assert result.frequency_hz >= 0.0
    assert result.frequency_hz == pytest.approx(137.0, abs=0.05)

    _, negative = generate_real_tone(fs=FS, freq=-137.0, duration=1.0)
    result_negative = estimate_peak_frequency(negative, FS)
    assert result_negative.frequency_hz >= 0.0
    assert result_negative.frequency_hz == pytest.approx(137.0, abs=0.05)


def test_off_bin_tone_refinement_improves_on_raw_bin():
    true_frequency = 137.4
    _, samples = generate_iq_tone(fs=FS, freq=true_frequency, duration=1.0)
    refined = estimate_peak_frequency(samples, FS, refine=True)
    raw = estimate_peak_frequency(samples, FS, refine=False)
    assert raw.frequency_hz == raw.bin_frequency_hz == 137.0
    assert refined.bin_frequency_hz == raw.bin_frequency_hz
    assert raw.refined is False
    assert refined.refined is True
    raw_error = abs(raw.frequency_hz - true_frequency)
    refined_error = abs(refined.frequency_hz - true_frequency)
    assert refined_error < raw_error
    assert refined_error < 0.5 * raw_error


def test_off_bin_negative_tone_refinement():
    true_frequency = -137.4
    _, samples = generate_iq_tone(fs=FS, freq=true_frequency, duration=1.0)
    refined = estimate_peak_frequency(samples, FS, refine=True)
    raw = estimate_peak_frequency(samples, FS, refine=False)
    assert refined.frequency_hz < 0.0
    assert abs(refined.frequency_hz - true_frequency) < abs(
        raw.frequency_hz - true_frequency
    )


def test_amplitude_scaling_does_not_change_frequency():
    _, samples = generate_iq_tone(fs=FS, freq=137.0, duration=1.0)
    base = estimate_peak_frequency(samples, FS)
    scaled = estimate_peak_frequency(samples * 5.0, FS)
    np.testing.assert_allclose(
        scaled.frequency_hz, base.frequency_hz, atol=1e-3
    )
    assert scaled.bin_index == base.bin_index


def test_phase_rotation_does_not_change_frequency():
    _, samples = generate_iq_tone(fs=FS, freq=137.0, duration=1.0)
    base = estimate_peak_frequency(samples, FS)
    rotated = estimate_peak_frequency(samples * np.exp(1j * 0.7), FS)
    np.testing.assert_allclose(
        rotated.frequency_hz, base.frequency_hz, atol=1e-3
    )


def test_stronger_of_two_tones_is_selected():
    _, strong = generate_iq_tone(fs=FS, freq=180.0, duration=1.0, amplitude=1.0)
    _, weak = generate_iq_tone(fs=FS, freq=-310.0, duration=1.0, amplitude=0.2)
    result = estimate_peak_frequency(strong + weak, FS)
    assert result.frequency_hz == pytest.approx(180.0, abs=0.05)

    _, strong_negative = generate_iq_tone(
        fs=FS, freq=-310.0, duration=1.0, amplitude=1.0
    )
    _, weak_positive = generate_iq_tone(
        fs=FS, freq=180.0, duration=1.0, amplitude=0.2
    )
    result_negative = estimate_peak_frequency(
        strong_negative + weak_positive, FS
    )
    assert result_negative.frequency_hz == pytest.approx(-310.0, abs=0.05)


def test_refine_false_returns_exact_fft_bin_center():
    _, samples = generate_iq_tone(fs=FS, freq=137.0, duration=1.0)
    result = estimate_peak_frequency(samples, FS, refine=False)
    assert result.refined is False
    assert result.frequency_hz == result.bin_frequency_hz == 137.0


def test_resolution_is_exactly_fs_over_n():
    _, long_tone = generate_iq_tone(fs=FS, freq=137.0, duration=1.0)
    assert estimate_peak_frequency(long_tone, FS).resolution_hz == FS / N
    _, short_tone = generate_iq_tone(fs=FS, freq=30.0, duration=0.25)
    assert estimate_peak_frequency(short_tone, FS).resolution_hz == FS / 250


def test_dc_dominant_nonconstant_signal_returns_zero():
    _, tone = generate_iq_tone(fs=FS, freq=137.0, duration=1.0, amplitude=0.01)
    samples = np.full(N, 5.0) + np.real(tone)
    result = estimate_peak_frequency(samples, FS)
    assert result.bin_index == 0
    assert result.bin_frequency_hz == 0.0
    assert result.frequency_hz == pytest.approx(0.0, abs=0.05)


def test_constant_and_zero_signals_rejected():
    for bad in (
        np.zeros(16),
        np.zeros(16, dtype=np.complex128),
        np.full(16, 5.0),
        np.full(16, 2.0 + 3.0j),
    ):
        with pytest.raises(ValueError):
            estimate_peak_frequency(bad, FS)


def test_invalid_samples_raise():
    for bad in (
        np.array([]),
        np.ones(1, dtype=np.complex128),
        np.ones(2, dtype=np.complex128),
        np.ones(3, dtype=np.complex128),
        np.ones((4, 4)),
        np.array([1.0, np.nan, 2.0, 3.0]),
        np.array([1.0 + np.inf * 1j, 2.0 + 1j, 3.0, 4.0]),
        np.array(["a", "b", "c", "d"]),
    ):
        with pytest.raises(ValueError):
            estimate_peak_frequency(bad, FS)


@pytest.mark.parametrize(
    "fs",
    [0.0, -1000.0, float("nan"), float("inf"), True, 1.0 + 2.0j],
)
def test_invalid_fs_raises(fs):
    _, samples = generate_iq_tone(fs=FS, freq=100.0, duration=0.1)
    with pytest.raises(ValueError):
        estimate_peak_frequency(samples, fs)


@pytest.mark.parametrize("refine", ["yes", 1, None])
def test_invalid_refine_argument_raises(refine):
    _, samples = generate_iq_tone(fs=FS, freq=100.0, duration=0.1)
    with pytest.raises(ValueError):
        estimate_peak_frequency(samples, FS, refine=refine)


def test_input_is_not_mutated():
    _, samples = generate_iq_tone(fs=FS, freq=137.0, duration=1.0)
    snapshot = samples.copy()
    estimate_peak_frequency(samples, FS, refine=False)
    estimate_peak_frequency(samples, FS, refine=True)
    assert np.array_equal(samples, snapshot)


def test_result_is_frozen_dataclass():
    _, samples = generate_iq_tone(fs=FS, freq=137.0, duration=1.0)
    result = estimate_peak_frequency(samples, FS)
    with pytest.raises(Exception):
        result.frequency_hz = 0.0
