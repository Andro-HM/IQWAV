"""Unit tests for cumulative-power occupied-bandwidth measurement."""

import numpy as np
import pytest

from iqwav.estimation import (
    OccupiedBandwidthEstimate,
    estimate_occupied_bandwidth,
)
from iqwav.modulation import generate_iq_tone, generate_real_tone

FS = 1000.0
N = 1000


def _iq_tone(freq, amplitude=1.0):
    _, samples = generate_iq_tone(
        fs=FS, freq=freq, duration=N / FS, amplitude=amplitude
    )
    return samples


def _real_tone(freq, amplitude=1.0):
    _, samples = generate_real_tone(
        fs=FS, freq=freq, duration=N / FS, amplitude=amplitude
    )
    return samples


def test_complex_exact_bin_tone_near_one_bin():
    result = estimate_occupied_bandwidth(_iq_tone(137.0), FS, power_fraction=0.99)
    assert isinstance(result, OccupiedBandwidthEstimate)
    assert result.lower_hz == 136.5
    assert result.upper_hz == 137.5
    assert result.center_hz == 137.0
    assert result.bandwidth_hz == 1.0
    assert result.wraps_nyquist is False
    assert result.requested_power_fraction == 0.99
    assert result.achieved_power_fraction >= 0.99


def test_complex_two_tones_span_both():
    signal = _iq_tone(100.0) + _iq_tone(200.0)
    result = estimate_occupied_bandwidth(signal, FS, power_fraction=0.99)
    assert result.lower_hz == 99.5
    assert result.upper_hz == 200.5
    assert result.bandwidth_hz == 101.0
    assert result.center_hz == 150.0


def test_increasing_power_fraction_never_shrinks_bandwidth():
    signal = (
        _iq_tone(100.0, 1.0) + _iq_tone(150.0, 0.5) + _iq_tone(200.0, 0.25)
    )
    bandwidths = [
        estimate_occupied_bandwidth(signal, FS, power_fraction=fraction).bandwidth_hz
        for fraction in (0.5, 0.8, 0.99)
    ]
    assert bandwidths == [1.0, 51.0, 101.0]


def test_amplitude_scaling_invariance():
    base = estimate_occupied_bandwidth(_iq_tone(137.0), FS)
    scaled = estimate_occupied_bandwidth(_iq_tone(137.0, amplitude=5.0), FS)
    assert scaled.lower_hz == base.lower_hz
    assert scaled.bandwidth_hz == base.bandwidth_hz


def test_phase_rotation_invariance():
    base = estimate_occupied_bandwidth(_iq_tone(137.0), FS)
    rotated = estimate_occupied_bandwidth(
        _iq_tone(137.0) * np.exp(1j * 0.7), FS
    )
    assert rotated.lower_hz == base.lower_hz
    assert rotated.bandwidth_hz == base.bandwidth_hz


def test_real_tone_nonnegative_interval():
    result = estimate_occupied_bandwidth(
        _real_tone(137.0), FS, power_fraction=0.99
    )
    assert result.lower_hz == 136.5
    assert result.upper_hz == 137.5
    assert result.bandwidth_hz == 1.0
    assert 0.0 <= result.lower_hz <= result.upper_hz <= FS / 2.0
    assert result.wraps_nyquist is False


def test_real_input_folds_conjugate_power():
    signal = _real_tone(100.0) + _real_tone(200.0)
    result = estimate_occupied_bandwidth(signal, FS, power_fraction=0.99)
    assert result.lower_hz == 99.5
    assert result.upper_hz == 200.5
    assert result.bandwidth_hz == 101.0


def test_real_even_length_nyquist_counted_once():
    samples = np.array([1.0 if i % 2 == 0 else -1.0 for i in range(N)])
    result = estimate_occupied_bandwidth(samples, FS, power_fraction=0.99)
    assert result.upper_hz == FS / 2.0
    assert result.lower_hz == FS / 2.0 - 0.5
    assert result.bandwidth_hz == 0.5
    assert result.achieved_power_fraction >= 0.99


def test_complex_nyquist_wrap_selects_narrow_wrapped_interval():
    signal = _iq_tone(496.0) + _iq_tone(-494.0)
    result = estimate_occupied_bandwidth(signal, FS, power_fraction=0.99)
    assert result.wraps_nyquist is True
    assert result.lower_hz > result.upper_hz
    assert result.lower_hz == 495.5
    assert result.upper_hz == -493.5
    assert result.bandwidth_hz == 11.0
    assert result.bandwidth_hz < 0.1 * FS
    assert result.center_hz == -499.0
    assert result.achieved_power_fraction >= 0.99


def test_non_wrapped_complex_tones_flag_false():
    positive = estimate_occupied_bandwidth(_iq_tone(137.0), FS)
    assert positive.wraps_nyquist is False
    assert positive.lower_hz < positive.upper_hz
    negative = estimate_occupied_bandwidth(
        _iq_tone(-137.0), FS, power_fraction=0.99
    )
    assert negative.wraps_nyquist is False
    assert negative.lower_hz == -137.5
    assert negative.upper_hz == -136.5


def test_achieved_fraction_meets_requested():
    signal = (
        _iq_tone(100.0, 1.0) + _iq_tone(150.0, 0.5) + _iq_tone(200.0, 0.25)
    )
    for fraction in (0.5, 0.8, 0.99, 1.0):
        result = estimate_occupied_bandwidth(signal, FS, power_fraction=fraction)
        assert result.achieved_power_fraction >= fraction - 1e-9
        assert result.achieved_power_fraction <= 1.0 + 1e-9


def test_power_fraction_one_behavior():
    clean = estimate_occupied_bandwidth(_iq_tone(137.0), FS, power_fraction=1.0)
    assert clean.bandwidth_hz == 1.0
    assert clean.achieved_power_fraction >= 1.0 - 1e-9
    rng = np.random.default_rng(5)
    noise = rng.standard_normal(N) + 1j * np.random.default_rng(6).standard_normal(N)
    spread = estimate_occupied_bandwidth(noise, FS, power_fraction=1.0)
    assert spread.lower_hz == -FS / 2.0
    assert spread.upper_hz == FS / 2.0
    assert spread.bandwidth_hz == FS
    assert spread.wraps_nyquist is False
    assert spread.achieved_power_fraction >= 1.0 - 1e-9


@pytest.mark.parametrize(
    "power_fraction",
    [0.0, -0.5, 1.5, float("nan"), float("inf"), True, False],
)
def test_invalid_power_fraction_raises(power_fraction):
    with pytest.raises(ValueError):
        estimate_occupied_bandwidth(_iq_tone(100.0), FS, power_fraction=power_fraction)


@pytest.mark.parametrize(
    "fs",
    [0.0, -1000.0, float("nan"), float("inf"), True, 1.0 + 2.0j],
)
def test_invalid_fs_raises(fs):
    with pytest.raises(ValueError):
        estimate_occupied_bandwidth(_iq_tone(100.0), fs)


def test_invalid_samples_raise():
    for bad in (
        np.array([]),
        np.ones(3, dtype=np.complex128),
        np.ones((4, 4)),
        np.array([1.0, np.nan, 2.0, 3.0]),
        np.array([1.0 + np.inf * 1j, 2.0 + 1j, 3.0, 4.0]),
        np.array(["a", "b", "c", "d"]),
    ):
        with pytest.raises(ValueError):
            estimate_occupied_bandwidth(bad, FS)


def test_zero_and_constant_signals_rejected():
    for bad in (
        np.zeros(16),
        np.zeros(16, dtype=np.complex128),
        np.full(16, 5.0),
        np.full(16, 2.0 + 3.0j),
    ):
        with pytest.raises(ValueError):
            estimate_occupied_bandwidth(bad, FS)


def test_input_not_mutated():
    samples = _iq_tone(137.0)
    snapshot = samples.copy()
    estimate_occupied_bandwidth(samples, FS)
    estimate_occupied_bandwidth(samples, FS, power_fraction=0.5)
    assert np.array_equal(samples, snapshot)


def test_result_is_frozen_dataclass():
    result = estimate_occupied_bandwidth(_iq_tone(137.0), FS)
    with pytest.raises(Exception):
        result.bandwidth_hz = 0.0
