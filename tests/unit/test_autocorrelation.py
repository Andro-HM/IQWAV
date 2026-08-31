"""Unit tests for autocorrelation primitives in iqwav.correlation."""

import numpy as np
import pytest

from iqwav.correlation import autocorrelation, normalized_autocorrelation
from iqwav.modulation import generate_iq_tone

CORRELATORS = [autocorrelation, normalized_autocorrelation]


def test_lag0_value_and_extreme_overlap_normalization():
    ac = autocorrelation(np.array([1.0, 2.0, 3.0, 4.0]))
    np.testing.assert_allclose(ac[0], 7.5)
    np.testing.assert_allclose(ac[3], 4.0)
    assert ac.shape == (4,)


def test_constant_signal_all_lags_equal_one():
    ac = autocorrelation(np.ones(100))
    np.testing.assert_allclose(ac, np.ones(100))


def test_known_repeated_sequence_period_peak():
    x = np.array([1.0, 2.0, 3.0, 1.0, 2.0, 3.0])
    ac = autocorrelation(x)
    np.testing.assert_allclose(ac[3], 14.0 / 3.0)
    assert ac[3] == ac[1:].max()
    assert np.argmax(ac[1:]) + 1 == 3


def test_complex_conjugation_convention():
    x = np.array([1.0 + 2.0j, 3.0 - 1.0j, 2.0 + 0.0j])
    ac = autocorrelation(x, max_lag=2)
    np.testing.assert_allclose(ac[0], 19.0 / 3.0)
    np.testing.assert_allclose(ac[1], 3.5 - 2.5j)
    np.testing.assert_allclose(ac[2], 2.0 - 4.0j)
    assert ac.dtype == np.complex128


def test_iq_tone_lag1_carries_phase_increment():
    _, tone = generate_iq_tone(fs=1000.0, freq=100.0, duration=0.05)
    ac = autocorrelation(tone, max_lag=1)
    np.testing.assert_allclose(np.abs(ac[1]), 1.0)
    np.testing.assert_allclose(np.angle(ac[1]), 2.0 * np.pi * 100.0 / 1000.0)


def test_normalized_lag0_is_one():
    _, tone = generate_iq_tone(fs=1000.0, freq=100.0, duration=0.05)
    norm = normalized_autocorrelation(tone, max_lag=10)
    np.testing.assert_allclose(norm[0], 1.0)
    assert norm.shape == (11,)


@pytest.mark.parametrize(
    ("max_lag", "expected_length"),
    [(None, 100), (0, 1), (10, 11)],
)
def test_max_lag_controls_output_length(max_lag, expected_length):
    x = np.ones(100)
    assert autocorrelation(x, max_lag=max_lag).shape == (expected_length,)


def test_real_input_real_output():
    ac = autocorrelation(np.array([1.0, -2.0, 3.0]))
    assert ac.dtype == np.float64
    assert np.isrealobj(ac)


def test_complex_input_complex_output():
    ac = autocorrelation(np.array([1.0 + 1.0j, 2.0 - 1.0j]))
    assert ac.dtype == np.complex128
    assert np.iscomplexobj(ac)


def test_zero_energy_input():
    zeros = np.zeros(8)
    raw = autocorrelation(zeros)
    np.testing.assert_array_equal(raw, np.zeros(8))
    with pytest.raises(ValueError):
        normalized_autocorrelation(zeros)


def test_white_noise_real_normalized_autocorrelation_small():
    rng = np.random.default_rng(12345)
    noise = rng.standard_normal(10000)
    norm = normalized_autocorrelation(noise, max_lag=100)
    np.testing.assert_allclose(norm[0], 1.0)
    assert float(np.max(np.abs(norm[1:]))) < 0.05


def test_white_noise_complex_normalized_autocorrelation_small():
    rng = np.random.default_rng(67890)
    noise = rng.standard_normal(10000) + 1j * rng.standard_normal(10000)
    norm = normalized_autocorrelation(noise, max_lag=100)
    np.testing.assert_allclose(np.abs(norm[0]), 1.0)
    assert float(np.max(np.abs(norm[1:]))) < 0.05


@pytest.mark.parametrize("function", CORRELATORS)
def test_invalid_samples_raise(function):
    for bad in (
        np.array([]),
        np.ones((4, 4)),
        np.array([1.0, np.nan]),
        np.array([1.0 + np.inf * 1j, 2.0]),
        np.array(["a", "b"]),
    ):
        with pytest.raises(ValueError):
            function(bad)


@pytest.mark.parametrize("function", CORRELATORS)
@pytest.mark.parametrize(
    "max_lag",
    [-1, 100, 101, 1.5, "3", True, False],
)
def test_invalid_max_lag_raises(function, max_lag):
    x = np.ones(100)
    with pytest.raises(ValueError):
        function(x, max_lag=max_lag)
