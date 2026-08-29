"""Unit tests for power and AWGN utilities in iqwav.dsp.noise."""

import numpy as np
import pytest

from iqwav.dsp import add_awgn, signal_power
from iqwav.modulation import generate_iq_tone, generate_real_tone


def test_signal_power_known_real_signal():
    _, samples = generate_real_tone(fs=1000.0, freq=250.0, duration=1.0)
    power = signal_power(samples)
    assert isinstance(power, float)
    assert power == pytest.approx(0.5)


def test_signal_power_known_iq_tone():
    _, samples = generate_iq_tone(
        fs=1000.0, freq=123.0, duration=1.0, amplitude=1.5
    )
    assert signal_power(samples) == pytest.approx(2.25)


def test_signal_power_of_zero_signal():
    assert signal_power(np.zeros(100)) == 0.0


def test_add_awgn_preserves_shape():
    _, samples = generate_iq_tone(fs=1000.0, freq=123.0, duration=1.0)
    noisy = add_awgn(samples, snr_db=10.0, rng=np.random.default_rng(0))
    assert noisy.shape == samples.shape


def test_real_input_gets_real_noise():
    _, samples = generate_real_tone(fs=1000.0, freq=100.0, duration=1.0)
    noisy = add_awgn(samples, snr_db=10.0, rng=np.random.default_rng(0))
    assert noisy.dtype == np.float64
    assert np.isrealobj(noisy)


def test_complex_input_gets_complex_noise():
    _, samples = generate_iq_tone(fs=1000.0, freq=100.0, duration=1.0)
    noisy = add_awgn(samples, snr_db=10.0, rng=np.random.default_rng(0))
    assert noisy.dtype == np.complex128
    assert np.iscomplexobj(noisy)


def test_add_awgn_deterministic_with_seeded_rng():
    _, samples = generate_iq_tone(
        fs=1000.0, freq=123.0, duration=1.0, amplitude=1.5
    )
    first = add_awgn(samples, snr_db=10.0, rng=np.random.default_rng(42))
    second = add_awgn(samples, snr_db=10.0, rng=np.random.default_rng(42))
    assert np.array_equal(first, second)


@pytest.mark.parametrize("snr_db", [0.0, 10.0, 20.0])
def test_measured_snr_close_to_requested(snr_db):
    _, samples = generate_iq_tone(fs=1000.0, freq=123.0, duration=10.0)
    rng = np.random.default_rng(7)
    noisy = add_awgn(samples, snr_db, rng=rng)
    noise = noisy - samples
    measured_db = 10.0 * np.log10(signal_power(samples) / signal_power(noise))
    assert measured_db == pytest.approx(snr_db, abs=0.5)


def test_measured_snr_close_to_requested_real_signal():
    _, samples = generate_real_tone(fs=1000.0, freq=123.0, duration=10.0)
    rng = np.random.default_rng(7)
    noisy = add_awgn(samples, snr_db=10.0, rng=rng)
    noise = noisy - samples
    measured_db = 10.0 * np.log10(signal_power(samples) / signal_power(noise))
    assert measured_db == pytest.approx(10.0, abs=0.5)


def test_invalid_signal_power_args_raise():
    with pytest.raises(ValueError):
        signal_power(np.ones((4, 4)))
    with pytest.raises(ValueError):
        signal_power(np.array([]))
    with pytest.raises(ValueError):
        signal_power(np.array([1.0, np.nan]))


def test_add_awgn_invalid_args_raise():
    _, samples = generate_iq_tone(fs=1000.0, freq=100.0, duration=0.1)
    for bad_snr in (float("nan"), float("inf"), -float("inf")):
        with pytest.raises(ValueError):
            add_awgn(samples, snr_db=bad_snr)
    for bad_samples in (np.ones((4, 4)), np.array([]), np.array([1.0, np.nan])):
        with pytest.raises(ValueError):
            add_awgn(bad_samples, snr_db=10.0)
    with pytest.raises(ValueError):
        add_awgn(samples, snr_db=10.0, rng="not-a-generator")
