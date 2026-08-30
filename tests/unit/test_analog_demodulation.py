"""Unit tests for the FM phase discriminator in iqwav.demod.analog."""

import numpy as np
import pytest

from iqwav.demod import fm_demodulate


def _fm_tone(increment, n=1000, amplitude_fn=None, theta0=0.7):
    n_values = np.arange(n, dtype=np.float64)
    theta = theta0 + increment * n_values
    amplitudes = (
        np.full(n, 1.0) if amplitude_fn is None else amplitude_fn(n_values)
    )
    return amplitudes * np.exp(1j * theta)


def test_output_shape():
    samples = _fm_tone(0.25, n=100)
    output = fm_demodulate(samples)
    assert output.shape == (99,)


def test_output_dtype():
    samples = _fm_tone(0.25)
    assert fm_demodulate(samples).dtype == np.float64


def test_constant_positive_phase_increment():
    samples = _fm_tone(0.25)
    output = fm_demodulate(samples)
    np.testing.assert_allclose(output, 0.25)


def test_constant_negative_phase_increment():
    samples = _fm_tone(-0.25)
    output = fm_demodulate(samples)
    np.testing.assert_allclose(output, -0.25)


def test_phase_wrapping_across_pi():
    samples = _fm_tone(1.5 * np.pi)
    output = fm_demodulate(samples)
    np.testing.assert_allclose(output, -0.5 * np.pi)


def test_amplitude_does_not_change_recovered_increment():
    samples = _fm_tone(0.25, amplitude_fn=lambda n: 1.0 + 0.5 * np.cos(2.0 * np.pi * n / 50.0))
    output = fm_demodulate(samples)
    np.testing.assert_allclose(output, 0.25)


def test_real_valued_input_rejected():
    with pytest.raises(ValueError):
        fm_demodulate(np.cos(np.arange(64) * 0.25))


def test_multidimensional_input_rejected():
    samples = _fm_tone(0.25, n=16).reshape(4, 4)
    with pytest.raises(ValueError):
        fm_demodulate(samples)


@pytest.mark.parametrize("n", [0, 1])
def test_fewer_than_two_samples_rejected(n):
    samples = np.ones(n, dtype=np.complex128)
    with pytest.raises(ValueError):
        fm_demodulate(samples)


def test_nonfinite_samples_rejected():
    samples = _fm_tone(0.25, n=64)
    samples[10] = np.nan + 1j
    with pytest.raises(ValueError):
        fm_demodulate(samples)
    samples[10] = 1.0 + np.inf * 1j
    with pytest.raises(ValueError):
        fm_demodulate(samples)
