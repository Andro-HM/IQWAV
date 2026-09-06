"""Unit tests for bounded analog AM/FM/PM modulators."""

import numpy as np
import pytest

from iqwav.demod import fm_demodulate
from iqwav.modulation import am_modulate, fm_modulate, generate_iq_tone, pm_modulate


def _tone_message(fs, freq, n, amplitude=1.0):
    time = np.arange(n, dtype=np.float64) / fs
    return amplitude * np.cos(2.0 * np.pi * freq * time)


def _am(message):
    return am_modulate(message, 0.5)


def _fm(message):
    return fm_modulate(message, 48_000.0, 5_000.0)


def _pm(message):
    return pm_modulate(message, 0.7)


MODULATORS = [_am, _fm, _pm]


def test_am_envelope_equals_one_plus_mu_m():
    message = _tone_message(8_000.0, 200.0, 256, amplitude=0.8)
    mu = 0.6
    samples = am_modulate(message, mu)
    np.testing.assert_allclose(samples.real, 1.0 + mu * message)
    np.testing.assert_allclose(samples.imag, 0.0)
    np.testing.assert_allclose(np.abs(samples), 1.0 + mu * message)


def test_am_modulation_depth_on_plus_minus_one_truth():
    message = np.array([1.0, -1.0, 1.0, -1.0, 1.0])
    mu = 0.35
    envelope = np.abs(am_modulate(message, mu))
    depth = (envelope.max() - envelope.min()) / (envelope.max() + envelope.min())
    np.testing.assert_allclose(depth, mu)


@pytest.mark.parametrize("mu", [0.0, 0.25, 1.0])
def test_am_modulation_depth_matches_index_at_full_scale(mu):
    message = np.array([1.0, -1.0])
    envelope = np.abs(am_modulate(message, mu))
    if mu == 0.0:
        np.testing.assert_allclose(envelope, 1.0)
        return
    depth = (envelope.max() - envelope.min()) / (envelope.max() + envelope.min())
    np.testing.assert_allclose(depth, mu)


def test_am_full_index_can_null_the_envelope():
    samples = am_modulate(np.array([-1.0, 1.0]), 1.0)
    np.testing.assert_allclose(samples[0], 0.0 + 0.0j)
    np.testing.assert_allclose(samples[1], 2.0 + 0.0j)


def test_am_zero_message_is_unit_carrier():
    samples = am_modulate(np.zeros(32), 0.8)
    np.testing.assert_allclose(samples, 1.0 + 0.0j)


def test_am_zero_index_is_unit_carrier():
    message = _tone_message(8_000.0, 150.0, 64)
    samples = am_modulate(message, 0.0)
    np.testing.assert_allclose(samples, 1.0 + 0.0j)


def test_am_output_dtype_shape_and_complexity():
    message = np.array([0.2, -0.4, 0.9])
    samples = am_modulate(message, 0.5)
    assert samples.shape == (3,)
    assert samples.dtype == np.complex128
    assert np.iscomplexobj(samples)


def test_fm_has_constant_unit_magnitude():
    message = _tone_message(48_000.0, 1_000.0, 512, amplitude=0.7)
    samples = fm_modulate(message, 48_000.0, 5_000.0)
    np.testing.assert_allclose(np.abs(samples), 1.0)


def test_fm_unwrapped_increments_recover_instantaneous_deviation():
    fs = 48_000.0
    deviation = 4_000.0
    message = _tone_message(fs, 800.0, 400, amplitude=0.9)
    samples = fm_modulate(message, fs, deviation)
    scale = 2.0 * np.pi * deviation / fs
    unwrapped = np.unwrap(np.angle(samples))
    np.testing.assert_allclose(unwrapped[0], scale * message[0], atol=1e-12)
    np.testing.assert_allclose(np.diff(unwrapped), scale * message[1:], atol=1e-12)


def test_fm_demodulate_recovers_message_from_index_one_at_documented_scale():
    fs = 48_000.0
    deviation = 5_000.0
    message = _tone_message(fs, 1_000.0, 1024, amplitude=0.8)
    samples = fm_modulate(message, fs, deviation)
    demodulated = fm_demodulate(samples)
    expected = 2.0 * np.pi * deviation / fs * message[1:]
    assert demodulated.shape == (message.size - 1,)
    np.testing.assert_allclose(demodulated, expected, atol=1e-12)


def test_fm_constant_message_is_tone_with_cumsum_initial_phase():
    fs = 8_000.0
    deviation = 250.0
    n = 256
    samples = fm_modulate(np.ones(n), fs, deviation)
    _, tone = generate_iq_tone(fs=fs, freq=deviation, duration=n / fs)
    extra_phase = np.exp(1j * 2.0 * np.pi * deviation / fs)
    np.testing.assert_allclose(samples, tone * extra_phase, atol=1e-12)


def test_fm_zero_message_is_unit_phasor():
    samples = fm_modulate(np.zeros(40), 8_000.0, 400.0)
    np.testing.assert_allclose(samples, 1.0 + 0.0j)


def test_fm_zero_deviation_is_unit_phasor():
    message = _tone_message(8_000.0, 200.0, 64)
    samples = fm_modulate(message, 8_000.0, 0.0)
    np.testing.assert_allclose(samples, 1.0 + 0.0j)


def test_fm_negative_message_produces_negative_frequency_increments():
    fs = 8_000.0
    deviation = 300.0
    samples = fm_modulate(-np.ones(32), fs, deviation)
    np.testing.assert_allclose(
        fm_demodulate(samples),
        -2.0 * np.pi * deviation / fs,
    )


def test_fm_output_dtype_shape_and_complexity():
    message = np.array([0.0, 0.5, -0.5])
    samples = fm_modulate(message, 1_000.0, 100.0)
    assert samples.shape == (3,)
    assert samples.dtype == np.complex128
    assert np.iscomplexobj(samples)


def test_pm_has_constant_unit_magnitude():
    message = _tone_message(8_000.0, 250.0, 256, amplitude=1.0)
    samples = pm_modulate(message, 0.9)
    np.testing.assert_allclose(np.abs(samples), 1.0)


def test_pm_unwrapped_phase_equals_beta_m_on_nonwrapping_truth():
    beta = 0.6
    message = _tone_message(8_000.0, 120.0, 200, amplitude=1.0)
    samples = pm_modulate(message, beta)
    np.testing.assert_allclose(np.angle(samples), beta * message, atol=1e-12)
    np.testing.assert_allclose(
        np.unwrap(np.angle(samples)),
        beta * message,
        atol=1e-12,
    )


def test_pm_phase_difference_follows_message_difference_not_message():
    beta = 0.8
    message = _tone_message(8_000.0, 200.0, 128, amplitude=1.0)
    samples = pm_modulate(message, beta)
    phase_diff = np.diff(np.unwrap(np.angle(samples)))
    np.testing.assert_allclose(phase_diff, beta * np.diff(message), atol=1e-12)
    assert np.linalg.norm(phase_diff - beta * message[1:]) > 0.1


def test_pm_constant_message_is_constant_phasor_not_a_tone():
    samples = pm_modulate(np.ones(64), 0.4)
    np.testing.assert_allclose(samples, np.exp(1j * 0.4))
    np.testing.assert_allclose(fm_demodulate(samples), 0.0, atol=1e-12)


def test_phase_difference_distinguishes_pm_contract_from_fm():
    fs = 8_000.0
    message = _tone_message(fs, 150.0, 256, amplitude=0.9)
    fm_samples = fm_modulate(message, fs, 400.0)
    pm_samples = pm_modulate(message, 0.5)
    fm_diff = np.diff(np.unwrap(np.angle(fm_samples)))
    pm_diff = np.diff(np.unwrap(np.angle(pm_samples)))
    fm_scale = 2.0 * np.pi * 400.0 / fs
    np.testing.assert_allclose(fm_diff, fm_scale * message[1:], atol=1e-12)
    np.testing.assert_allclose(pm_diff, 0.5 * np.diff(message), atol=1e-12)
    assert not np.allclose(fm_samples, pm_samples)
    # A single-tone message does not make FM and PM generally identifiable.


def test_pm_zero_message_is_unit_phasor():
    samples = pm_modulate(np.zeros(24), 1.2)
    np.testing.assert_allclose(samples, 1.0 + 0.0j)


def test_pm_zero_deviation_is_unit_phasor():
    message = _tone_message(8_000.0, 90.0, 48)
    samples = pm_modulate(message, 0.0)
    np.testing.assert_allclose(samples, 1.0 + 0.0j)


def test_pm_output_dtype_shape_and_complexity():
    message = np.array([-1.0, 0.0, 0.3])
    samples = pm_modulate(message, 0.25)
    assert samples.shape == (3,)
    assert samples.dtype == np.complex128
    assert np.iscomplexobj(samples)


@pytest.mark.parametrize("modulator", MODULATORS)
def test_list_input_is_accepted(modulator):
    samples = modulator([0.0, 0.5, -0.25, 1.0, -1.0])
    assert samples.shape == (5,)
    assert samples.dtype == np.complex128


@pytest.mark.parametrize("modulator", MODULATORS)
def test_integer_message_is_accepted(modulator):
    samples = modulator(np.array([1, 0, -1], dtype=np.int64))
    assert samples.shape == (3,)


@pytest.mark.parametrize("modulator", MODULATORS)
def test_single_sample_message_is_accepted(modulator):
    samples = modulator(np.array([0.25]))
    assert samples.shape == (1,)


@pytest.mark.parametrize("modulator", MODULATORS)
def test_input_not_mutated(modulator):
    message = np.array([0.1, -0.2, 0.3, -0.4])
    before = message.copy()
    modulator(message)
    np.testing.assert_array_equal(message, before)


@pytest.mark.parametrize("modulator", MODULATORS)
def test_output_is_new_array(modulator):
    message = np.array([0.2, -0.3, 0.4], dtype=np.float64)
    samples = modulator(message)
    assert samples is not message
    assert not np.shares_memory(samples, message)


@pytest.mark.parametrize("modulator", MODULATORS)
def test_output_memory_independence(modulator):
    message = np.array([0.2, -0.3, 0.4], dtype=np.float64)
    before = message.copy()
    samples = modulator(message)
    samples[0] = 9.0 + 8.0j
    np.testing.assert_array_equal(message, before)


@pytest.mark.parametrize("modulator", MODULATORS)
def test_deterministic_output(modulator):
    message = _tone_message(8_000.0, 110.0, 80, amplitude=0.5)
    first = modulator(message)
    second = modulator(message)
    np.testing.assert_array_equal(first, second)


@pytest.mark.parametrize("modulator", MODULATORS)
def test_empty_message_rejected(modulator):
    with pytest.raises(ValueError):
        modulator(np.array([], dtype=np.float64))


@pytest.mark.parametrize("modulator", MODULATORS)
def test_multidimensional_message_rejected(modulator):
    with pytest.raises(ValueError):
        modulator(np.ones((4, 4)))


@pytest.mark.parametrize("modulator", MODULATORS)
def test_complex_message_rejected(modulator):
    with pytest.raises(ValueError):
        modulator(np.array([0.2 + 0.0j, -0.1 + 0.0j]))


@pytest.mark.parametrize("modulator", MODULATORS)
@pytest.mark.parametrize(
    "bad_message",
    [
        np.array([0.0, np.nan, 0.1]),
        np.array([0.0, np.inf]),
        np.array([1.0000001, 0.0]),
        np.array([-1.1, 0.0]),
        np.array(["0.1", "0.2"]),
    ],
)
def test_invalid_message_values_rejected(modulator, bad_message):
    with pytest.raises(ValueError):
        modulator(bad_message)


@pytest.mark.parametrize("mu", [-0.1, 1.1, float("nan"), float("inf")])
def test_invalid_am_modulation_index_rejected(mu):
    with pytest.raises(ValueError):
        am_modulate(np.zeros(8), mu)


@pytest.mark.parametrize("mu", [True, False, "0.5", None, [0.5]])
def test_non_scalar_am_modulation_index_rejected(mu):
    with pytest.raises(ValueError):
        am_modulate(np.zeros(8), mu)


@pytest.mark.parametrize("fs", [0.0, -8_000.0, float("nan"), float("inf")])
def test_invalid_fm_fs_rejected(fs):
    with pytest.raises(ValueError):
        fm_modulate(np.zeros(8), fs, 100.0)


@pytest.mark.parametrize(
    "deviation",
    [-1.0, float("nan"), float("inf"), 4_000.0, 8_000.0],
)
def test_invalid_fm_deviation_rejected(deviation):
    with pytest.raises(ValueError):
        fm_modulate(np.zeros(8), 8_000.0, deviation)


@pytest.mark.parametrize("deviation", [True, False, "100", None, [100.0]])
def test_non_scalar_fm_deviation_rejected(deviation):
    with pytest.raises(ValueError):
        fm_modulate(np.zeros(8), 8_000.0, deviation)


def test_fm_nyquist_deviation_rejected():
    with pytest.raises(ValueError):
        fm_modulate(np.zeros(8), 8_000.0, 4_000.0)


@pytest.mark.parametrize(
    "beta",
    [-0.1, float("nan"), float("inf"), True, False, "0.5", None],
)
def test_invalid_pm_deviation_rejected(beta):
    with pytest.raises(ValueError):
        pm_modulate(np.zeros(8), beta)


def test_am_does_not_embed_cfo_or_static_phase():
    message = _tone_message(8_000.0, 100.0, 64, amplitude=0.5)
    samples = am_modulate(message, 0.4)
    np.testing.assert_allclose(samples.imag, 0.0)
    # A static carrier phase or CFO would rotate or spin this real envelope.


def test_fm_and_pm_do_not_embed_amplitude_scaling():
    message = _tone_message(8_000.0, 100.0, 64, amplitude=0.5)
    np.testing.assert_allclose(np.abs(fm_modulate(message, 8_000.0, 200.0)), 1.0)
    np.testing.assert_allclose(np.abs(pm_modulate(message, 0.3)), 1.0)
