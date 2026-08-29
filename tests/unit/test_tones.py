"""Unit tests for the synthetic tone generators in iqwav.modulation.tones."""

import numpy as np
import pytest

from iqwav.modulation import generate_iq_tone, generate_real_tone

GENERATORS = [generate_real_tone, generate_iq_tone]


@pytest.mark.parametrize("generator", GENERATORS)
@pytest.mark.parametrize(
    ("fs", "duration", "expected_n"),
    [
        (1000.0, 0.1, 100),
        (8000.0, 0.5, 4000),
        (48000.0, 1.0, 48000),
        (44100.0, 0.0015, 66),
    ],
)
def test_sample_count(generator, fs, duration, expected_n):
    time, samples = generator(fs=fs, freq=100.0, duration=duration)
    assert samples.shape == (expected_n,)
    assert time.shape == (expected_n,)


@pytest.mark.parametrize("generator", GENERATORS)
def test_time_array_is_uniform_and_starts_at_zero(generator):
    fs = 1000.0
    time, _ = generator(fs=fs, freq=100.0, duration=0.01)
    assert time.dtype == np.float64
    assert time[0] == 0.0
    np.testing.assert_allclose(np.diff(time), 1.0 / fs)
    np.testing.assert_allclose(time, np.arange(len(time)) / fs)


def test_real_tone_is_real_float64():
    _, samples = generate_real_tone(fs=1000.0, freq=100.0, duration=0.01)
    assert samples.dtype == np.float64
    assert np.isrealobj(samples)


def test_iq_tone_is_complex128():
    _, samples = generate_iq_tone(fs=1000.0, freq=100.0, duration=0.01)
    assert samples.dtype == np.complex128
    assert np.iscomplexobj(samples)


def test_real_tone_peak_amplitude():
    fs = 8.0
    amplitude = 0.75
    _, samples = generate_real_tone(
        fs=fs, freq=fs / 4, duration=1.0, amplitude=amplitude
    )
    np.testing.assert_allclose(np.max(np.abs(samples)), amplitude)


def test_iq_tone_constant_envelope_amplitude():
    amplitude = 0.75
    _, samples = generate_iq_tone(
        fs=1000.0, freq=123.0, duration=0.02, amplitude=amplitude
    )
    np.testing.assert_allclose(np.abs(samples), amplitude)


def test_real_tone_initial_phase():
    fs, freq, amplitude, phase = 8.0, 2.0, 1.5, 0.7
    _, samples = generate_real_tone(
        fs=fs, freq=freq, duration=0.5, amplitude=amplitude, phase=phase
    )
    np.testing.assert_allclose(samples[0], amplitude * np.cos(phase))
    _, quadrature = generate_real_tone(
        fs=fs, freq=freq, duration=0.5, amplitude=amplitude, phase=np.pi / 2
    )
    np.testing.assert_allclose(
        quadrature[:4], [0.0, -amplitude, 0.0, amplitude], atol=1e-12
    )


def test_iq_tone_initial_phase():
    amplitude, phase = 1.5, 0.7
    _, samples = generate_iq_tone(
        fs=1000.0, freq=100.0, duration=0.01, amplitude=amplitude, phase=phase
    )
    np.testing.assert_allclose(samples[0], amplitude * np.exp(1j * phase))
    np.testing.assert_allclose(np.abs(samples[0]), amplitude)


@pytest.mark.parametrize("generator", GENERATORS)
def test_phase_pi_negates_signal(generator):
    kwargs = dict(fs=1000.0, freq=100.0, duration=0.01, amplitude=2.0)
    _, base = generator(**kwargs)
    _, shifted = generator(phase=np.pi, **kwargs)
    np.testing.assert_allclose(shifted, -base, atol=1e-12)


def test_real_tone_known_sample_sequence():
    fs, freq, amplitude = 8.0, 2.0, 1.5
    _, samples = generate_real_tone(
        fs=fs, freq=freq, duration=1.0, amplitude=amplitude
    )
    expected = amplitude * np.array(
        [1.0, 0.0, -1.0, 0.0, 1.0, 0.0, -1.0, 0.0]
    )
    np.testing.assert_allclose(samples, expected, atol=1e-12)


def test_iq_tone_known_sample_sequence():
    fs, freq, amplitude = 8.0, 2.0, 1.5
    _, samples = generate_iq_tone(
        fs=fs, freq=freq, duration=0.5, amplitude=amplitude
    )
    expected = amplitude * np.array(
        [1.0, 1.0j, -1.0, -1.0j], dtype=np.complex128
    )
    np.testing.assert_allclose(samples, expected, atol=1e-12)


def test_real_tone_frequency_matches_fft_bin():
    fs, freq, amplitude, n = 1000.0, 100.0, 0.8, 1000
    _, samples = generate_real_tone(
        fs=fs, freq=freq, duration=1.0, amplitude=amplitude
    )
    spectrum = np.abs(np.fft.fft(samples))
    assert np.argmax(spectrum[: n // 2]) == 100
    np.testing.assert_allclose(spectrum[100], n * amplitude / 2, rtol=1e-6)
    np.testing.assert_allclose(spectrum[n - 100], n * amplitude / 2, rtol=1e-6)


def test_iq_tone_frequency_matches_fft_bin():
    fs, freq, amplitude, n = 1000.0, -100.0, 0.8, 1000
    _, samples = generate_iq_tone(
        fs=fs, freq=freq, duration=1.0, amplitude=amplitude
    )
    spectrum = np.abs(np.fft.fft(samples))
    assert np.argmax(spectrum) == 900
    np.testing.assert_allclose(spectrum[900], n * amplitude, rtol=1e-6)
    np.testing.assert_allclose(spectrum[100], 0.0, atol=1e-9)


def test_iq_negative_frequency_is_conjugate_mirror():
    kwargs = dict(fs=1000.0, duration=0.02, amplitude=1.5, phase=0.0)
    _, positive = generate_iq_tone(freq=100.0, **kwargs)
    _, negative = generate_iq_tone(freq=-100.0, **kwargs)
    np.testing.assert_allclose(negative, np.conj(positive), atol=1e-12)


def test_real_tone_at_nyquist_is_accepted():
    fs = 8.0
    _, samples = generate_real_tone(fs=fs, freq=fs / 2, duration=0.5, amplitude=1.0)
    np.testing.assert_allclose(samples, [1.0, -1.0, 1.0, -1.0], atol=1e-12)


@pytest.mark.parametrize("freq", [4.0, -4.0])
def test_iq_tone_rejects_nyquist(freq):
    with pytest.raises(ValueError):
        generate_iq_tone(fs=8.0, freq=freq, duration=0.5)


@pytest.mark.parametrize("generator", GENERATORS)
def test_zero_amplitude_is_allowed(generator):
    _, samples = generator(fs=1000.0, freq=100.0, duration=0.01, amplitude=0.0)
    assert np.all(samples == 0)


@pytest.mark.parametrize("generator", GENERATORS)
def test_default_amplitude_and_phase(generator):
    _, samples = generator(fs=1000.0, freq=100.0, duration=0.01)
    np.testing.assert_allclose(samples[0], 1.0)


@pytest.mark.parametrize("generator", GENERATORS)
@pytest.mark.parametrize(
    "kwargs",
    [
        dict(fs=0.0, freq=10.0, duration=0.1),
        dict(fs=-1000.0, freq=10.0, duration=0.1),
        dict(fs=1000.0, freq=10.0, duration=0.0),
        dict(fs=1000.0, freq=10.0, duration=-0.5),
        dict(fs=1000.0, freq=10.0, duration=0.1, amplitude=-1.0),
        dict(fs=1000.0, freq=600.0, duration=0.1),
        dict(fs=1000.0, freq=-600.0, duration=0.1),
        dict(fs=1000.0, freq=10.0, duration=0.0004),
        dict(fs=1e308, freq=0.0, duration=1e308),
        dict(fs=float("nan"), freq=10.0, duration=0.1),
        dict(fs=1000.0, freq=10.0, duration=float("inf")),
    ],
)
def test_invalid_arguments_raise_value_error(generator, kwargs):
    with pytest.raises(ValueError):
        generator(**kwargs)
