"""Unit tests for known-timing BPSK/QPSK demodulation in iqwav.demod.digital."""

import numpy as np
import pytest

from iqwav.demod import bpsk_demodulate, qpsk_demodulate
from iqwav.dsp import add_awgn
from iqwav.modulation import bpsk_waveform, qpsk_waveform

DEMODULATORS = [bpsk_demodulate, qpsk_demodulate]

SQRT2 = np.sqrt(2.0)


def test_bpsk_exact_recovery_round_trip():
    bits = np.array([0, 1, 1, 0, 1, 0, 0, 1], dtype=np.int64)
    waveform = bpsk_waveform(bits, 4)
    recovered = bpsk_demodulate(waveform, 4)
    np.testing.assert_array_equal(recovered, bits)
    assert np.issubdtype(recovered.dtype, np.integer)


def test_bpsk_samples_per_symbol_one():
    bits = np.array([1, 0, 1], dtype=np.int64)
    recovered = bpsk_demodulate(bpsk_waveform(bits, 1), 1)
    np.testing.assert_array_equal(recovered, bits)


def test_bpsk_real_input_supported():
    bits = np.array([0, 1, 0, 0, 1], dtype=np.int64)
    waveform = np.real(bpsk_waveform(bits, 3))
    assert np.isrealobj(waveform)
    recovered = bpsk_demodulate(waveform, 3)
    np.testing.assert_array_equal(recovered, bits)


def test_bpsk_output_bit_count():
    recovered = bpsk_demodulate(np.ones(20, dtype=np.complex128), 4)
    assert recovered.shape == (5,)


def test_qpsk_all_four_gray_pairs_recovered():
    samples = np.array(
        [
            (1.0 + 1.0j) / SQRT2,
            (-1.0 + 1.0j) / SQRT2,
            (-1.0 - 1.0j) / SQRT2,
            (1.0 - 1.0j) / SQRT2,
        ]
    )
    recovered = qpsk_demodulate(samples, 1)
    np.testing.assert_array_equal(recovered, [0, 0, 0, 1, 1, 1, 1, 0])


def test_qpsk_exact_recovery_round_trip():
    bits = np.array([0, 1, 1, 1, 0, 0, 1, 0], dtype=np.int64)
    waveform = qpsk_waveform(bits, 5)
    recovered = qpsk_demodulate(waveform, 5)
    np.testing.assert_array_equal(recovered, bits)
    assert np.issubdtype(recovered.dtype, np.integer)


def test_qpsk_samples_per_symbol_one():
    bits = np.array([0, 0, 0, 1, 1, 1, 1, 0], dtype=np.int64)
    recovered = qpsk_demodulate(qpsk_waveform(bits, 1), 1)
    np.testing.assert_array_equal(recovered, bits)


def test_qpsk_output_bit_count():
    recovered = qpsk_demodulate(np.ones(24, dtype=np.complex128), 4)
    assert recovered.shape == (12,)


def test_qpsk_real_input_raises():
    with pytest.raises(ValueError):
        qpsk_demodulate(np.ones(8), 4)


def test_bpsk_recovery_after_moderate_awgn():
    rng = np.random.default_rng(123)
    bits = np.array([0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0], dtype=np.int64)
    waveform = bpsk_waveform(bits, 20)
    noisy = add_awgn(waveform, snr_db=6.0, rng=rng)
    recovered = bpsk_demodulate(noisy, 20)
    np.testing.assert_array_equal(recovered, bits)


def test_qpsk_recovery_after_moderate_awgn():
    rng = np.random.default_rng(321)
    bits = (np.arange(24) % 3 % 2).astype(np.int64)
    waveform = qpsk_waveform(bits, 20)
    noisy = add_awgn(waveform, snr_db=6.0, rng=rng)
    recovered = qpsk_demodulate(noisy, 20)
    np.testing.assert_array_equal(recovered, bits)


@pytest.mark.parametrize("demodulator", DEMODULATORS)
def test_invalid_dimensions_raise(demodulator):
    with pytest.raises(ValueError):
        demodulator(np.ones((4, 4), dtype=np.complex128), 2)


@pytest.mark.parametrize("demodulator", DEMODULATORS)
def test_empty_samples_raise(demodulator):
    with pytest.raises(ValueError):
        demodulator(np.array([], dtype=np.complex128), 2)


@pytest.mark.parametrize("demodulator", DEMODULATORS)
def test_nonfinite_samples_raise(demodulator):
    with pytest.raises(ValueError):
        demodulator(np.array([1.0, np.nan] * 3, dtype=np.complex128), 2)


@pytest.mark.parametrize("demodulator", DEMODULATORS)
@pytest.mark.parametrize("samples_per_symbol", [0, -1, 2.5, "3"])
def test_invalid_samples_per_symbol_raises(demodulator, samples_per_symbol):
    with pytest.raises(ValueError):
        demodulator(np.ones(6, dtype=np.complex128), samples_per_symbol)


@pytest.mark.parametrize("demodulator", DEMODULATORS)
def test_non_divisible_length_raises(demodulator):
    with pytest.raises(ValueError):
        demodulator(np.ones(7, dtype=np.complex128), 3)
