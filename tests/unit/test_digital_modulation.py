"""Unit tests for BPSK/QPSK symbol mapping in iqwav.modulation.digital."""

import numpy as np
import pytest

from iqwav.modulation import bpsk_modulate, qpsk_modulate

MODULATORS = [bpsk_modulate, qpsk_modulate]

SQRT2 = np.sqrt(2.0)


def test_bpsk_exact_mapping():
    symbols = bpsk_modulate([0, 1, 0, 1])
    np.testing.assert_allclose(symbols, [1.0, -1.0, 1.0, -1.0])
    assert symbols.shape == (4,)
    assert symbols.dtype == np.complex128


def test_bpsk_one_symbol_per_bit():
    symbols = bpsk_modulate(np.array([1, 0, 1, 1, 0], dtype=np.int32))
    assert symbols.shape == (5,)


def test_bpsk_unit_magnitude():
    symbols = bpsk_modulate([0, 1, 1, 0, 1])
    np.testing.assert_allclose(np.abs(symbols), 1.0)


def test_qpsk_exact_mapping():
    symbols = qpsk_modulate([0, 0, 0, 1, 1, 1, 1, 0])
    expected = [
        (1.0 + 1.0j) / SQRT2,
        (-1.0 + 1.0j) / SQRT2,
        (-1.0 - 1.0j) / SQRT2,
        (1.0 - 1.0j) / SQRT2,
    ]
    np.testing.assert_allclose(symbols, expected)
    assert symbols.shape == (4,)
    assert symbols.dtype == np.complex128


def test_qpsk_gray_order():
    sequence = qpsk_modulate([0, 0, 0, 1, 1, 1, 1, 0])
    for current, nxt in zip(sequence, np.roll(sequence, -1)):
        np.testing.assert_allclose(np.abs(current - nxt), SQRT2)


def test_qpsk_two_bits_per_symbol():
    symbols = qpsk_modulate([1, 0, 0, 1, 1, 0])
    assert symbols.shape == (3,)


def test_qpsk_unit_magnitude():
    symbols = qpsk_modulate([0, 0, 0, 1, 1, 1, 1, 0])
    np.testing.assert_allclose(np.abs(symbols), 1.0)


def test_qpsk_odd_bit_count_raises():
    with pytest.raises(ValueError):
        qpsk_modulate([0, 1, 1])


@pytest.mark.parametrize("modulator", MODULATORS)
def test_invalid_dimensions_raise(modulator):
    with pytest.raises(ValueError):
        modulator([[0, 1], [1, 0]])


@pytest.mark.parametrize("modulator", MODULATORS)
def test_empty_input_raises(modulator):
    with pytest.raises(ValueError):
        modulator(np.array([], dtype=np.int64))


@pytest.mark.parametrize("modulator", MODULATORS)
@pytest.mark.parametrize(
    "bad_bits",
    [
        np.array([0, 2, 1], dtype=np.int64),
        np.array([0, -1, 1], dtype=np.int64),
        np.array([0.2, 1.0, 0.0]),
        np.array([0.0, 1.0]),
        np.array([0.0, np.nan]),
        np.array([1.0 + 0j, 0.0 + 0j]),
        np.array(["0", "1"], dtype="<U1"),
        [0, "1"],
    ],
)
def test_invalid_bit_values_and_types_raise(modulator, bad_bits):
    with pytest.raises(ValueError):
        modulator(bad_bits)
