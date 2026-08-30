"""Unit tests for rectangular-pulse waveform utilities in iqwav.modulation."""

import numpy as np
import pytest

from iqwav.modulation import bpsk_waveform, qpsk_waveform, symbols_to_samples


def test_symbols_to_samples_exact_repetition():
    output = symbols_to_samples(np.array([1.0, -1.0]), 3)
    np.testing.assert_allclose(output, [1.0, 1.0, 1.0, -1.0, -1.0, -1.0])


@pytest.mark.parametrize(
    ("n_symbols", "samples_per_symbol"),
    [(1, 1), (5, 2), (16, 7)],
)
def test_symbols_to_samples_output_length(n_symbols, samples_per_symbol):
    symbols = np.ones(n_symbols, dtype=np.complex128)
    output = symbols_to_samples(symbols, samples_per_symbol)
    assert output.shape == (n_symbols * samples_per_symbol,)


def test_symbols_to_samples_preserves_real():
    output = symbols_to_samples(np.array([1.0, -1.0]), 3)
    assert output.dtype == np.float64
    assert np.isrealobj(output)


def test_symbols_to_samples_preserves_complex():
    output = symbols_to_samples(np.array([1.0 + 1.0j, -1.0 + 1.0j]), 3)
    assert output.dtype == np.complex128
    assert np.iscomplexobj(output)


def test_symbols_to_samples_sps_one_unchanged():
    symbols = np.array([1.0 + 2.0j, -3.0 - 4.0j, 0.5 + 0.0j])
    output = symbols_to_samples(symbols, 1)
    np.testing.assert_allclose(output, symbols)


@pytest.mark.parametrize(
    "bad_symbols",
    [
        np.ones((4, 4)),
        np.array([]),
        np.array([1.0, np.nan]),
        np.array([1.0 + np.inf * 1j, 2.0]),
    ],
)
def test_invalid_symbols_raise(bad_symbols):
    with pytest.raises(ValueError):
        symbols_to_samples(bad_symbols, 3)


@pytest.mark.parametrize("samples_per_symbol", [0, -2, 2.5, "4", None])
def test_invalid_samples_per_symbol_raises(samples_per_symbol):
    with pytest.raises(ValueError):
        symbols_to_samples(np.array([1.0, -1.0]), samples_per_symbol)


def test_bpsk_waveform_exact():
    output = bpsk_waveform([0, 1, 0], 3)
    np.testing.assert_allclose(
        output, [1.0, 1.0, 1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0]
    )
    assert output.dtype == np.complex128


def test_bpsk_waveform_length():
    output = bpsk_waveform([1, 0, 1, 1], 5)
    assert output.shape == (4 * 5,)


def test_bpsk_waveform_invalid_inputs_raise():
    with pytest.raises(ValueError):
        bpsk_waveform([0, 2], 3)
    with pytest.raises(ValueError):
        bpsk_waveform([0, 1], 0)


def test_qpsk_waveform_exact_repetition():
    scale = 1.0 / np.sqrt(2.0)
    output = qpsk_waveform([0, 0, 0, 1, 1, 1, 1, 0], 2)
    expected = np.array(
        [
            (1.0 + 1.0j) * scale,
            (1.0 + 1.0j) * scale,
            (-1.0 + 1.0j) * scale,
            (-1.0 + 1.0j) * scale,
            (-1.0 - 1.0j) * scale,
            (-1.0 - 1.0j) * scale,
            (1.0 - 1.0j) * scale,
            (1.0 - 1.0j) * scale,
        ]
    )
    np.testing.assert_allclose(output, expected)
    assert output.dtype == np.complex128


def test_qpsk_waveform_length():
    output = qpsk_waveform([1, 0, 0, 1, 1, 0], 4)
    assert output.shape == (3 * 4,)


def test_qpsk_waveform_invalid_inputs_raise():
    with pytest.raises(ValueError):
        qpsk_waveform([0, 1, 1], 2)
    with pytest.raises(ValueError):
        qpsk_waveform([0, 0, 2, 2], 2)
    with pytest.raises(ValueError):
        qpsk_waveform([0, 1], 0)
