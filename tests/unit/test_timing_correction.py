"""Unit tests for caller-supplied integer symbol-timing correction."""

import numpy as np
import pytest

from iqwav.demod import bpsk_demodulate, qpsk_demodulate
from iqwav.dsp import add_awgn, apply_phase_offset
from iqwav.estimation import estimate_symbol_timing
from iqwav.modulation import bpsk_waveform, qpsk_waveform
from iqwav.synchronization import correct_symbol_timing

N_SYMBOLS = 256


def _bits(seed, n):
    return np.random.default_rng(seed).integers(0, 2, n, dtype=np.int64)


def _make(modulation, seed, sps, n_symbols=N_SYMBOLS):
    if modulation == "bpsk":
        bits = _bits(seed, n_symbols)
        return bpsk_waveform(bits, sps), bits
    bits = _bits(seed, 2 * n_symbols)
    return qpsk_waveform(bits, sps), bits


def _oracle_slice(clean, crop, sps):
    offset = (-crop) % sps
    start = crop + offset
    n_complete = (clean.size - start) // sps
    return clean[start : start + n_complete * sps], offset, start, n_complete


def _remaining_bits(bits, modulation, start, n_complete, sps):
    start_sym = start // sps
    if modulation == "bpsk":
        return bits[start_sym : start_sym + n_complete]
    return bits[2 * start_sym : 2 * (start_sym + n_complete)]


def _demod_errors(samples, true_bits, modulation, sps):
    if modulation == "qpsk":
        n_sym = true_bits.size // 2
        true_pairs = true_bits.reshape(n_sym, 2)
        best_bit = true_bits.size
        best_sym = n_sym
        for k in range(4):
            recovered = qpsk_demodulate(samples * (1j**k), sps)
            bit_err = int(np.count_nonzero(recovered != true_bits))
            rec_pairs = recovered.reshape(n_sym, 2)
            sym_err = int(np.count_nonzero(np.any(rec_pairs != true_pairs, axis=1)))
            if bit_err < best_bit:
                best_bit = bit_err
                best_sym = sym_err
        return best_bit, best_sym, true_bits.size, n_sym
    best = true_bits.size
    for sign in (1.0, -1.0):
        recovered = bpsk_demodulate(samples * sign, sps)
        err = int(np.count_nonzero(recovered != true_bits))
        if err < best:
            best = err
    return best, best, true_bits.size, true_bits.size


@pytest.mark.parametrize("modulation", ["bpsk", "qpsk"])
@pytest.mark.parametrize("sps", [2, 4, 8, 16])
def test_true_offset_matches_clean_slice(modulation, sps):
    clean, _bits_true = _make(modulation, 51, sps)
    for crop in range(sps):
        received = clean[crop:]
        truth, offset, _start, _n = _oracle_slice(clean, crop, sps)
        corrected = correct_symbol_timing(received, sps, offset)
        np.testing.assert_array_equal(corrected, truth)
        assert corrected.size % sps == 0
        assert corrected is not received
        assert not np.shares_memory(corrected, received)


@pytest.mark.parametrize("modulation", ["bpsk", "qpsk"])
@pytest.mark.parametrize("sps", [2, 4, 8, 16])
def test_estimate_then_correct_equals_clean_truth(modulation, sps):
    """Independent check: aligned output equals the known clean slice.
    Does not rerun the estimator after correction."""
    clean, _bits_true = _make(modulation, 52, sps)
    for crop in range(sps):
        received = clean[crop:]
        estimate = estimate_symbol_timing(received, sps)
        corrected = correct_symbol_timing(
            received, sps, estimate.boundary_offset
        )
        truth, _offset, _start, _n = _oracle_slice(clean, crop, sps)
        np.testing.assert_array_equal(corrected, truth)


def test_zero_offset_trims_trailing_incomplete_symbol():
    clean, _bits_true = _make("bpsk", 53, 8)
    extra = np.concatenate([clean, clean[:3]])
    corrected = correct_symbol_timing(extra, 8, 0)
    np.testing.assert_array_equal(corrected, clean)
    assert corrected.size == clean.size
    assert corrected is not extra


def test_drops_leading_partial_symbol_only():
    clean = bpsk_waveform(_bits(54, 16), 8)
    received = clean[3:]
    corrected = correct_symbol_timing(received, 8, 5)
    np.testing.assert_array_equal(corrected, clean[8:])


def test_wrong_offset_does_not_match_truth():
    clean, _bits_true = _make("qpsk", 55, 8)
    received = clean[3:]
    truth, true_offset, _start, _n = _oracle_slice(clean, 3, 8)
    assert true_offset != 0
    wrong = correct_symbol_timing(received, 8, 0)
    assert wrong.shape != truth.shape or not np.array_equal(wrong, truth)


def test_input_not_mutated_and_new_array():
    samples = _make("bpsk", 56, 8)[0]
    before = samples.copy()
    corrected = correct_symbol_timing(samples, 8, 0)
    np.testing.assert_array_equal(samples, before)
    assert corrected is not samples
    assert not np.shares_memory(corrected, samples)
    corrected[0] = 99.0 + 0.0j
    np.testing.assert_array_equal(samples, before)


def test_complex64_dtype_preserved():
    samples = np.ones(32, dtype=np.complex64)
    samples[8:] = -1.0
    corrected = correct_symbol_timing(samples, 8, 0)
    assert corrected.dtype == np.complex64
    assert corrected.size == 32


def test_real_input_accepted():
    samples = np.ones(24, dtype=np.float64)
    samples[8:] = -1.0
    corrected = correct_symbol_timing(samples, 8, 0)
    assert corrected.dtype == np.float64
    np.testing.assert_array_equal(corrected, samples)


@pytest.mark.parametrize("modulation", ["bpsk", "qpsk"])
@pytest.mark.parametrize("sps", [4, 8])
def test_downstream_demod_clean(modulation, sps):
    clean, bits = _make(modulation, 61, sps)
    for crop in range(sps):
        received = clean[crop:]
        estimate = estimate_symbol_timing(received, sps)
        aligned = correct_symbol_timing(
            received, sps, estimate.boundary_offset
        )
        _truth, _offset, start, n_complete = _oracle_slice(clean, crop, sps)
        remaining = _remaining_bits(bits, modulation, start, n_complete, sps)
        bit_err, _sym_err, n_bits, _n_sym = _demod_errors(
            aligned, remaining, modulation, sps
        )
        assert bit_err == 0
        assert n_bits == remaining.size


@pytest.mark.parametrize("modulation", ["bpsk", "qpsk"])
def test_downstream_demod_20db(modulation):
    sps = 8
    clean, bits = _make(modulation, 62, sps)
    for crop in (0, 3, 7):
        noisy = add_awgn(clean, 20.0, rng=np.random.default_rng(90000 + crop))
        received = noisy[crop:]
        estimate = estimate_symbol_timing(received, sps)
        assert estimate.boundary_offset == (-crop) % sps
        aligned = correct_symbol_timing(
            received, sps, estimate.boundary_offset
        )
        _truth, _offset, start, n_complete = _oracle_slice(clean, crop, sps)
        remaining = _remaining_bits(bits, modulation, start, n_complete, sps)
        bit_err, _sym_err, _n_bits, _n_sym = _demod_errors(
            aligned, remaining, modulation, sps
        )
        assert bit_err == 0


def test_phase_then_timing_then_demod():
    sps = 8
    clean, bits = _make("qpsk", 63, sps)
    crop = 5
    received = apply_phase_offset(clean, 0.8)[crop:]
    estimate = estimate_symbol_timing(received, sps)
    aligned = correct_symbol_timing(received, sps, estimate.boundary_offset)
    _truth, _offset, start, n_complete = _oracle_slice(clean, crop, sps)
    remaining = _remaining_bits(bits, "qpsk", start, n_complete, sps)
    bit_err, _sym_err, _n_bits, _n_sym = _demod_errors(
        aligned, remaining, "qpsk", sps
    )
    assert estimate.boundary_offset == (-crop) % sps
    assert bit_err == 0


@pytest.mark.parametrize(
    ("n", "sps", "offset"),
    [
        (10, 8, 7),
        (8, 8, 1),
        (7, 8, 0),
        (5, 8, 5),
        (2, 2, 1),
    ],
)
def test_fewer_than_one_complete_symbol_raises(n, sps, offset):
    """Incomplete leftover after the leading drop is an error, not an
    empty array. Existing demodulators require a positive multiple of
    SPS, so returning [] would be an accidental empty result."""
    samples = np.ones(n, dtype=np.complex128)
    remaining = n - offset
    assert remaining < sps
    with pytest.raises(ValueError, match="no complete symbol"):
        correct_symbol_timing(samples, sps, offset)


def test_exactly_one_complete_symbol_succeeds():
    samples = np.arange(11, dtype=np.complex128)
    corrected = correct_symbol_timing(samples, 8, 3)
    np.testing.assert_array_equal(corrected, samples[3:11])
    assert corrected.size == 8
    assert corrected.size % 8 == 0
    assert corrected is not samples


@pytest.mark.parametrize(
    "samples_per_symbol",
    [1, 0, -1, True, False, 2.5, "8"],
)
def test_invalid_sps_raises(samples_per_symbol):
    with pytest.raises(ValueError):
        correct_symbol_timing(np.ones(32, dtype=np.complex128), samples_per_symbol, 0)


@pytest.mark.parametrize("offset", [-1, 8, 9, True, False, 1.5, "0"])
def test_invalid_offset_raises(offset):
    with pytest.raises(ValueError):
        correct_symbol_timing(np.ones(32, dtype=np.complex128), 8, offset)


def test_invalid_samples_raise():
    for bad in (
        np.array([]),
        np.ones((4, 4), dtype=np.complex128),
        np.array([1.0, np.nan, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
        np.array(["a", "b", "c", "d"]),
    ):
        with pytest.raises(ValueError):
            correct_symbol_timing(bad, 2, 0)


def test_deterministic_output():
    samples = _make("bpsk", 71, 8)[0][3:]
    first = correct_symbol_timing(samples, 8, 5)
    second = correct_symbol_timing(samples, 8, 5)
    np.testing.assert_array_equal(first, second)
