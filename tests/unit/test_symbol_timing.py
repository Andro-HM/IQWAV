"""Unit tests for known-SPS integer timing estimation."""

import inspect

import numpy as np
import pytest

from iqwav.dsp import add_awgn, apply_phase_offset
from iqwav.estimation import (
    SymbolTimingEstimate,
    estimate_rectangular_symbol_grid,
    estimate_symbol_timing,
)
from iqwav.modulation import bpsk_waveform, qpsk_waveform

FS = 80_000.0
N_SYMBOLS = 256


def _bits(seed, n):
    return np.random.default_rng(seed).integers(0, 2, n, dtype=np.int64)


def _waveform(modulation, seed, sps, n_symbols=N_SYMBOLS):
    if modulation == "bpsk":
        return bpsk_waveform(_bits(seed, n_symbols), sps)
    return qpsk_waveform(_bits(seed, 2 * n_symbols), sps)


def _one_transition(modulation, sps, n_symbols=64):
    if modulation == "bpsk":
        bits = np.zeros(n_symbols, dtype=np.int64)
        bits[n_symbols // 2 :] = 1
        return bpsk_waveform(bits, sps)
    bits = np.zeros(2 * n_symbols, dtype=np.int64)
    bits[2 * (n_symbols // 2) :] = 1
    return qpsk_waveform(bits, sps)


@pytest.mark.parametrize("modulation", ["bpsk", "qpsk"])
@pytest.mark.parametrize("sps", [2, 4, 8, 16])
def test_clean_all_residues(modulation, sps):
    waveform = _waveform(modulation, 70000 + sps, sps)
    for crop in range(sps):
        estimate = estimate_symbol_timing(waveform[crop:], sps)
        assert isinstance(estimate, SymbolTimingEstimate)
        assert estimate.boundary_offset == (-crop) % sps
        assert 0 <= estimate.boundary_offset < sps
        assert estimate.quality == pytest.approx(1.0)


def test_uncropped_offset_is_zero():
    waveform = _waveform("bpsk", 11, 8)
    estimate = estimate_symbol_timing(waveform, 8)
    assert estimate.boundary_offset == 0
    assert estimate.quality == pytest.approx(1.0)


def test_result_is_frozen_dataclass():
    estimate = estimate_symbol_timing(_waveform("qpsk", 12, 8), 8)
    with pytest.raises(Exception):
        estimate.boundary_offset = 0
    fields = set(SymbolTimingEstimate.__dataclass_fields__)
    assert fields == {"boundary_offset", "quality"}


def test_min_quality_is_not_part_of_the_public_api():
    samples = _waveform("bpsk", 13, 8)
    signature = inspect.signature(estimate_symbol_timing)
    assert "min_quality" not in signature.parameters
    with pytest.raises(TypeError):
        estimate_symbol_timing(samples, 8, min_quality=0.0)


def test_constant_phase_invariance():
    waveform = _waveform("bpsk", 21, 8)
    received = waveform[3:]
    base = estimate_symbol_timing(received, 8)
    rotated = estimate_symbol_timing(apply_phase_offset(received, 0.8), 8)
    assert rotated.boundary_offset == base.boundary_offset
    np.testing.assert_allclose(rotated.quality, base.quality)


def test_positive_amplitude_invariance():
    waveform = _waveform("qpsk", 22, 8)
    received = waveform[5:]
    base = estimate_symbol_timing(received, 8)
    scaled = estimate_symbol_timing(received * 3.7, 8)
    assert scaled.boundary_offset == base.boundary_offset
    np.testing.assert_allclose(scaled.quality, base.quality)


@pytest.mark.parametrize("modulation", ["bpsk", "qpsk"])
@pytest.mark.parametrize("sps", [4, 8])
def test_20db_regression(modulation, sps):
    waveform = _waveform(modulation, 31, sps)
    for crop in range(sps):
        noisy = add_awgn(
            waveform, 20.0, rng=np.random.default_rng(80000 + crop + sps)
        )
        estimate = estimate_symbol_timing(noisy[crop:], sps)
        assert estimate.boundary_offset == (-crop) % sps
        assert estimate.quality > 0.0


@pytest.mark.parametrize("modulation", ["bpsk", "qpsk"])
@pytest.mark.parametrize("sps", [2, 4, 8, 16])
def test_one_clean_transition_is_estimable(modulation, sps):
    waveform = _one_transition(modulation, sps)
    n_trans = int(np.count_nonzero(np.abs(np.diff(waveform)) > 0.0))
    assert n_trans == 1
    for crop in range(sps):
        received = waveform[crop:]
        assert int(np.count_nonzero(np.abs(np.diff(received)) > 0.0)) == 1
        estimate = estimate_symbol_timing(received, sps)
        assert estimate.boundary_offset == (-crop) % sps
        assert estimate.quality == pytest.approx(1.0)


def test_grid_estimator_still_rejects_one_transition():
    """Known-SPS timing may use one transition; unknown-period grid search
    still cannot identify the period from a single impulse."""
    waveform = _one_transition("bpsk", 8)
    with pytest.raises(ValueError):
        estimate_rectangular_symbol_grid(waveform, FS)


def test_zero_transition_rejected():
    for bad in (
        np.full(128, 1.0 + 1.0j),
        np.zeros(128, dtype=np.complex128),
        bpsk_waveform(np.zeros(40, dtype=np.int64), 8),
        qpsk_waveform(np.zeros(80, dtype=np.int64), 8),
    ):
        with pytest.raises(ValueError, match="no usable symbol-transition"):
            estimate_symbol_timing(bad, 8)


def test_input_not_mutated():
    samples = _waveform("bpsk", 41, 8)
    before = samples.copy()
    estimate_symbol_timing(samples[2:], 8)
    np.testing.assert_array_equal(samples, before)


def test_real_bpsk_accepted():
    waveform = np.real(_waveform("bpsk", 42, 8))
    estimate = estimate_symbol_timing(waveform[3:], 8)
    assert estimate.boundary_offset == (-3) % 8


@pytest.mark.parametrize(
    "samples_per_symbol",
    [1, 0, -2, True, False, 2.5, "8", None],
)
def test_invalid_sps_raises(samples_per_symbol):
    with pytest.raises(ValueError):
        estimate_symbol_timing(_waveform("bpsk", 43, 8), samples_per_symbol)


def test_invalid_samples_raise():
    for bad in (
        np.array([]),
        np.array([1.0 + 0.0j]),
        np.ones((4, 4), dtype=np.complex128),
        np.array([1.0, np.nan, 2.0, 3.0]),
        np.array([1.0 + np.inf * 1j, 2.0 + 1j]),
        np.array(["a", "b"]),
    ):
        with pytest.raises(ValueError):
            estimate_symbol_timing(bad, 2)


def test_deterministic_output():
    samples = _waveform("qpsk", 44, 8)[2:]
    first = estimate_symbol_timing(samples, 8)
    second = estimate_symbol_timing(samples, 8)
    assert first == second
