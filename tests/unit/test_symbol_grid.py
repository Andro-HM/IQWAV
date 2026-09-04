"""Unit tests for the rectangular symbol-grid estimator in iqwav.estimation."""

import numpy as np
import pytest

from iqwav.dsp import (
    add_awgn,
    apply_frequency_offset,
    apply_phase_offset,
)
from iqwav.estimation import (
    RectangularSymbolGridEstimate,
    estimate_rectangular_symbol_grid,
    estimate_symbol_rate,
)
from iqwav.modulation import bpsk_waveform, qpsk_waveform

FS = 96_000.0


def _bits(seed, n):
    return np.random.default_rng(seed).integers(0, 2, n)


def _bpsk(seed, sps, n_symbols=4000):
    return bpsk_waveform(_bits(seed, n_symbols), sps)


def _qpsk(seed, sps, n_symbols=4000):
    return qpsk_waveform(_bits(seed, 2 * n_symbols), sps)


def _grid(samples, **kwargs):
    return estimate_rectangular_symbol_grid(samples, FS, **kwargs)


@pytest.mark.parametrize("sps", [2, 4, 8, 16])
def test_clean_bpsk(sps):
    result = _grid(_bpsk(100 + sps, sps))
    assert isinstance(result, RectangularSymbolGridEstimate)
    assert result.samples_per_symbol == sps
    assert result.boundary_offset == 0
    assert result.quality == pytest.approx(1.0)
    assert result.concentration == pytest.approx(1.0)
    assert result.symbol_rate_hz == pytest.approx(FS / sps)
    assert result.symbol_count == pytest.approx(4000, abs=1)
    assert result.effective_transitions > 100.0
    assert result.searched_samples_per_symbol == (2, 64)


@pytest.mark.parametrize("sps", [3, 5, 12])
def test_clean_qpsk(sps):
    result = _grid(_qpsk(200 + sps, sps))
    assert result.samples_per_symbol == sps
    assert result.boundary_offset == 0
    assert result.quality == pytest.approx(1.0)
    assert result.symbol_rate_hz == pytest.approx(FS / sps)


def test_nonzero_crop_exact_boundary_offset():
    waveform = _bpsk(7, 8)
    result = _grid(waveform[3:])
    assert result.samples_per_symbol == 8
    assert result.boundary_offset == (-3) % 8
    assert result.boundary_offset == 5


@pytest.mark.parametrize(
    ("sps", "crop"),
    [(8, crop) for crop in range(8)] + [(5, crop) for crop in range(5)],
)
def test_all_boundary_offsets(sps, crop):
    waveform = _bpsk(7, sps)
    result = _grid(waveform[crop:])
    assert result.samples_per_symbol == sps
    assert result.boundary_offset == (-crop) % sps


def test_constant_phase_invariance():
    waveform = _bpsk(11, 8)
    base = _grid(waveform)
    rotated = _grid(apply_phase_offset(waveform, 0.7))
    assert rotated.samples_per_symbol == base.samples_per_symbol
    assert rotated.boundary_offset == base.boundary_offset
    np.testing.assert_allclose(rotated.quality, base.quality)


def test_amplitude_invariance():
    waveform = _bpsk(12, 8)
    base = _grid(waveform)
    scaled = _grid(waveform * 3.7)
    assert scaled.samples_per_symbol == base.samples_per_symbol
    assert scaled.boundary_offset == base.boundary_offset
    np.testing.assert_allclose(scaled.quality, base.quality)


@pytest.mark.parametrize(
    ("modulation", "snr_db"),
    [
        ("bpsk", 20.0),
        ("bpsk", 10.0),
        ("bpsk", 5.0),
        ("bpsk", 0.0),
        ("qpsk", 20.0),
        ("qpsk", 10.0),
        ("qpsk", 5.0),
        ("qpsk", 0.0),
    ],
)
def test_awgn_recovery(modulation, snr_db):
    waveform = _bpsk(21, 8) if modulation == "bpsk" else _qpsk(21, 8)
    noisy = add_awgn(
        waveform, snr_db, rng=np.random.default_rng(int(snr_db) * 10 + 21)
    )
    result = _grid(noisy)
    assert result.samples_per_symbol == 8
    assert result.boundary_offset == 0
    assert result.quality >= 0.02


@pytest.mark.parametrize("fraction", [0.05, -0.05])
def test_moderate_cfo_recovery(fraction):
    waveform = apply_frequency_offset(_bpsk(31, 8), FS, fraction * (FS / 8))
    result = _grid(waveform)
    assert result.samples_per_symbol == 8
    assert result.boundary_offset == 0
    assert result.quality > 0.5


@pytest.mark.parametrize(
    ("n_symbols", "sps"),
    [(32, 2), (32, 4), (64, 2), (64, 4)],
)
def test_short_but_valid_blocks(n_symbols, sps):
    waveform = _bpsk(41 + sps + n_symbols, sps, n_symbols)
    result = _grid(waveform)
    assert result.samples_per_symbol == sps
    assert result.boundary_offset == 0
    assert result.searched_samples_per_symbol[1] <= (waveform.size - 1) // 4


def test_all_identical_and_constant_rejected():
    for bad in (
        np.full(1024, 1.0 + 1.0j),
        np.zeros(1024),
        bpsk_waveform(np.zeros(400, dtype=np.int64), 8),
    ):
        with pytest.raises(ValueError):
            _grid(bad)


def test_single_transition_not_identifiable_rejected():
    bits = np.zeros(400, dtype=np.int64)
    bits[-1] = 1
    waveform = bpsk_waveform(bits, 8)
    with pytest.raises(ValueError):
        _grid(waveform)


def test_observable_period_is_multiple_of_true_sps():
    rng = np.random.default_rng(51)
    base = rng.integers(0, 2, 1000)
    runs = np.repeat(base, 4)
    waveform = bpsk_waveform(runs, 8)
    result = _grid(waveform)
    assert result.samples_per_symbol == 32
    assert result.quality == pytest.approx(1.0)


def test_out_of_range_observable_period_returns_in_range_divisor():
    rng = np.random.default_rng(52)
    base = rng.integers(0, 2, 500)
    runs = np.repeat(base, 8)
    waveform = bpsk_waveform(runs, 16)
    result = _grid(waveform)
    assert result.samples_per_symbol == 64
    assert 128 % result.samples_per_symbol == 0
    assert result.searched_samples_per_symbol == (2, 64)


def test_invalid_search_bounds_raise():
    waveform = _bpsk(61, 8)
    for min_sps in (1, True, 2.5, "2"):
        with pytest.raises(ValueError):
            _grid(waveform, min_sps=min_sps)
    with pytest.raises(ValueError):
        _grid(waveform, min_sps=8, max_sps=4)
    with pytest.raises(ValueError):
        _grid(waveform, max_sps=True)
    with pytest.raises(ValueError):
        _grid(waveform, max_sps=2.5)


@pytest.mark.parametrize(
    "fs",
    [0.0, -96_000.0, float("nan"), float("inf"), True, 1.0 + 2.0j],
)
def test_invalid_fs_raises(fs):
    with pytest.raises(ValueError):
        estimate_rectangular_symbol_grid(_bpsk(62, 8), fs)


def test_invalid_quality_parameters_raise():
    waveform = _bpsk(63, 8)
    for quality_ratio in (0.0, 1.5, float("nan"), True):
        with pytest.raises(ValueError):
            _grid(waveform, quality_ratio=quality_ratio)
    for min_quality in (-0.1, 1.0, float("nan"), True):
        with pytest.raises(ValueError):
            _grid(waveform, min_quality=min_quality)


def test_invalid_samples_raise():
    for bad in (
        np.array([]),
        np.ones((4, 4)),
        np.ones(8, dtype=np.complex128),
        np.array([1.0, np.nan, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]),
        np.array([1.0 + np.inf * 1j, 2.0 + 1j, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]),
        np.array(["a", "b", "c", "d", "e", "f", "g", "h", "i"]),
    ):
        with pytest.raises(ValueError):
            _grid(bad)


def test_input_not_mutated():
    waveform = _bpsk(64, 8)
    snapshot = waveform.copy()
    _grid(waveform)
    assert np.array_equal(waveform, snapshot)


def test_result_is_frozen_dataclass():
    result = _grid(_bpsk(65, 8))
    with pytest.raises(Exception):
        result.samples_per_symbol = 4


@pytest.mark.parametrize(
    ("modulation", "sps", "seed"),
    [("bpsk", 8, 61), ("qpsk", 12, 62)],
)
def test_agreement_with_hm_symbol_rate_estimator(modulation, sps, seed):
    waveform = _bpsk(seed, sps) if modulation == "bpsk" else _qpsk(seed, sps)
    hm_result = estimate_symbol_rate(waveform, FS)
    grid_result = _grid(waveform)
    assert grid_result.samples_per_symbol == hm_result.samples_per_symbol == sps
    np.testing.assert_allclose(
        grid_result.symbol_rate_hz, hm_result.symbol_rate_hz
    )
