"""Unit tests for the blind symbol-rate baseline in iqwav.estimation."""

import numpy as np
import pytest

from iqwav.dsp import add_awgn, apply_phase_offset
from iqwav.estimation import SymbolRateEstimate, estimate_symbol_rate
from iqwav.modulation import bpsk_waveform, qpsk_waveform

FS = 80_000.0


def _bits(seed, n):
    return np.random.default_rng(seed).integers(0, 2, n)


def _bpsk(seed, sps, n_symbols=4096):
    return bpsk_waveform(_bits(seed, n_symbols), sps)


def _qpsk(seed, sps, n_symbols=2048):
    return qpsk_waveform(_bits(seed, 2 * n_symbols), sps)


def test_estimate_field_consistency():
    estimate = estimate_symbol_rate(_bpsk(1, 8), FS)
    assert isinstance(estimate, SymbolRateEstimate)
    assert 2 <= estimate.samples_per_symbol <= 64
    np.testing.assert_allclose(
        estimate.symbol_rate_hz, FS / estimate.samples_per_symbol
    )
    assert np.isfinite(estimate.score)
    assert estimate.score >= 0.10


@pytest.mark.parametrize("sps", [4, 8, 16])
def test_bpsk_sps_recovered(sps):
    estimate = estimate_symbol_rate(_bpsk(100 + sps, sps), FS)
    assert estimate.samples_per_symbol == sps


@pytest.mark.parametrize("sps", [4, 8, 16])
def test_qpsk_sps_recovered(sps):
    estimate = estimate_symbol_rate(_qpsk(200 + sps, sps), FS)
    assert estimate.samples_per_symbol == sps


def test_symbol_rate_hz_relation():
    estimate = estimate_symbol_rate(_bpsk(7, 8), 80_000.0)
    assert estimate.samples_per_symbol == 8
    np.testing.assert_allclose(estimate.symbol_rate_hz, 10_000.0)


@pytest.mark.parametrize(
    ("modulation", "snr_db"),
    [("bpsk", 20.0), ("bpsk", 10.0), ("qpsk", 20.0), ("qpsk", 10.0)],
)
def test_awgn_recovery(modulation, snr_db):
    waveform = _bpsk(7, 8) if modulation == "bpsk" else _qpsk(9, 8)
    noisy = add_awgn(waveform, snr_db, rng=np.random.default_rng(int(snr_db)))
    estimate = estimate_symbol_rate(noisy, FS)
    assert estimate.samples_per_symbol == 8


def test_constant_phase_rotation_invariance():
    estimate = estimate_symbol_rate(apply_phase_offset(_bpsk(7, 8), 0.7), FS)
    assert estimate.samples_per_symbol == 8


def test_amplitude_scaling_invariance():
    estimate = estimate_symbol_rate(_bpsk(7, 8) * 3.7, FS)
    assert estimate.samples_per_symbol == 8


def test_start_offset_recovered():
    estimate = estimate_symbol_rate(_bpsk(7, 8)[3:], FS)
    assert estimate.samples_per_symbol == 8


def test_constant_waveform_raises():
    with pytest.raises(ValueError):
        estimate_symbol_rate(np.full(1024, 1.0 + 1.0j), FS)
    with pytest.raises(ValueError):
        estimate_symbol_rate(np.ones(1024), FS)


def test_unrealistic_min_score_fails_cleanly():
    with pytest.raises(ValueError):
        estimate_symbol_rate(_bpsk(7, 8), FS, min_score=0.99)


def test_search_range_controls_candidates():
    waveform = _bpsk(7, 8)
    inside = estimate_symbol_rate(waveform, FS, min_sps=4, max_sps=16)
    assert inside.samples_per_symbol == 8
    with pytest.raises(ValueError):
        estimate_symbol_rate(waveform, FS, min_sps=2, max_sps=7)
    excluded = estimate_symbol_rate(waveform, FS, min_sps=10, max_sps=64)
    assert excluded.samples_per_symbol != 8


def test_invalid_samples_raise():
    for bad in (np.array([]), np.ones((4, 4)), np.array([1.0, np.nan] * 3), np.ones(1)):
        with pytest.raises(ValueError):
            estimate_symbol_rate(bad, FS)


@pytest.mark.parametrize(
    "fs",
    [0.0, -80_000.0, float("nan"), float("inf"), True, 1.0 + 2.0j],
)
def test_invalid_fs_raises(fs):
    with pytest.raises(ValueError):
        estimate_symbol_rate(np.ones(256), fs)


@pytest.mark.parametrize("min_sps", [1, 0, True, 2.5, "4"])
def test_invalid_min_sps_raises(min_sps):
    with pytest.raises(ValueError):
        estimate_symbol_rate(np.ones(256), FS, min_sps=min_sps)


def test_invalid_max_sps_raises():
    waveform = np.ones(256)
    with pytest.raises(ValueError):
        estimate_symbol_rate(waveform, FS, min_sps=8, max_sps=4)
    with pytest.raises(ValueError):
        estimate_symbol_rate(waveform, FS, max_sps=True)
    with pytest.raises(ValueError):
        estimate_symbol_rate(waveform, FS, max_sps=2.5)


@pytest.mark.parametrize("min_score", [0.0, -0.5, 1.5, float("nan"), True])
def test_invalid_min_score_raises(min_score):
    with pytest.raises(ValueError):
        estimate_symbol_rate(np.ones(256), FS, min_score=min_score)


def test_insufficient_samples_for_max_sps_raise():
    short = np.random.default_rng(0).standard_normal(20)
    with pytest.raises(ValueError):
        estimate_symbol_rate(short, FS, max_sps=64)
