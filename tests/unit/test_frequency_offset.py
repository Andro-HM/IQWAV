"""Unit tests for coarse frequency-offset estimation in iqwav.estimation."""

import numpy as np
import pytest

from iqwav.dsp import (
    add_awgn,
    apply_frequency_offset,
    apply_phase_offset,
)
from iqwav.estimation import FrequencyOffsetEstimate, estimate_frequency_offset
from iqwav.modulation import bpsk_waveform, qpsk_waveform

FS = 80_000.0


def _bits(seed, n):
    return np.random.default_rng(seed).integers(0, 2, n)


def _bpsk(seed, sps=8, n_symbols=4096):
    return bpsk_waveform(_bits(seed, n_symbols), sps)


def _qpsk(seed, sps=8, n_symbols=2048):
    return qpsk_waveform(_bits(seed, 2 * n_symbols), sps)


def test_estimate_field_consistency():
    samples = apply_frequency_offset(_bpsk(1), FS, 1000.0)
    estimate = estimate_frequency_offset(samples, FS)
    assert isinstance(estimate, FrequencyOffsetEstimate)
    np.testing.assert_allclose(
        estimate.frequency_offset_hz,
        FS * estimate.phase_increment_rad / (2.0 * np.pi),
    )
    assert np.isfinite(estimate.frequency_offset_hz)
    assert np.isfinite(estimate.coherence)
    assert estimate.coherence >= 0.05


def test_constant_complex_sequence_zero_offset():
    estimate = estimate_frequency_offset(np.full(4096, 1.0 + 0.5j), FS)
    np.testing.assert_allclose(estimate.frequency_offset_hz, 0.0, atol=1e-6)
    np.testing.assert_allclose(estimate.coherence, 1.0, atol=1e-9)


@pytest.mark.parametrize(
    "offset_hz", [-5000.0, -1000.0, 0.0, 1000.0, 5000.0]
)
def test_bpsk_cfo_recovery(offset_hz):
    samples = apply_frequency_offset(_bpsk(2), FS, offset_hz)
    estimate = estimate_frequency_offset(samples, FS)
    assert abs(estimate.frequency_offset_hz - offset_hz) <= 50.0


@pytest.mark.parametrize("offset_hz", [-1000.0, 1000.0])
def test_qpsk_cfo_recovery(offset_hz):
    samples = apply_frequency_offset(_qpsk(3), FS, offset_hz)
    estimate = estimate_frequency_offset(samples, FS)
    assert abs(estimate.frequency_offset_hz - offset_hz) <= 100.0


def test_sign_convention():
    positive = estimate_frequency_offset(
        apply_frequency_offset(_bpsk(5), FS, 2500.0), FS
    )
    negative = estimate_frequency_offset(
        apply_frequency_offset(_bpsk(6), FS, -2500.0), FS
    )
    assert positive.frequency_offset_hz > 0.0
    assert negative.frequency_offset_hz < 0.0
    np.testing.assert_allclose(
        positive.frequency_offset_hz, -negative.frequency_offset_hz, atol=50.0
    )


def test_phase_offset_invariance():
    base = apply_frequency_offset(_bpsk(7), FS, 1000.0)
    base_estimate = estimate_frequency_offset(base, FS)
    rotated_estimate = estimate_frequency_offset(
        apply_phase_offset(base, 0.9), FS
    )
    np.testing.assert_allclose(
        rotated_estimate.frequency_offset_hz,
        base_estimate.frequency_offset_hz,
        atol=1.0,
    )
    assert abs(rotated_estimate.frequency_offset_hz - 1000.0) <= 50.0


def test_amplitude_scaling_invariance():
    base = apply_frequency_offset(_bpsk(8), FS, 1000.0)
    scaled_estimate = estimate_frequency_offset(base * 3.7, FS)
    assert abs(scaled_estimate.frequency_offset_hz - 1000.0) <= 50.0


def test_start_offset_invariance():
    samples = apply_frequency_offset(_bpsk(9), FS, 1000.0)[5:]
    estimate = estimate_frequency_offset(samples, FS)
    assert abs(estimate.frequency_offset_hz - 1000.0) <= 50.0


@pytest.mark.parametrize(
    ("modulation", "snr_db"),
    [("bpsk", 20.0), ("bpsk", 10.0), ("qpsk", 20.0), ("qpsk", 10.0)],
)
def test_awgn_cfo_recovery(modulation, snr_db):
    waveform = _bpsk(10) if modulation == "bpsk" else _qpsk(11)
    samples = add_awgn(
        apply_frequency_offset(waveform, FS, 1000.0),
        snr_db,
        rng=np.random.default_rng(int(snr_db)),
    )
    estimate = estimate_frequency_offset(samples, FS)
    assert abs(estimate.frequency_offset_hz - 1000.0) <= 150.0


def test_low_coherence_rejected():
    rng = np.random.default_rng(42)
    noise = rng.standard_normal(8192) + 1j * rng.standard_normal(8192)
    with pytest.raises(ValueError):
        estimate_frequency_offset(noise, FS)


def test_min_coherence_zero_allows_diagnostic_estimate():
    rng = np.random.default_rng(42)
    noise = rng.standard_normal(8192) + 1j * rng.standard_normal(8192)
    estimate = estimate_frequency_offset(noise, FS, min_coherence=0.0)
    assert np.isfinite(estimate.frequency_offset_hz)
    assert np.isfinite(estimate.coherence)
    assert estimate.coherence < 0.05
    assert abs(estimate.frequency_offset_hz) < FS / 2.0


def test_invalid_samples_raise():
    for bad in (
        np.array([], dtype=np.complex128),
        np.ones(1, dtype=np.complex128),
        np.ones((4, 4), dtype=np.complex128),
        np.ones(64),
        np.array([1.0 + np.nan * 1j, 2.0 + 1j]),
        np.zeros(64, dtype=np.complex128),
    ):
        with pytest.raises(ValueError):
            estimate_frequency_offset(bad, FS)


@pytest.mark.parametrize(
    "fs",
    [0.0, -80_000.0, float("nan"), float("inf"), True, 1.0 + 2.0j],
)
def test_invalid_fs_raises(fs):
    with pytest.raises(ValueError):
        estimate_frequency_offset(np.full(64, 1.0 + 1j), fs)


@pytest.mark.parametrize(
    "min_coherence",
    [-0.1, 1.5, float("nan"), True],
)
def test_invalid_min_coherence_raises(min_coherence):
    with pytest.raises(ValueError):
        estimate_frequency_offset(
            np.full(64, 1.0 + 1j), FS, min_coherence=min_coherence
        )
