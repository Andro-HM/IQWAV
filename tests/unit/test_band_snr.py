"""Unit tests for blind in-band SNR estimation in iqwav.estimation."""

import numpy as np
import pytest

from iqwav.dsp import add_awgn
from iqwav.estimation import (
    OccupiedBand,
    detect_occupied_bands,
    estimate_band_snr,
)

FS = 1_000_000.0
N = 16384
NPERSEG = 2048
DF = FS / NPERSEG
HALF_WIDTH = 40.0 * DF
IN_BAND_HZ = 81.0 * DF


def _band(lower_hz, upper_hz):
    return OccupiedBand(
        lower_hz=lower_hz,
        upper_hz=upper_hz,
        center_hz=(lower_hz + upper_hz) / 2.0,
        bandwidth_hz=upper_hz - lower_hz,
        peak_hz=(lower_hz + upper_hz) / 2.0,
        peak_db=0.0,
        peak_above_noise_db=0.0,
    )


BAND = _band(-HALF_WIDTH, HALF_WIDTH)
BAND_SHIFTED = _band(100_000.0 - HALF_WIDTH, 100_000.0 + HALF_WIDTH)
BAND_REAL = _band(30_000.0 - HALF_WIDTH, 30_000.0 + HALF_WIDTH)


def _unit_band_signal(seed, center_hz):
    rng = np.random.default_rng(seed)
    white = rng.standard_normal(N) + 1j * rng.standard_normal(N)
    spectrum = np.fft.fftshift(np.fft.fft(white))
    freqs = np.fft.fftshift(np.fft.fftfreq(N, d=1.0 / FS))
    spectrum[np.abs(freqs - center_hz) > HALF_WIDTH] = 0.0
    signal = np.fft.ifft(np.fft.ifftshift(spectrum))
    return signal / np.sqrt(np.mean(np.abs(signal) ** 2))


def _synthetic_samples(
    signal_power,
    inband_noise_power,
    center_hz=0.0,
    seed_signal=1,
    seed_noise=2,
):
    signal = _unit_band_signal(seed_signal, center_hz) * np.sqrt(signal_power)
    total_noise_power = inband_noise_power * FS / IN_BAND_HZ
    awgn_db = 10.0 * np.log10(signal_power / total_noise_power)
    return add_awgn(signal, awgn_db, rng=np.random.default_rng(seed_noise))


def test_estimate_fields_consistent():
    samples = _synthetic_samples(1.0, 0.1)
    estimate = estimate_band_snr(samples, FS, BAND, nperseg=NPERSEG)
    np.testing.assert_allclose(
        estimate.total_inband_power,
        estimate.signal_power + estimate.noise_power,
        rtol=1e-9,
    )
    np.testing.assert_allclose(
        estimate.noise_psd_db, 10.0 * np.log10(estimate.noise_psd), rtol=1e-9
    )
    for value in (
        estimate.signal_power,
        estimate.noise_power,
        estimate.total_inband_power,
        estimate.noise_psd,
    ):
        assert np.isfinite(value)
        assert value >= 0.0
    assert np.isfinite(estimate.snr_db)


@pytest.mark.parametrize("snr_db", [0.0, 5.0, 10.0, 20.0])
def test_known_snr_levels(snr_db):
    samples = _synthetic_samples(1.0, 10.0 ** (-snr_db / 10.0))
    estimate = estimate_band_snr(samples, FS, BAND, nperseg=NPERSEG)
    assert abs(estimate.snr_db - snr_db) <= 1.5


def test_shifted_band_same_snr():
    base = estimate_band_snr(
        _synthetic_samples(1.0, 0.1), FS, BAND, nperseg=NPERSEG
    )
    shifted = estimate_band_snr(
        _synthetic_samples(1.0, 0.1, center_hz=100_000.0, seed_signal=3, seed_noise=4),
        FS,
        BAND_SHIFTED,
        nperseg=NPERSEG,
    )
    assert abs(shifted.snr_db - base.snr_db) <= 1.0
    assert abs(shifted.snr_db - 10.0) <= 1.5


def test_signal_amplitude_direction():
    low = estimate_band_snr(
        _synthetic_samples(1.0, 0.1), FS, BAND, nperseg=NPERSEG
    )
    high = estimate_band_snr(
        _synthetic_samples(4.0, 0.1), FS, BAND, nperseg=NPERSEG
    )
    assert 4.0 <= high.snr_db - low.snr_db <= 8.0


def test_noise_amplitude_direction():
    quiet = estimate_band_snr(
        _synthetic_samples(1.0, 0.1), FS, BAND, nperseg=NPERSEG
    )
    loud = estimate_band_snr(
        _synthetic_samples(1.0, 0.4), FS, BAND, nperseg=NPERSEG
    )
    assert -8.0 <= loud.snr_db - quiet.snr_db <= -4.0


def test_noise_only_band_returns_zero_signal_and_minus_inf():
    rng = np.random.default_rng(21)
    white = rng.standard_normal(N) + 1j * rng.standard_normal(N)
    spectrum = np.fft.fftshift(np.fft.fft(white))
    freqs = np.fft.fftshift(np.fft.fftfreq(N, d=1.0 / FS))
    spectrum[np.abs(freqs) <= HALF_WIDTH] = 0.0
    samples = np.fft.ifft(np.fft.ifftshift(spectrum))
    estimate = estimate_band_snr(samples, FS, BAND, nperseg=NPERSEG)
    assert estimate.signal_power == 0.0
    assert estimate.snr_db == float("-inf")
    assert np.isfinite(estimate.noise_psd)
    assert estimate.noise_psd > 0.0


def test_real_valued_input():
    real = np.real(_unit_band_signal(31, 30_000.0))
    real = real / np.sqrt(np.mean(np.abs(real) ** 2)) * np.sqrt(2.0)
    total_noise = 0.1 * FS / IN_BAND_HZ
    samples = add_awgn(
        real, 10.0 * np.log10(2.0 / total_noise), rng=np.random.default_rng(32)
    )
    estimate = estimate_band_snr(samples, FS, BAND_REAL, nperseg=NPERSEG)
    assert abs(estimate.snr_db - 10.0) <= 2.0


def test_detector_composes_with_snr_estimator():
    samples = _synthetic_samples(
        1.0, 0.1, center_hz=30_000.0, seed_signal=41, seed_noise=42
    )
    bands, _ = detect_occupied_bands(samples, FS, nperseg=NPERSEG)
    assert len(bands) == 1
    assert abs(bands[0].center_hz - 30_000.0) <= 5000.0
    estimate = estimate_band_snr(samples, FS, bands[0], nperseg=NPERSEG)
    assert abs(estimate.snr_db - 10.0) <= 3.0


def test_invalid_samples_raise():
    for bad in (
        np.array([]),
        np.ones((4, 4)),
        np.array([1.0, np.nan] * 3),
        np.array([1.0 + np.inf * 1j, 2.0]),
        np.ones(1),
    ):
        with pytest.raises(ValueError):
            estimate_band_snr(bad, FS, BAND)


@pytest.mark.parametrize(
    "fs",
    [0.0, -1000.0, float("nan"), float("inf"), True, 1.0 + 2.0j],
)
def test_invalid_fs_raises(fs):
    samples = np.random.default_rng(0).standard_normal(64)
    with pytest.raises(ValueError):
        estimate_band_snr(samples, fs, BAND)


def test_invalid_band_raise():
    samples = np.random.default_rng(0).standard_normal(4096)
    for bad in ("not a band", {"lower_hz": 0.0, "upper_hz": 1.0}, (0.0, 1.0), None):
        with pytest.raises(ValueError):
            estimate_band_snr(samples, FS, bad, nperseg=NPERSEG)
    with pytest.raises(ValueError):
        estimate_band_snr(samples, FS, _band(100.0, 50.0), nperseg=NPERSEG)


def test_band_outside_frequency_range_raise():
    samples = np.random.default_rng(0).standard_normal(4096)
    with pytest.raises(ValueError):
        estimate_band_snr(samples, FS, _band(600_000.0, 700_000.0), nperseg=NPERSEG)
    with pytest.raises(ValueError):
        estimate_band_snr(samples, FS, _band(-700_000.0, -600_000.0), nperseg=NPERSEG)


def test_band_without_welch_bin_raise():
    samples = np.random.default_rng(0).standard_normal(4096)
    with pytest.raises(ValueError):
        estimate_band_snr(samples, FS, _band(10.0, 20.0), nperseg=NPERSEG)


def test_band_covering_whole_spectrum_raise():
    samples = np.random.default_rng(0).standard_normal(4096)
    with pytest.raises(ValueError):
        estimate_band_snr(samples, FS, _band(-1e9, 1e9), nperseg=NPERSEG)
