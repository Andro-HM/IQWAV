"""Unit tests for blind occupied-band detection in iqwav.estimation."""

import numpy as np
import pytest

from iqwav.dsp import add_awgn
from iqwav.estimation import OccupiedBand, detect_occupied_bands
from iqwav.modulation import generate_iq_tone

FS = 1_000_000.0
N = 16384
NPERSEG = 2048


def _band_limited_signal(seed, center_hz, width_hz):
    rng = np.random.default_rng(seed)
    white = rng.standard_normal(N) + 1j * rng.standard_normal(N)
    spectrum = np.fft.fftshift(np.fft.fft(white))
    freqs = np.fft.fftshift(np.fft.fftfreq(N, d=1.0 / FS))
    spectrum[np.abs(freqs - center_hz) > width_hz / 2.0] = 0.0
    signal = np.fft.ifft(np.fft.ifftshift(spectrum))
    return signal / np.sqrt(np.mean(np.abs(signal) ** 2))


def _occupied_samples(center_hz, width_hz, seed_signal, seed_noise, snr_db=20.0):
    signal = _band_limited_signal(seed_signal, center_hz, width_hz)
    return add_awgn(signal, snr_db, rng=np.random.default_rng(seed_noise))


def test_band_fields_internally_consistent():
    samples = _occupied_samples(30_000.0, 20_000.0, seed_signal=1, seed_noise=2)
    bands, floor = detect_occupied_bands(samples, FS, nperseg=NPERSEG)
    assert len(bands) == 1
    band = bands[0]
    assert isinstance(band, OccupiedBand)
    assert band.lower_hz < band.peak_hz < band.upper_hz
    np.testing.assert_allclose(band.center_hz, (band.lower_hz + band.upper_hz) / 2.0)
    np.testing.assert_allclose(band.bandwidth_hz, band.upper_hz - band.lower_hz)
    np.testing.assert_allclose(band.peak_above_noise_db, band.peak_db - floor)
    assert band.bandwidth_hz > 0.0


def test_pure_noise_returns_no_bands():
    rng = np.random.default_rng(11)
    noise = rng.standard_normal(N) + 1j * rng.standard_normal(N)
    bands, floor = detect_occupied_bands(
        noise, FS, nperseg=NPERSEG, threshold_db=10.0
    )
    assert bands == []
    assert isinstance(floor, float)
    assert np.isfinite(floor)


def test_single_occupied_complex_band_recovered():
    samples = _occupied_samples(30_000.0, 20_000.0, seed_signal=1, seed_noise=2)
    bands, _ = detect_occupied_bands(samples, FS, nperseg=NPERSEG)
    assert len(bands) == 1
    band = bands[0]
    assert abs(band.center_hz - 30_000.0) <= 2000.0
    assert abs(band.bandwidth_hz - 20_000.0) <= 5000.0
    assert band.peak_above_noise_db > 10.0


def test_frequency_shifted_band_center():
    samples = _occupied_samples(100_000.0, 20_000.0, seed_signal=3, seed_noise=4)
    bands, _ = detect_occupied_bands(samples, FS, nperseg=NPERSEG)
    assert len(bands) == 1
    assert abs(bands[0].center_hz - 100_000.0) <= 2000.0


def test_two_separated_bands_not_merged():
    two = _band_limited_signal(5, -150_000.0, 15_000.0) + _band_limited_signal(
        6, 200_000.0, 15_000.0
    )
    samples = add_awgn(two, 20.0, rng=np.random.default_rng(7))
    bands, _ = detect_occupied_bands(samples, FS, nperseg=NPERSEG)
    assert len(bands) == 2
    assert abs(bands[0].center_hz + 150_000.0) <= 3000.0
    assert abs(bands[1].center_hz - 200_000.0) <= 3000.0
    assert bands[0].center_hz < bands[1].center_hz
    assert bands[0].upper_hz < bands[1].lower_hz


def test_higher_threshold_can_eliminate_band():
    samples = _occupied_samples(30_000.0, 20_000.0, seed_signal=1, seed_noise=2)
    bands_low, _ = detect_occupied_bands(
        samples, FS, nperseg=NPERSEG, threshold_db=6.0
    )
    bands_high, _ = detect_occupied_bands(
        samples, FS, nperseg=NPERSEG, threshold_db=200.0
    )
    assert len(bands_low) == 1
    assert bands_high == []


def test_min_bins_rejects_narrow_artifact():
    _, tone = generate_iq_tone(fs=FS, freq=-75_000.0, duration=N / FS, amplitude=1.0)
    samples = add_awgn(tone, 20.0, rng=np.random.default_rng(8))
    bands_default, _ = detect_occupied_bands(samples, FS, nperseg=NPERSEG)
    bands_rejected, _ = detect_occupied_bands(
        samples, FS, nperseg=NPERSEG, min_bins=12
    )
    assert len(bands_default) == 1
    assert bands_rejected == []


def test_real_valued_input_symmetric_bands():
    real = np.real(_band_limited_signal(9, 30_000.0, 20_000.0))
    real = real / np.sqrt(np.mean(np.abs(real) ** 2))
    samples = add_awgn(real, 20.0, rng=np.random.default_rng(10))
    bands, _ = detect_occupied_bands(samples, FS, nperseg=NPERSEG)
    assert len(bands) == 2
    assert abs(bands[0].center_hz + 30_000.0) <= 3000.0
    assert abs(bands[1].center_hz - 30_000.0) <= 3000.0


def test_zero_energy_input_raises():
    with pytest.raises(ValueError):
        detect_occupied_bands(np.zeros(1000), FS)


def test_invalid_samples_raise():
    for bad in (
        np.array([]),
        np.ones((4, 4)),
        np.array([1.0, np.nan] * 3),
        np.array([1.0 + np.inf * 1j, 2.0]),
        np.array(["a", "b"]),
        np.ones(1),
    ):
        with pytest.raises(ValueError):
            detect_occupied_bands(bad, FS)


@pytest.mark.parametrize(
    "fs",
    [0.0, -1000.0, float("nan"), float("inf"), True, 1.0 + 2.0j, "1000", np.array([1000.0])],
)
def test_invalid_fs_raises(fs):
    with pytest.raises(ValueError):
        detect_occupied_bands(np.random.default_rng(0).standard_normal(64), fs)


@pytest.mark.parametrize(
    "threshold_db",
    [0.0, -3.0, float("nan"), float("inf"), True, 5.0 + 1.0j],
)
def test_invalid_threshold_raises(threshold_db):
    samples = np.random.default_rng(0).standard_normal(64)
    with pytest.raises(ValueError):
        detect_occupied_bands(samples, FS, threshold_db=threshold_db)


@pytest.mark.parametrize("min_bins", [0, -1, 2.5, "3", True, False])
def test_invalid_min_bins_raises(min_bins):
    samples = np.random.default_rng(0).standard_normal(64)
    with pytest.raises(ValueError):
        detect_occupied_bands(samples, FS, min_bins=min_bins)


@pytest.mark.parametrize("nperseg", [0, -1, 2.5, "4"])
def test_invalid_nperseg_raises(nperseg):
    samples = np.random.default_rng(0).standard_normal(64)
    with pytest.raises(ValueError):
        detect_occupied_bands(samples, FS, nperseg=nperseg)
